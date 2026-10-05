"""Public entry point for the existing registered stock research workflows."""
import argparse
from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

from expertflow.compiler.pipeline import EnvironmentBlocked
from expertflow.compiler.preflight import file_sha256
from expertflow.compiler.schema import canonical_sha256
from .readers import reuse_readers

DRIVERS = {
    'reference': ('benchmark_compiler_stock_reference.py', {'generate', 'run', 'validate'}),
    'product': ('benchmark_compiler_refinement.py', {'run'}),
    'search': ('benchmark_compiler_stock_search.py', {'generate', 'run', 'validate', 'execute'}),
    'utility': ('benchmark_compiler_stock_utility.py', {'validate'}),
    'repeatability': ('benchmark_compiler_stock_repeatability.py', {'validate'}),
}


def _check_project(path):
    root = Path(path).resolve()
    if not (root/'pyproject.toml').is_file() or not (root/'scripts/benchmark_compiler_stock_search.py').is_file():
        raise EnvironmentBlocked('stock workflows require a research checkout; supply --project PATH')
    # A wheel may run against an explicit checkout, but it must execute the
    # same compiler bytes that the driver's source manifest will describe.
    import expertflow.compiler
    actual = Path(expertflow.compiler.__file__).parent
    declared = root/'src/expertflow/compiler'
    files = {p.relative_to(actual) for p in actual.rglob('*.py')}
    if files != {p.relative_to(declared) for p in declared.rglob('*.py')} or any(
            file_sha256(actual/p) != file_sha256(declared/p) for p in files):
        raise ValueError('installed compiler source differs from project source; use the matching checkout/package')
    return root


@contextmanager
def _load_driver(project, filename):
    """Use original project paths and restore process state after delegation."""
    project = Path(project).resolve()
    previous_cwd, previous_path = Path.cwd(), sys.path[:]
    def driver_module(name):
        return name == 'scripts' or name.startswith('scripts.') or name in {
            'benchmark_compiler_stock_utility', '_expertflow_stock_driver'}
    previous_modules = {k:v for k,v in sys.modules.items() if driver_module(k)}
    try:
        for name in previous_modules:
            del sys.modules[name]
        os.chdir(project)
        sys.path[:0] = [str(project/'scripts'), str(project)]
        spec = importlib.util.spec_from_file_location('_expertflow_stock_driver', project/'scripts'/filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        os.chdir(previous_cwd)
        sys.path[:] = previous_path
        for name in list(sys.modules):
            if driver_module(name):
                del sys.modules[name]
        sys.modules.update(previous_modules)


def _receipt_path(workflow, action, driver_args):
    if workflow == 'reference' and action == 'validate':
        flag, name = '--reference-dir', 'reference-receipt.json'
    elif workflow == 'search' and action in ('validate', 'execute'):
        flag, name = '--recommendation', 'search-receipt.json'
    else:
        return None
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(flag, type=Path)
    args, _ = parser.parse_known_args(driver_args)
    directory = getattr(args, flag[2:].replace('-', '_'))
    return directory/name if directory is not None else None


def _decision(result, code, action, *, reconstructed=False, report=None):
    negative = {'NO-UTILITY-GAIN','VARIANCE-STOP','MANUAL-BASELINE-STOP','TUNING-COST-STOP',
        'PRODUCT-VALIDATION-STOP','CONSUMER-VALIDATION-STOP','REPEATABILITY-STOP',
        'TRANSFER-VALIDATION-STOP','VALIDATION-STOP'}
    verified = (code == 0 and action != 'generate') or (
        reconstructed and code == 2 and result['status'] in negative)
    decision = {
        'evidence_verified': verified,
        'utility_gain_established': verified and result['status'] in {
            'PASS-STOCK-UTILITY-PRODUCT', 'PASS-STOCK-REPEATABILITY-TRANSFER'},
        'global_optimum_established': False,
        'serving_throughput_established': False,
    }
    report_path = result.get('report')
    if verified and (report is not None or report_path and Path(report_path).is_file()):
        if report is None:
            report = json.loads(Path(report_path).read_text(encoding='utf-8'))
        manifest = report.get('manifest', report.get('frozen', {}))
        candidate = manifest.get('candidate') or manifest.get('source_plan', {}).get('candidate') or (
            manifest.get('candidates', {}).get(report.get('recommended_id') or
                manifest.get('default_id') or manifest.get('incumbent_id'), {}))
        identities = manifest.get('identities') or candidate.get('identities', {})
        coverage = {key:manifest[key] for key in (
            'protocol_version', 'protocol_scope', 'default_controls', 'space',
            'maximum_native_processes') if key in manifest}
        coverage['host_environment_sha256'] = canonical_sha256(manifest.get('host_environment', {}))
        coverage['inputs'] = {}
        for name in ('inputs', 'main_inputs', 'transfer_inputs'):
            snapshot = manifest.get(name)
            if snapshot and 'model' in snapshot:
                workload = snapshot.get('workload') or identities.get('workload')
                coverage['inputs'][name] = {
                    'family': snapshot['model']['family'], 'quantization': snapshot['model']['quantization'],
                    'weights_sha256': snapshot['model']['identity']['sha256'],
                    'model_ir_sha256': canonical_sha256(snapshot['model']),
                    'hardware_sha256': canonical_sha256(snapshot['hardware']),
                    'runtime_sha256': canonical_sha256(snapshot['stock']),
                }
                if workload is not None:
                    coverage['inputs'][name]['workload_sha256'] = canonical_sha256(workload)
        if manifest.get('identities'):
            coverage['inputs']['source_plan'] = {
                'model_ir_sha256':identities['model_sha256'],
                'hardware_sha256':identities['hardware_sha256'],
                'runtime_sha256':identities['runtime_sha256'],
                'workload_sha256':identities['workload_sha256'],
            }
        settings = manifest.get('settings') or candidate.get('settings')
        if settings is not None:
            coverage['settings'] = settings
        if candidate.get('candidate_id'):
            coverage['candidate_id'] = candidate['candidate_id']
        decision['coverage'] = coverage
        decision['cost'] = {'retained_attempts': len(report.get('attempts', report.get('outcomes', [])))}
        if 'collector_elapsed_ns' in report:
            decision['cost']['collector_seconds'] = report['collector_elapsed_ns']/1e9
        if 'statistics' in report:
            decision['statistics'] = report['statistics']
            if report.get('automatic_id') and manifest.get('default_id'):
                decision['selected_default'] = report['automatic_id'] == manifest['default_id']
            rows = report.get('rows', [])
            if rows and all('run_wall_seconds' in row for row in rows):
                decision['cost']['utility_rows_native_phase_seconds'] = sum(row['run_wall_seconds'] for row in rows)
                for label in ('automatic-', 'manual-'):
                    decision['cost'][label.rstrip('-')+'_search_native_phase_seconds'] = sum(
                        row['run_wall_seconds'] for row in rows if row['label'].startswith(label)
                        and '-pair-' not in row['label'])
        elif result['status'] == 'PASS-STOCK-REPEATABILITY-TRANSFER':
            transfer = Path(manifest['experiment_root'])/'transfer/utility/report.json'
            decision['statistics'] = {'transfer_utility':json.loads(transfer.read_text())['statistics'],
                'main_blocks': [{k:b[k] for k in ('status', 'geometric_change_pct', 'ci90_pct') if k in b}
                    for b in report['blocks']]}
        else:
            decision['statistics'] = {key:report[key] for key in (
                'mean_tps','cv_pct','geometric_change_pct','ci90_pct','ci95_pct',
                'direct_cv_pct','sealed_cv_pct','one_sided95_lower_pct') if key in report}
        if report.get('recommended_id') and manifest.get('incumbent_id'):
            decision['retained_incumbent'] = report['recommended_id'] == manifest['incumbent_id']
            decision['recommended_id'] = report['recommended_id']
        decision['scope_note'] = 'Only the reported model/workload/runtime/host and controls are covered.'
    return decision


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='expertflow stock', add_help=False, allow_abbrev=False,
        description='Registered stock workflows in a matching research checkout. Closed studies are validation-only.')
    parser.add_argument('--project', type=Path, default=Path.cwd(), help='research checkout; relative driver paths resolve here')
    parser.add_argument('workflow', nargs='?', choices=(*DRIVERS, 'coverage'))
    parser.add_argument('action', nargs='?', help='generate, run, validate or execute as supported by the workflow')
    args, driver_args = parser.parse_known_args(argv)
    if args.workflow is None or args.action is None:
        parser.print_help()
        return 0 if not argv or any(a in ('--help', '-h') for a in argv) else 2
    result, code, validation = {}, 2, None
    try:
        if any(a.startswith('--') and any(flag.startswith(a.split('=', 1)[0])
               for flag in ('--action', '--experiment')) for a in driver_args):
            raise ValueError('action/experiment override flags are not allowed; select the public workflow/action')
        if args.workflow == 'coverage':
            from .coverage import inspect_registration
            return inspect_registration(args.action, driver_args, args.project)
        filename, actions = DRIVERS[args.workflow]
        if args.action not in actions:
            if args.workflow in ('utility', 'repeatability'):
                raise ValueError('this study is closed; only validate is exposed, with no new native calls')
            raise ValueError('unsupported action for this workflow: ' + args.action)
        project = _check_project(args.project)
        forwarded = (['--experiment', 'stock-product'] if args.workflow == 'product'
                     else ['--action', args.action]) + driver_args
        with _load_driver(project, filename) as driver:
            if any(a in ('--help', '-h') for a in driver_args):
                return driver.main(forwarded)
            receipt_path = _receipt_path(args.workflow, args.action, driver_args)
            receipt_bytes = receipt_path.read_bytes() if receipt_path else None
            captured = io.StringIO()
            modules = [driver]
            if hasattr(driver, 'utility'):
                modules.append(driver.utility)
            if args.action == 'validate':
                with reuse_readers(*modules) as readers, redirect_stdout(captured):
                    code = driver.main(forwarded)
                    validation = {'cache_lifetime':'this invocation', 'database_readers':readers.database_count,
                        'model_digest_cache_entries':readers.model_verification_count,
                        'artifact_and_record_checks':'repeated normally', 'native_calls':0}
            else:
                with redirect_stdout(captured):
                    code = driver.main(forwarded)
            result = json.loads(captured.getvalue())
            report = None
            if code == 0 and receipt_path is not None:
                if receipt_path.read_bytes() != receipt_bytes:
                    raise ValueError('validated receipt changed while constructing public decision scope')
                report = json.loads(receipt_bytes)['experiment']
            result['decision'] = _decision(result, code, args.action,
                reconstructed=args.action == 'validate' and args.workflow in ('utility','repeatability'),
                report=report)
    except (EnvironmentBlocked, RuntimeError, subprocess.TimeoutExpired) as error:
        result, code = {'status':'ENVIRONMENT-BLOCKED', 'reason':str(error)}, 3
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        result, code = {'status':'IDENTITY-STOP', 'reason':str(error)}, 2
    if 'decision' not in result:
        result['decision'] = _decision(result, code, args.action)
    result.update(workflow=args.workflow, action=args.action)
    if validation is not None:
        result['validation'] = validation
    print(json.dumps(result, indent=2, sort_keys=True))
    return code
