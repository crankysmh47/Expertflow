"""Readable archived stock results and delegated read-only qualification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
CATALOGUE = PROJECT/'docs/evidence/stock-followthrough-20261006/catalogue.json'
CATALOGUE_SHA256 = '4dfeaa11fe240014310b548e3949ad5ef36eeef7d519e991f20eb694e66d04f8'
STATUSES = {'q6': 'PASS-STOCK-REPEATABILITY-TRANSFER', 'wider': 'COMPLETE-STOCK-COVERAGE'}
COUNTS = {'q6': 148, 'wider': 344}
CASES = ('gemma4-q4-prose', 'gemma4-q4-code', 'granite-q6-prose', 'granite-q6-code')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_json(path, expected):
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('evidence digest mismatch: '+str(path))
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError('evidence JSON object required: '+str(path))
    return value


def load(project, catalogue):
    project = Path(project).resolve()
    catalogue = Path(catalogue).resolve()
    catalog = checked_json(catalogue, CATALOGUE_SHA256)
    if (catalog.get('schema_version') != 1 or not isinstance(catalog.get('studies'), dict)
            or set(catalog['studies']) != set(STATUSES)):
        raise ValueError('unsupported evidence catalogue')
    reports, roots = {}, {}
    for name, entry in catalog['studies'].items():
        if not isinstance(entry, dict):
            raise ValueError('study catalogue entry object required')
        path = Path(entry['report'])
        if not path.is_absolute():
            path = project/path
        path = path.resolve()
        if path.name != 'report.json':
            raise ValueError('authoritative report.json required')
        reports[name] = checked_json(path, entry['sha256'])
        roots[name] = path.parent
        if Path(reports[name]['manifest']['experiment_root']).resolve() != roots[name]:
            raise ValueError('catalogue root differs from frozen study root: '+name)
        if reports[name].get('status') != STATUSES[name]:
            raise ValueError('unexpected study status: '+name)
    if roots['q6'] == roots['wider'] or any(a in b.parents for a, b in (
            (roots['q6'], roots['wider']), (roots['wider'], roots['q6']))):
        raise ValueError('separate native study roots required')
    attempts = reports['q6']['attempts']
    if len(attempts) != 148 or any(a['status'] != 'measured' for a in attempts):
        raise ValueError('original native attempt inventory mismatch')
    wider = reports['wider']
    if wider['attempts'] != 344 or wider['native_processes'] != 344:
        raise ValueError('wider native attempt inventory mismatch')
    reports['transfer'] = checked_json(roots['q6']/'transfer/utility/report.json',
                                      reports['q6']['transfer']['report_sha256'])
    if reports['transfer']['status'] != 'PASS-STOCK-UTILITY-PRODUCT':
        raise ValueError('unexpected transfer status')
    return catalog, reports, roots


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError('finite numeric statistics required')
    return value


def interval(values):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError('two-sided interval required')
    lower, upper = map(finite, values)
    if lower > upper:
        raise ValueError('reversed interval')
    return [lower, upper]


def row(name, status, stats, **extra):
    if stats['automatic_evaluations'] != 18 or stats['manual_evaluations'] != 18:
        raise ValueError('matched 18/18 evaluation budgets required')
    return {'case_id': name, 'status': status,
            'gain_pct': finite(stats['gain']['geometric_change_pct']),
            'gain_ci95_pct': interval(stats['gain']['ci95_pct']),
            'manual_ci90_pct': interval(stats['manual_equivalence']['ci90_pct']),
            'automatic_evaluations': 18, 'manual_evaluations': 18, **extra}


def snapshot(project=PROJECT, catalogue=CATALOGUE):
    _, reports, _ = load(project, catalogue)
    cases = reports['wider']['cases']
    if len(cases) != 4 or tuple(c['case_id'] for c in cases) != CASES:
        raise ValueError('four separate registered wider cases required')
    transfer = reports['transfer']
    rows = [row('gemma4-q6-heldout', 'PASS-STOCK-UTILITY-PRODUCT', transfer['statistics'],
                selected_default=transfer['automatic_id'] == transfer['manifest']['default_id'])]
    for case in cases:
        if case['status'] != 'NO-UTILITY-GAIN' or case['native_processes'] != 86:
            raise ValueError('unexpected closed wider case outcome')
        if not isinstance(case['selected_default'], bool):
            raise ValueError('explicit default-selection boolean required')
        rows.append(row(case['case_id'], case['status'], case['statistics'],
                        selected_default=case['selected_default']))
    return {'status': 'ARCHIVED-STOCK-RESULTS', 'fresh_validation': False,
            'native_calls': dict(COUNTS), 'cases': rows,
            'scope': 'Pinned Q6 workloads and four Q4/Granite cases on one Windows/NVIDIA host/build. '
                     'Archived report digests checked; raw artifacts, model/runtime identity and current host '
                     'require the delegated fresh validators.',
            'global_optimum_established': False, 'serving_throughput_established': False,
            'new_placement_gain_established': False}


def inventory(project, catalogue):
    _, reports, roots = load(project, catalogue)
    state = {str(Path(catalogue).resolve()): digest(catalogue)}
    for name, root in roots.items():
        state[str(root/'report.json')] = digest(root/'report.json')
        for path, expected in reports[name]['manifest']['source_files'].items():
            file = Path(path)
            if not file.is_absolute():
                file = Path(project)/file
            actual = digest(file)
            if actual != expected:
                raise ValueError('measured source changed: '+str(file))
            state[str(file.resolve())] = actual
        starts = list(root.rglob('run-start.json'))
        if len(starts) != COUNTS[name]:
            raise ValueError('native start inventory changed: '+name)
        state.update({str(path.resolve()): digest(path) for path in starts})
    transfer = roots['q6']/'transfer/utility/report.json'
    state[str(transfer)] = digest(transfer)
    return state


def verify(project, catalogue, output):
    project, catalogue, output = map(lambda path: Path(path).resolve(), (project, catalogue, output))
    _, reports, roots = load(project, catalogue)
    if any(output == root or root in output.parents for root in roots.values()):
        raise ValueError('verification output must be outside native study roots')
    if output.exists():
        raise ValueError('fresh verification output required; no overwrite or resume')
    before = inventory(project, catalogue)
    archived = snapshot(project, catalogue)
    output.mkdir(parents=True)
    result = {'status': 'VALIDATION-STOP', 'fresh_validation': False,
              'started_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': [],
              'archived_results': archived, 'additional_native_calls': None}
    started = time.perf_counter()
    try:
        for name, workflow in (('q6', 'repeatability'), ('wider', 'coverage')):
            command = [sys.executable, '-m', 'expertflow.cli.main', 'stock', '--project',
                       str(project), workflow, 'validate']
            if name == 'q6':
                command += ['--output-dir', str(roots[name])]
            else:
                command += ['--registration', str(reports[name]['manifest']['registration'])]
            check_started = time.perf_counter()
            stdout, stderr = output/(name+'-stdout.json'), output/(name+'-stderr.log')
            try:
                completed = subprocess.run(command, cwd=project, capture_output=True,
                                           text=True, encoding='utf-8', timeout=3600)
            except subprocess.TimeoutExpired as error:
                for path, partial in ((stdout, error.stdout), (stderr, error.stderr)):
                    path.write_bytes(partial.encode('utf-8') if isinstance(partial, str) else partial or b'')
                result['checks'].append({'study': name, 'command': command, 'exit_code': None,
                    'timed_out': True, 'wall_seconds': time.perf_counter()-check_started,
                    'stdout_sha256': digest(stdout), 'stderr_sha256': digest(stderr)})
                raise
            stdout.write_text(completed.stdout, encoding='utf-8', newline='\n')
            stderr.write_text(completed.stderr, encoding='utf-8', newline='\n')
            check = {'study': name, 'command': command, 'exit_code': completed.returncode,
                     'wall_seconds': time.perf_counter()-check_started,
                     'stdout_sha256': digest(stdout), 'stderr_sha256': digest(stderr)}
            result['checks'].append(check)
            response = json.loads(completed.stdout)
            if not isinstance(response, dict) or not isinstance(response.get('decision'), dict):
                raise ValueError('public validator and decision JSON objects required')
            check['status'] = response.get('status')
            if (completed.returncode != 0 or check['status'] != STATUSES[name]
                    or Path(response.get('report', '')).resolve() != roots[name]/'report.json'
                    or response.get('decision', {}).get('evidence_verified') is not True):
                raise ValueError('public validation did not verify '+name)
        after = inventory(project, catalogue)
        if before != after:
            raise ValueError('evidence/source/native-start inventory changed during validation')
        result.update(status='PASS-LOCAL-STOCK-QUALIFICATION', fresh_validation=True,
                      additional_native_calls=0,
                      integrity_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest())
    except (OSError, ValueError, TypeError, KeyError, subprocess.TimeoutExpired) as error:
        result['reason'] = str(error)
    result['wall_seconds'] = time.perf_counter()-started
    result['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('status', 'verify'))
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--catalogue', type=Path, default=CATALOGUE)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.action == 'verify':
            if args.output_dir is None:
                raise ValueError('verify requires --output-dir with a fresh directory')
            result = verify(args.project, args.catalogue, args.output_dir)
        else:
            if args.output_dir is not None:
                raise ValueError('status does not write outputs')
            result = snapshot(args.project, args.catalogue)
        if args.json or args.action == 'verify':
            print(json.dumps(result, indent=2))
        else:
            print('Archived stock results (fresh validation not performed):')
            print('Retained native calls: Q6 study 148; wider study 344 (86 per wider case).')
            for case in result['cases']:
                low, high = case['gain_ci95_pct']
                print(f"  {case['case_id']}: {case['status']}; gain {case['gain_pct']:+.2f}% "
                      f"(CI95 [{low:+.2f}%, {high:+.2f}%])")
                manual_low, manual_high = case['manual_ci90_pct']
                print(f"    manual CI90 [{manual_low:+.2f}%, {manual_high:+.2f}%]; "
                      f"18/18 evaluations; selected default: {case['selected_default']}")
            print(result['scope'])
            print('Use verify with a fresh --output-dir for local read-only reconstruction.')
        return 0 if result['status'] != 'VALIDATION-STOP' else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'IDENTITY-STOP', 'reason': str(error), 'fresh_validation': False}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
