"""Prepare by default, explicitly collect, or validate a fresh stock reference."""

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess

from expertflow.compiler.diagnostics import DiagnosticSampler
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import CompilationRequest, EnvironmentBlocked, atomic_json, load_compiler_inputs
from expertflow.compiler.preflight import capture_host_environment
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.stock_reference import prepare_reference, execute_stock_reference, load_reference_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('generate', 'run', 'validate'), default='generate')
    for name in ('descriptor', 'inventory', 'hardware', 'workload', 'runtime-identity', 'source-repository'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--manifest-output', type=Path)
    parser.add_argument('--evidence-db', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    args = parser.parse_args(argv)
    sampler = base = None
    try:
        if args.action == 'validate':
            if args.evidence_db is None or not args.evidence_db.is_file() or args.reference_dir is None:
                raise ValueError('validate requires existing --evidence-db and --reference-dir')
            output = args.reference_dir.parent
        else:
            if args.output_dir is None or args.output_dir.exists():
                raise ValueError('fresh --output-dir required; no retry/resume')
            output = args.output_dir
        if args.action == 'run' and (args.evidence_db is None or args.evidence_db.exists()):
            raise ValueError('run requires fresh --evidence-db')
        if args.action == 'generate' and (args.manifest_output is None or args.manifest_output.exists()):
            raise ValueError('generate requires fresh --manifest-output')
        if args.action == 'generate' and args.manifest_output.resolve().is_relative_to(output.resolve()):
            raise ValueError('preview must remain outside future native output directory')
        request = CompilationRequest(args.descriptor, args.inventory, args.hardware, args.workload,
            args.runtime_identity, (), args.evidence_db or output / 'compiler.sqlite3', output)
        inputs = load_compiler_inputs(request, live=True)
        host = capture_host_environment()
        if args.action == 'generate':
            manifest = prepare_reference(inputs, output, host_environment=host, source_repository=args.source_repository)
            atomic_json(args.manifest_output, manifest)
            result, code = {'status': 'GENERATED-REFERENCE-MANIFEST',
                'manifest_sha256': manifest['manifest_sha256'], 'manifest': str(args.manifest_output), 'native_samples': 0}, 0
        elif args.action == 'validate':
            plan = load_reference_plan(args.reference_dir, EvidenceStore(args.evidence_db),
                identities=inputs.identities(inputs.stock), host_environment=host)
            result, code = {'status': 'VALIDATED-STOCK-REFERENCE', 'plan_sha256': plan.plan_sha256,
                'product_accepted': False}, 0
        else:
            store = EvidenceStore(args.evidence_db)
            base = WindowsGpuMemorySampler(inputs.hardware.gpu_uuid)
            sampler = DiagnosticSampler(base)
            runner = ServerMeasurementRunner(store, memory_sampler=sampler)
            report = execute_stock_reference(inputs, store, runner, output,
                source_repository=args.source_repository)
            result = {'status': report['status'], 'reason': report.get('reason'),
                'native_outcomes': len(report['outcomes']), 'report': str(output / 'report.json'),
                'mean_tps': report.get('mean_tps'), 'cv_pct': report.get('cv_pct'), 'product_accepted': False}
            code = 0 if report['status'] == 'REFERENCE-STABLE' else 3 if report['status'] == 'ENVIRONMENT-BLOCKED' else 2
    except (EnvironmentBlocked, RuntimeError, subprocess.TimeoutExpired) as error:
        result, code = {'status': 'ENVIRONMENT-BLOCKED', 'reason': str(error)}, 3
    except (ValueError, OSError, KeyError, TypeError, sqlite3.Error) as error:
        result, code = {'status': 'IDENTITY-STOP', 'reason': str(error)}, 2
    finally:
        if sampler:
            sampler.close()
        elif base:
            base.close()
    print(json.dumps(result))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
