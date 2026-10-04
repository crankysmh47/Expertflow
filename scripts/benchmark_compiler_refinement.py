"""Run the predeclared twenty-process A/A gate, without changing old acceptance."""

import argparse
import json
from pathlib import Path

from expertflow.compiler.diagnostics import DiagnosticSampler
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import CompilationRequest, EnvironmentBlocked, load_compiler_inputs
from expertflow.compiler.refinement import execute_pairs, execute_thread_pairs
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.stock_validation import execute_stock_product


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', choices=('aa', 'threads8', 'stock-product'), default='aa')
    parser.add_argument('--aa-report', type=Path)
    parser.add_argument('--aa-evidence-db', type=Path)
    parser.add_argument('--descriptor', type=Path, default=Path('configs/compiler/gemma4-q6-model.json'))
    parser.add_argument('--inventory', type=Path, default=Path('docs/evidence/q6-download/tensor-inventory.json'))
    parser.add_argument('--hardware', type=Path, default=Path('docs/evidence/compiler-phase3/inputs/hardware.json'))
    parser.add_argument('--workload', type=Path, default=Path('configs/compiler/gemma4-q6-single-request.json'))
    parser.add_argument('--runtime-identity', type=Path, default=Path('docs/evidence/compiler-phase3/inputs/runtime-identity.json'))
    parser.add_argument('--source-plan', type=Path, default=Path('docs/evidence/compiler-phase3/execution-plan.pending.json'))
    parser.add_argument('--source-evidence-db', type=Path, default=Path('C:/models/expertflow/runs/compiler-phase3-20261003-reviewed/compiler.sqlite3'))
    parser.add_argument('--evidence-db', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    sampler = None
    base_sampler = None
    try:
        source_db = args.aa_evidence_db if args.experiment == 'threads8' else args.source_evidence_db
        if source_db is None or (args.experiment == 'threads8' and (args.aa_report is None or not args.aa_report.is_file())):
            raise ValueError('threads8 requires --aa-report and --aa-evidence-db')
        if not source_db.is_file():
            raise EnvironmentBlocked('original source evidence database unavailable')
        if args.evidence_db.exists() or args.output_dir.exists():
            raise ValueError('fresh output directory and database required; no resuming or optional retries')
        request = CompilationRequest(args.descriptor, args.inventory, args.hardware, args.workload,
            args.runtime_identity, (), args.evidence_db, args.output_dir)
        inputs = load_compiler_inputs(request, live=True)
        source = EvidenceStore(source_db)
        target = EvidenceStore(args.evidence_db)
        base_sampler = WindowsGpuMemorySampler(inputs.hardware.gpu_uuid)
        sampler = DiagnosticSampler(base_sampler)
        runner = ServerMeasurementRunner(target, memory_sampler=sampler)
        if args.experiment == 'aa':
            report = execute_pairs(inputs, args.source_plan, source, target, runner, args.output_dir)
        elif args.experiment == 'stock-product':
            report = execute_stock_product(inputs, args.source_plan, source, target, runner, args.output_dir)
        else:
            report = execute_thread_pairs(inputs, args.aa_report, source, target, runner, args.output_dir)
        code = 0 if report['status'] in ('PASS-MEASUREMENT', 'PASS-OPTIMIZATION', 'PASS-STOCK-FALLBACK') else 3 if report['status'] in ('INCONCLUSIVE', 'ENVIRONMENT-BLOCKED') else 2
        print(json.dumps({'status': report['status'], 'reason': report.get('reason'),
                          'change_pct': report.get('geometric_change_pct'), 'ci90_pct': report.get('ci90_pct'),
                          'report': str(args.output_dir / 'report.json')}))
        return code
    except (EnvironmentBlocked, RuntimeError) as error:
        print(json.dumps({'status': 'ENVIRONMENT-BLOCKED', 'reason': str(error)}))
        return 3
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'IDENTITY-STOP', 'reason': str(error)}))
        return 2
    finally:
        if sampler:
            sampler.close()
        elif base_sampler:
            base_sampler.close()


if __name__ == '__main__':
    raise SystemExit(main())
