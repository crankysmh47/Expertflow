"""Run once or reconstruct the bounded qualified Q6 PDL diagnostic."""
import argparse
import json
from pathlib import Path

from expertflow.compiler.cuda_pdl import execute, audit
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import CompilationRequest, load_compiler_inputs
from expertflow.compiler.preflight import capture_host_environment
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=('run', 'audit'), default='audit')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--evidence-db', type=Path, required=True)
    p.add_argument('--source-repository', type=Path, default=Path('C:/models/expertflow/worktrees/llama-q6-placement-final'))
    p.add_argument('--accepted-dir', type=Path, default=Path('docs/research/evidence/stock-discovery-20261004/accepted'))
    p.add_argument('--source-evidence-db', type=Path, default=Path('C:/models/expertflow/runs/compiler-stock-product-20261004/compiler.sqlite3'))
    a = p.parse_args()
    sampler = None
    try:
        if a.action == 'audit':
            if not a.evidence_db.is_file(): raise ValueError('existing evidence database required')
            report = json.loads((a.output_dir / 'report.json').read_text())
            stats = audit(report, EvidenceStore(a.evidence_db), host_environment=capture_host_environment())
            print(json.dumps({'status': 'VERIFIED-PDL-DIAGNOSTIC', 'statistics': stats}))
            return 0
        if a.evidence_db.exists() or a.output_dir.exists():
            raise ValueError('fresh output/database required; no retries or resuming')
        if not a.source_evidence_db.is_file(): raise ValueError('accepted source database required')
        request = CompilationRequest(Path('configs/compiler/gemma4-q6-model.json'),
            Path('docs/research/evidence/q6-download/tensor-inventory.json'),
            Path('docs/research/evidence/compiler-phase3/inputs/hardware.json'),
            Path('configs/compiler/gemma4-q6-single-request.json'),
            Path('docs/research/evidence/compiler-phase3/inputs/runtime-identity.json'), (), a.evidence_db, a.output_dir)
        inputs = load_compiler_inputs(request, live=True)
        source, target = EvidenceStore(a.source_evidence_db), EvidenceStore(a.evidence_db)
        sampler = WindowsGpuMemorySampler(inputs.hardware.gpu_uuid)
        report = execute(inputs, a.accepted_dir, source, target,
            ServerMeasurementRunner(target, memory_sampler=sampler), a.output_dir, source_repository=a.source_repository)
        print(json.dumps({'status': report['status'], 'reason': report.get('reason'),
                          'statistics': report.get('statistics'), 'native_outcomes': len(report['outcomes'])}))
        return 0 if report['status'] in ('NO-GO', 'PASS-DIAGNOSTIC') else 2
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'IDENTITY-STOP', 'reason': str(error)}))
        return 2
    finally:
        if sampler: sampler.close()


if __name__ == '__main__':
    raise SystemExit(main())
