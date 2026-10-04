"""Generate, execute once, or reconstruct a bounded exact stock search."""

import argparse
import json
from pathlib import Path
import sqlite3
import subprocess

from expertflow.compiler.diagnostics import DiagnosticSampler
from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.pipeline import CompilationRequest, EnvironmentBlocked, load_compiler_inputs, atomic_json
from expertflow.compiler.preflight import capture_host_environment
from expertflow.compiler.runner import ServerMeasurementRunner, WindowsGpuMemorySampler
from expertflow.compiler.schema import canonical_payload
from expertflow.compiler.stock_discovery import prepare_search, execute_stock_search, load_search_recommendation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('generate','run','validate'),required=True)
    parser.add_argument('--descriptor',type=Path,default=Path('configs/compiler/gemma4-q6-model.json'))
    parser.add_argument('--inventory',type=Path,default=Path('docs/evidence/q6-download/tensor-inventory.json'))
    parser.add_argument('--hardware',type=Path,default=Path('docs/evidence/compiler-phase3/inputs/hardware.json'))
    parser.add_argument('--workload',type=Path,default=Path('configs/compiler/gemma4-q6-single-request.json'))
    parser.add_argument('--runtime-identity',type=Path,default=Path('docs/evidence/compiler-phase3/inputs/runtime-identity.json'))
    parser.add_argument('--source-plan',type=Path,default=Path('docs/evidence/stock-discovery-20261004/accepted/execution-plan.json'))
    parser.add_argument('--source-receipt',type=Path,default=Path('docs/evidence/stock-discovery-20261004/accepted/acceptance-receipt.json'))
    parser.add_argument('--source-evidence-db',type=Path,default=Path('C:/models/expertflow/runs/compiler-stock-product-20261004/compiler.sqlite3'))
    parser.add_argument('--source-repository',type=Path,default=Path('C:/models/expertflow/worktrees/llama-q6-placement-final'))
    parser.add_argument('--exclude-threads',type=int,action='append',default=[])
    parser.add_argument('--space-config',type=Path,help='explicit alternate coverage/budget JSON; default stays in approved32-process space')
    parser.add_argument('--evidence-db',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--manifest-output',type=Path)
    parser.add_argument('--recommendation',type=Path)
    args = parser.parse_args(argv)
    sampler = base = None
    try:
        if args.action != 'validate' and not args.source_evidence_db.is_file():
            raise EnvironmentBlocked('accepted stock prerequisite database unavailable')
        if args.action == 'validate':
            if args.evidence_db is None or not args.evidence_db.is_file() or args.recommendation is None:
                raise ValueError('validate requires existing --evidence-db and --recommendation')
            output = args.recommendation.parent
        else:
            if args.output_dir is None or args.output_dir.exists():
                raise ValueError('fresh --output-dir required; no retries/resume')
            output = args.output_dir
        if args.action == 'run' and (args.evidence_db is None or args.evidence_db.exists()):
            raise ValueError('run requires fresh --evidence-db')
        if args.action == 'generate' and (args.manifest_output is None or args.manifest_output.exists()):
            raise ValueError('generate requires fresh --manifest-output outside future output directory')
        if args.action == 'generate' and args.manifest_output.resolve().is_relative_to(output.resolve()):
            raise ValueError('preview must remain outside future native output directory')
        request = CompilationRequest(args.descriptor,args.inventory,args.hardware,args.workload,
            args.runtime_identity,(),args.evidence_db or output/'compiler.sqlite3',output)
        inputs = load_compiler_inputs(request,live=True)
        exclusions = {count:'explicit exclusion; prior eight-thread rejection is not current-host cached evidence'
                      for count in args.exclude_threads}
        config = json.loads(args.space_config.read_text(encoding='utf-8')) if args.space_config else None
        if args.action == 'validate':
            store = EvidenceStore(args.evidence_db)
            plan = load_search_recommendation(args.recommendation,store,host_environment=capture_host_environment())
            receipt = json.loads((args.recommendation/'search-receipt.json').read_text(encoding='utf-8'))
            expected = canonical_payload(inputs.identities(inputs.stock))
            if receipt['experiment']['manifest']['prerequisite_plan']['candidate']['identities'] != expected:
                raise ValueError('actual inputs differ from search semantic/runtime identity')
            result = {'status':'VALIDATED-STOCK-RECOMMENDATION','plan_sha256':plan.plan_sha256}
            code = 0
        else:
            source = EvidenceStore(args.source_evidence_db)
            if args.action == 'generate':
                manifest = prepare_search(inputs,args.source_plan,args.source_receipt,source,output,
                    host_environment=capture_host_environment(),source_repository=args.source_repository,
                    excluded_threads=exclusions,space_config=config)
                atomic_json(args.manifest_output,manifest)
                result = {'status':'GENERATED-SEARCH-MANIFEST','manifest_sha256':manifest['manifest_sha256'],
                          'manifest':str(args.manifest_output),'native_samples':0}
                code = 0
            else:
                target = EvidenceStore(args.evidence_db)
                base = WindowsGpuMemorySampler(inputs.hardware.gpu_uuid)
                sampler = DiagnosticSampler(base)
                runner = ServerMeasurementRunner(target,memory_sampler=sampler)
                report = execute_stock_search(inputs,args.source_plan,args.source_receipt,source,target,runner,output,
                    source_repository=args.source_repository,excluded_threads=exclusions,space_config=config)
                result = {'status':report['status'],'reason':report.get('reason'),
                    'report':str(output/'report.json'),'native_outcomes':len(report['outcomes']),
                    'recommended_id':report.get('recommended_id'),'statistics':report.get('statistics')}
                code = 0 if report['status'].startswith('RECOMMENDED-') else 3 if report['status'] == 'ENVIRONMENT-BLOCKED' else 2
    except (EnvironmentBlocked,RuntimeError,subprocess.TimeoutExpired) as error:
        result,code = {'status':'ENVIRONMENT-BLOCKED','reason':str(error)},3
    except (ValueError,OSError,KeyError,TypeError,sqlite3.Error) as error:
        result,code = {'status':'IDENTITY-STOP','reason':str(error)},2
    finally:
        if sampler:
            sampler.close()
        elif base:
            base.close()
    print(json.dumps(result))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
