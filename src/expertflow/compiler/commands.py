"""Thin JSON command handlers for the compiler, with atomic output artifacts."""

import json
from pathlib import Path
import subprocess
import sqlite3

from .evidence import EvidenceStore
from .pipeline import (CompilationRequest, EnvironmentBlocked, atomic_json, compile_phase3,
                       inspect_model, load_compiler_inputs, replay_sealed_plan)
from .plan import load_execution_plan
from .reference import read_json
from .schema import canonical_payload, canonical_sha256


def _input_arguments(parser, *, required=True):
    for name in ('descriptor', 'inventory', 'hardware', 'workload', 'runtime-identity', 'evidence-db'):
        parser.add_argument('--' + name, type=Path, required=required)


def configure_sealed_run(parser):
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--acceptance', type=Path)
    _input_arguments(parser, required=False)
    parser.add_argument('--output-dir', type=Path)


def add_compiler_commands(commands):
    inspect = commands.add_parser('inspect', help='Normalize supported GGUF metadata into ModelIR.')
    for name in ('descriptor', 'inventory', 'output'):
        inspect.add_argument('--' + name, type=Path, required=True)
    compile_cmd = commands.add_parser('compile', help='Compile, confirm and replay a measured exact execution plan.')
    _input_arguments(compile_cmd)
    compile_cmd.add_argument('--layer-profile', type=Path, action='append', required=True)
    compile_cmd.add_argument('--output-dir', type=Path, required=True)
    compile_cmd.add_argument('--recorded-evidence', type=Path)
    validate = commands.add_parser('validate', help='Verify a sealed plan, evidence and live identities.')
    _input_arguments(validate)
    validate.add_argument('--plan', type=Path, required=True)
    validate.add_argument('--acceptance', type=Path)
    explain = commands.add_parser('explain', help='Export decisions without promoting estimates to measurements.')
    for name in ('plan', 'report', 'output'):
        explain.add_argument('--' + name, type=Path, required=True)


def _request(args):
    required = ('descriptor', 'inventory', 'hardware', 'workload', 'runtime_identity', 'evidence_db')
    if any(getattr(args, name, None) is None for name in required):
        raise ValueError('sealed-plan operation requires descriptor, inventory, hardware, workload, runtime-identity and evidence-db')
    return CompilationRequest(args.descriptor, args.inventory, args.hardware, args.workload, args.runtime_identity,
        tuple(getattr(args, 'layer_profile', ()) or ()), args.evidence_db,
        getattr(args, 'output_dir', None) or Path.cwd(), getattr(args, 'recorded_evidence', None))


def handle_compiler_command(args):
    try:
        if args.command == 'inspect':
            model = inspect_model(args.descriptor, args.inventory)
            atomic_json(args.output, model)
            result = {'status': 'NORMALIZED-METADATA', 'model_sha256': canonical_sha256(model),
                'output': str(args.output.resolve())}
            code = 0
        elif args.command == 'compile':
            request = _request(args)
            compiled = compile_phase3(request)
            result = {'status': compiled.status, 'reason': compiled.report.get('reason'),
                      'plan': str((request.output_dir / 'execution-plan.json').resolve()) if compiled.execution_plan else None,
                      'report': str((request.output_dir / 'explanation.json').resolve())}
            code = 0 if compiled.status in {'PASS-STOCK-FALLBACK','PASS-STATIC','RECORDED-DIAGNOSTIC','RECORDED-REPLAY'} else (
                3 if compiled.status in {'ENVIRONMENT-BLOCKED','INCONCLUSIVE'} else 2)
            if code == 2 and not compiled.report['measurements']:
                result['status'] = 'IDENTITY-STOP'
        elif args.command in {'validate', 'run'}:
            if args.command == 'run' and (args.deployment is not None or args.runtime is not None or args.model is not None or args.dry_run):
                raise ValueError('mixed legacy deployment and sealed-plan arguments')
            request = _request(args)
            if args.command == 'run' and args.output_dir is None:
                raise ValueError('sealed-plan run requires output-dir')
            inputs = load_compiler_inputs(request, live=True)
            store = EvidenceStore(request.evidence_db_path)
            acceptance = getattr(args, 'acceptance', None)
            if acceptance:
                from .stock_validation import load_validated_stock_plan
                from .preflight import capture_host_environment
                plan = load_validated_stock_plan(args.plan, acceptance, store,
                    identities=inputs.identities(inputs.stock), host_environment=capture_host_environment())
            else:
                plan = load_execution_plan(args.plan, identities=inputs.identities(inputs.stock), store=store)
            if args.command == 'validate':
                result = {'status': 'VALIDATED-STOCK-FALLBACK' if acceptance else 'VALIDATED', 'plan_sha256': plan.plan_sha256}
                code = 0
            else:
                if acceptance:
                    from .stock_validation import run_accepted_stock_plan
                    status, replay = run_accepted_stock_plan(args.plan, acceptance, inputs, store,
                        request.output_dir / 'raw-replay')
                else:
                    status, replay = replay_sealed_plan(args.plan, inputs, None, store, request.output_dir / 'raw-replay')
                result = {'status': status, **replay}
                atomic_json(request.output_dir / 'replay.json', result)
                code = 0 if status.startswith('PASS-') or status == 'MEASURED-ACCEPTED-STOCK' else 3 if status == 'ENVIRONMENT-BLOCKED' else 2
        elif args.command == 'explain':
            plan = load_execution_plan(args.plan)
            report = read_json(args.report)
            result = {'plan_sha256': plan.plan_sha256, 'plan': canonical_payload(plan),
                      'report': report, 'validation_scope': 'structural_only'}
            atomic_json(args.output, result)
            code = 0
        else:
            raise ValueError('unsupported compiler command')
    except (EnvironmentBlocked, RuntimeError, subprocess.TimeoutExpired) as error:
        result, code = {'status': 'ENVIRONMENT-BLOCKED', 'reason': str(error)}, 3
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as error:
        result, code = {'status': 'IDENTITY-STOP', 'reason': str(error)}, 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return code
