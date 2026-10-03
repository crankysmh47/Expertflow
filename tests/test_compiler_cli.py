import json
from pathlib import Path

from expertflow.cli.main import main


def test_inspect_unknown_family_structured_failure(tmp_path,capsys):
    descriptor=tmp_path/'descriptor.json'
    descriptor.write_text(json.dumps({'family':'unknown','architecture':'unknown','quantization':'Q6_K',
        'expert_count':128,'expert_top_k':8,'kv_kind':'standard','mtp_kind':'none'}))
    inventory=tmp_path/'inventory.json'
    inventory.write_text('{}')
    assert main(['inspect','--descriptor',str(descriptor),'--inventory',str(inventory),'--output',str(tmp_path/'out.json')]) == 2
    assert 'unsupported' in json.loads(capsys.readouterr().out)['reason']


def test_compile_missing_inputs_and_run_reject_mixed_legacy_arguments(tmp_path,capsys):
    argv=['compile']
    for name in ('descriptor','inventory','hardware','workload','runtime-identity','layer-profile','evidence-db','output-dir'):
        argv += ['--'+name,str(tmp_path/name)]
    assert main(argv)==2
    assert json.loads(capsys.readouterr().out)['status']=='IDENTITY-STOP'
    assert main(['run','legacy.json','--plan','sealed.json'])==2
    assert 'mixed' in json.loads(capsys.readouterr().out)['reason']


def test_explain_never_relabels_estimates(tmp_path,capsys):
    report=tmp_path/'report.json';report.write_text('{"estimated_decode_tps": 30}')
    plan=tmp_path/'invalid.json';plan.write_text('{}')
    assert main(['explain','--plan',str(plan),'--report',str(report),'--output',str(tmp_path/'out.json')])==2
    assert not (tmp_path/'out.json').exists()
