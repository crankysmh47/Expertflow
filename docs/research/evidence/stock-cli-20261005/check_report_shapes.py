"""Check scope extraction from saved real report schemas, without validating native records."""
import json
from pathlib import Path

from expertflow.stock.cli import _decision

HERE = Path(__file__).resolve().parent


def main():
    paths = {
        'product':Path('C:/models/expertflow/runs/compiler-stock-repeatability-20261004/block-b/report.json'),
        'search':Path('C:/models/expertflow/runs/compiler-q4-stock-search-20261004/search/report.json'),
        'reference':Path('docs/evidence/compiler-granite-20261004/reference-diagnostic/reference-receipt.json'),
    }
    checks = []
    for workflow,path in paths.items():
        saved = json.loads(path.read_text())
        report = saved['experiment'] if workflow == 'reference' else saved
        # This utility checks the presentation schema only. The hypothetical
        # driver result below is not a fresh evidence validation or a new PASS.
        decision = _decision({'status':report['status']},0,'validate',report=report)
        scope = decision['coverage']['inputs']
        assert scope and all('workload_sha256' in value for value in scope.values())
        if workflow == 'product':
            assert scope['source_plan']['model_ir_sha256'] == report['frozen']['identities']['model_sha256']
        else:
            candidate = (report['manifest']['candidate'] if workflow == 'reference' else
                         report['manifest']['candidates'][report['recommended_id']])
            assert scope['inputs']['workload_sha256'] == candidate['identities']['workload_sha256']
        assert 'selected_default' not in decision and decision['utility_gain_established'] is False
        if workflow == 'search':
            assert decision['retained_incumbent'] is True
        checks.append({'workflow':workflow,'report':str(path),'workload_scope_present':True,
            'false_default_claim_absent':True})
    (HERE/'report-shape-controls.json').write_text(json.dumps({'status':'PASS-SCHEMA-CONTROLS',
        'native_calls':0,'validation_scope':'Saved report extraction only; no native record/receipt acceptance is certified here.',
        'checks':checks},indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({'status':'PASS-SCHEMA-CONTROLS','checks':len(checks),'native_calls':0}))


if __name__ == '__main__':
    main()
