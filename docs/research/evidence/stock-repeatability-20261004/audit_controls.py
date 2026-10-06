"""Independent audit controls; no native process or live evidence mutation."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile

spec = importlib.util.spec_from_file_location('raw_audit', 'docs/evidence/stock-repeatability-20261004/independent_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
ids = list('abcdef')
schedule = [ids, ids[2:]+ids[:2], ids[4:]+ids[:4]]
report = {'status':'PASS-STOCK-UTILITY', 'manifest':{'candidates':{cid:{} for cid in ids},
    'default_id':'a', 'screening_schedule':schedule}, 'rows':[], 'automatic_id':'b', 'manual_id':'b'}
rates, costs = {}, {}
def add(label, cid, rate):
    mid = label
    report['rows'].append({'label':label,'candidate_id':cid,'measurement_id':mid,
        'decode_tps':rate,'run_wall_seconds':40.0})
    rates[mid] = rate
    costs[mid] = 40.0
for n in range(10):
    add(f'reference-{n:02}', 'a', 20.0)
for method in ('automatic','manual'):
    for block, order in enumerate(schedule):
        for cid in order:
            add(f'{method}-screen-{block:02}-{cid}', cid, 24.0 if cid=='b' else 20.0)
for purpose in ('defaults','manual'):
    for n in range(10):
        for arm in ('direct','sealed'):
            cid = 'a' if purpose=='defaults' and arm=='direct' else 'b'
            add(f'{purpose}-pair-{n:02}-{arm}',cid,20.0 if cid=='a' else 24.0)
def claimed(control, selected):
    return {key:value for key,value in audit.paired(control,selected).items()
        if key not in ('noninferior','equivalent','variance_pass','status')}
report['statistics'] = {'status':'PASS-STOCK-UTILITY',
    'gain':claimed([20.0]*10,[24.0]*10), 'manual_equivalence':claimed([24.0]*10,[24.0]*10),
    'automatic_evaluations':18,'manual_evaluations':18,'native_evaluation_cost_pass':True}
with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    result = audit.audit_transfer(report,rates,costs,root)
    assert result['utility_verdict']=='PASS-STOCK-UTILITY'
    assert abs(result['comparisons']['defaults']['geometric_change_pct']-20)<1e-10
    assert result['tuning_cost']['automatic']['native_evaluations']==18
    assert result['tuning_cost']['automatic']['native_phase_seconds']==720
    rejected = 0
    for mutation in ('statistics','selection','rate','cost'):
        bad = copy.deepcopy(report)
        if mutation=='statistics':bad['statistics']['gain']['ci95_pct'][0]=999
        elif mutation=='selection':bad['automatic_id']='c'
        elif mutation=='rate':bad['rows'][0]['decode_tps']=999
        else:bad['rows'][0]['run_wall_seconds']=999
        try:
            audit.audit_transfer(bad,rates,costs,root)
        except AssertionError:
            rejected += 1
    assert rejected==4, rejected
    partial = copy.deepcopy(report)
    partial['rows'] = partial['rows'][:28]
    partial.pop('automatic_id')
    partial.pop('manual_id')
    partial.pop('statistics')
    assert audit.audit_transfer(partial,rates,costs,root)['utility_verdict']=='INCOMPLETE-NATIVE-PROOF'
    neutral = copy.deepcopy(report)
    neutral_rates = dict(rates)
    for row in neutral['rows']:
        if row['label'].startswith('defaults-pair-') and row['label'].endswith('-direct'):
            neutral_rates[row['measurement_id']] = row['decode_tps'] = 24.0
    neutral['statistics']['gain'] = claimed([24.0]*10,[24.0]*10)
    neutral['statistics']['status'] = neutral['status'] = 'NO-UTILITY-GAIN'
    assert audit.audit_transfer(neutral,neutral_rates,costs,root)['utility_verdict']=='NO-UTILITY-GAIN'
    product = {'status':'PASS-STOCK-FALLBACK','rows':[],**claimed([24.0]*10,[24.0]*10)}
    for n in range(10):
        for arm in ('direct','sealed'):
            mid = f'product-{n:02}-{arm}'
            product['rows'].append({'pair':n,'arm':arm,'measurement_id':mid})
            rates[mid] = 24.0
    (root/'product').mkdir()
    (root/'product/report.json').write_text(json.dumps(product),encoding='utf-8')
    report['status'] = 'PASS-STOCK-UTILITY-PRODUCT'
    report['consumer'] = {'status':'MEASURED-ACCEPTED-STOCK','measurement_id':'consumer','decode_tps':24.0}
    rates['consumer'] = 24.0
    complete = audit.audit_transfer(report,rates,costs,root)
    assert complete['product_comparison']['status']=='PASS-MEASUREMENT'
    assert complete['consumer_status']=='MEASURED-ACCEPTED-STOCK'
print(json.dumps({'status':'PASS','positive_controls':2,'partial_control':1,'neutral_stop_control':1,
    'tamper_controls_rejected':rejected,'native_processes':0}))
