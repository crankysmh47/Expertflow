"""Read-only audit after collection; never launches a native model process."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.preflight import capture_host_environment, file_sha256
from expertflow.compiler.schema import canonical_payload
from expertflow.compiler.stock_validation import reconstruct_product, load_validated_stock_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-plan', type=Path,
        default=Path('docs/evidence/compiler-phase3/execution-plan.pending.json'))
    args = parser.parse_args()
    report = json.loads((args.root/'report.json').read_text(encoding='utf-8'))
    store = EvidenceStore(args.database)
    freeze = report['frozen']
    assert file_sha256(Path(freeze['source_evidence_db'])) == freeze['source_evidence_db_sha256']
    assert file_sha256(args.source_plan) == freeze['source_plan_file_sha256']
    rows = [{**store.verify_measurement(row['measurement_id']),
             'pair':row['pair'], 'arm':row['arm'], 'measurement_id':row['measurement_id']}
            for row in report['rows']]
    assert len(rows) == 20 and len({r['owned_run_sha256'] for r in rows}) == 20
    pairs = {(r['pair'],r['arm']):r['decode_tps'] for r in rows}
    direct = [pairs[i,'direct'] for i in range(10)]
    sealed = [pairs[i,'sealed'] for i in range(10)]
    differences = [math.log(b)-math.log(a) for a,b in zip(direct,sealed)]
    rng = random.Random(20261003)
    draws = sorted(100*math.expm1(statistics.mean(rng.choices(differences,k=10)))
                   for _ in range(10000))
    independent = {'geometric_change_pct':100*math.expm1(statistics.mean(differences)),
                   'ci90_pct':[draws[499],draws[9499]], 'ci95_pct':[draws[249],draws[9749]],
                   'direct_mean_tps':statistics.mean(direct), 'sealed_mean_tps':statistics.mean(sealed),
                   'direct_cv_pct':100*statistics.stdev(direct)/statistics.mean(direct),
                   'sealed_cv_pct':100*statistics.stdev(sealed)/statistics.mean(sealed)}
    for key,value in independent.items():
        assert canonical_payload(value) == report[key], key
    passed = (-2 < independent['ci90_pct'][0] and independent['ci90_pct'][1] < 2
              and max(independent['direct_cv_pct'],independent['sealed_cv_pct']) <= 10)
    assert passed == (report['status'] == 'PASS-STOCK-FALLBACK')
    if passed:
        host = capture_host_environment()
        source, _, _ = reconstruct_product({**report,'status':'PASS-MEASUREMENT'}, store, host_environment=host)
        plan = load_validated_stock_plan(args.root/'accepted/execution-plan.json',
            args.root/'accepted/acceptance-receipt.json',store,
            identities=source.candidate.identities,host_environment=host)
        plan_hash = plan.plan_sha256
    else:
        assert not (args.root/'accepted').exists()
        plan_hash = None
    artifacts = {mid:{a.role:a.identity.sha256 for a in store.measurement(mid).artifacts}
                 for mid in (r['measurement_id'] for r in rows)}
    result = {'status':'VERIFIED-PASS' if passed else 'VERIFIED-NON-PASS',
              'source_plan_unchanged':True, 'source_database_unchanged':True,
              'plan_sha256':plan_hash,'independent_statistics':independent,
              'measurement_ids':[r['measurement_id'] for r in rows], 'artifacts':artifacts,
              'report_sha256':file_sha256(args.root/'report.json')}
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'], **independent}))


if __name__ == '__main__':
    main()
