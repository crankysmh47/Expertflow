"""Read-only stock-search audit; never launches a model process."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics

from expertflow.compiler.evidence import EvidenceStore
from expertflow.compiler.preflight import capture_host_environment, file_sha256
from expertflow.compiler.schema import canonical_payload, canonical_sha256
from expertflow.compiler.stock_discovery import load_search_recommendation


def audit(report, store):
    manifest = report['manifest']
    body = dict(manifest)
    assert body.pop('manifest_sha256') == canonical_sha256(body)
    for path, digest in {**manifest['source_files'], **manifest['prerequisite_files']}.items():
        assert file_sha256(Path(path)) == digest, path
    expected = [(block, cid) for block, order in enumerate(manifest['screening_schedule']) for cid in order]
    assert len(report['screening']) == len(expected)
    rows = [*report['screening'], *report['confirmation']]
    verified = {}
    artifacts = {}
    owners = set()
    for row in rows:
        mid = row['measurement_id']
        native = store.verify_measurement(mid)
        assert all(canonical_payload(row[key]) == canonical_payload(value) for key, value in native.items())
        assert native['measured'] is True and native['exit_code'] == 0
        assert all(native['validations'][key] is True for key in ('exact_tokens', 'memory', 'cleanup'))
        assert mid not in verified and native['owned_run_sha256'] not in owners
        owners.add(native['owned_run_sha256'])
        verified[mid] = native
        artifacts[mid] = {a.role: a.identity.sha256 for a in store.measurement(mid).artifacts}
    assert len({r['prompt_tokens_sha256'] for r in rows}) == 1
    assert len({r['generated_tokens_sha256'] for r in rows}) == 1
    rates = {}
    for row, (block, cid) in zip(report['screening'], expected):
        assert (row['block'], row['candidate_id']) == (block, cid)
        rates[block, cid] = row['decode_tps']
    incumbent = manifest['incumbent_id']
    ranking = []
    for cid in manifest['candidates']:
        ratios = [rates[block, cid] / rates[block, incumbent] for block in range(3)]
        ranking.append({'candidate_id': cid,
            'geometric_ratio': math.exp(statistics.mean(math.log(rates[b, cid]) - math.log(rates[b, incumbent]) for b in range(3))),
            'block_ratios': ratios, 'block_ratio_range': [min(ratios), max(ratios)],
            'block_ratio_cv_pct': 100 * statistics.stdev(ratios) / statistics.mean(ratios),
            'uncertainty_scope': 'three-block descriptive variation, not confirmation evidence'})
    ranking.sort(key=lambda r: (-r['geometric_ratio'], r['candidate_id'] != incumbent, r['candidate_id']))
    assert ranking == report['ranking']
    finalist = ranking[0]['candidate_id']
    assert finalist == report['finalist_id']
    stats = None
    accepted = False
    if finalist != incumbent:
        assert len(report['confirmation']) == 20
        for index, row in enumerate(report['confirmation']):
            pair, position = divmod(index, 2)
            arm = manifest['confirmation_schedule'][pair][position]
            assert (row['pair'], row['arm']) == (pair, arm)
            assert row['candidate_id'] == (incumbent if arm == 'direct' else finalist)
        pairs = {(r['pair'], r['arm']): r['decode_tps'] for r in report['confirmation']}
        direct = [pairs[i, 'direct'] for i in range(10)]
        sealed = [pairs[i, 'sealed'] for i in range(10)]
        changes = [math.log(b) - math.log(a) for a, b in zip(direct, sealed)]
        rng = random.Random(manifest['confirmation_seed'])
        draws = sorted(100 * math.expm1(statistics.mean(rng.choices(changes, k=10))) for _ in range(10000))
        stats = {'geometric_change_pct': 100 * math.expm1(statistics.mean(changes)),
            'ci95_pct': [draws[249], draws[9749]], 'ci90_pct': [draws[499], draws[9499]],
            'one_sided95_lower_pct': draws[499],
            'direct_mean_tps': statistics.mean(direct), 'sealed_mean_tps': statistics.mean(sealed),
            'direct_cv_pct': statistics.stdev(direct) * 100 / statistics.mean(direct),
            'sealed_cv_pct': statistics.stdev(sealed) * 100 / statistics.mean(sealed),
            'direct_tps': direct, 'sealed_tps': sealed, 'paired_log_ratios': changes,
            'bootstrap_seed': manifest['confirmation_seed'], 'bootstrap_samples': 10000,
            'method': 'percentile bootstrap of ten paired mean log ratios'}
        assert stats == report['statistics']
        accepted = stats['geometric_change_pct'] >= 2 and stats['ci95_pct'][0] > 0 and max(stats['direct_cv_pct'], stats['sealed_cv_pct']) <= 10
    else:
        assert not report['confirmation'] and report['statistics'] is None
    assert accepted == report['confirmation_accepted']
    assert report['recommended_id'] == (finalist if accepted else incumbent)
    assert report['status'] == ('RECOMMENDED-CHALLENGER' if accepted else 'RECOMMENDED-INCUMBENT')
    assert len(report['outcomes']) == len(rows)
    assert all(o['status'] == 'measured' and o['measurement_id'] == r['measurement_id'] for o, r in zip(report['outcomes'], rows))
    return {'native_runs': len(rows), 'independent_ranking': ranking,
        'independent_statistics': stats, 'recommended_id': report['recommended_id'],
        'confirmation_accepted': accepted, 'artifacts': artifacts,
        'prerequisite_files_unchanged': True, 'source_files_unchanged': True,
        'measurement_ids': list(verified)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.root / 'report.json').read_text(encoding='utf-8'))
    assert json.loads((args.root / 'frozen-manifest.json').read_text(encoding='utf-8')) == report['manifest']
    assert report['status'].startswith('RECOMMENDED-'), 'terminal recommendation required'
    store = EvidenceStore(args.database)
    result = audit(report, store)
    plan = load_search_recommendation(args.root / 'recommended', store,
        host_environment=capture_host_environment())
    result.update(status='VERIFIED-STOCK-SEARCH', plan_sha256=plan.plan_sha256,
        report_sha256=file_sha256(args.root / 'report.json'))
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('status', 'native_runs', 'recommended_id', 'confirmation_accepted', 'plan_sha256')}))


if __name__ == '__main__':
    main()
