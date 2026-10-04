import json
from pathlib import Path

from expertflow.compiler.pipeline import atomic_json
from expertflow.compiler.preflight import capture_host_environment, file_sha256

root = Path('docs/evidence/stock-discovery-20261004')
root.mkdir(parents=True, exist_ok=True)
paths = [Path(p) for p in (
    'docs/evidence/q6-runtime-final/baseline-results.json',
    'docs/evidence/q6-runtime-final/baseline-audit.md',
    'docs/evidence/compiler-phase3/README.md',
    'docs/evidence/compiler-phase3/verification.json',
    'docs/evidence/compiler-refinement/report.md',
    'docs/evidence/q6-placement-final/quality-results.json',
    'docs/evidence/product-release/throughput-profile.json',
    'docs/evidence/compiler-cpu-prefetch-20261004/report.json')]
baseline = json.loads(paths[0].read_text())
quality = json.loads(paths[5].read_text())
throughput = json.loads(paths[6].read_text(encoding='utf-8-sig'))
prefetch = json.loads(paths[7].read_text())
assert quality['verdict'] == 'QUALITY STOP' and quality['perplexity']['gate_pass'] is False
assert throughput['output_deterministic_across_concurrent_runs'] is False
assert prefetch['status'] == 'INCONCLUSIVE'
audit = {'schema_version':'1.0.0', 'status':'HISTORICAL-AUDIT',
    'sources':{str(p):file_sha256(p) for p in paths},
    'records':[
        {'name':'historical pristine stock','mean_decode_tps':baseline['fair_stock_baseline']['mean_decode_tps'],
         'interface':'llama-cli','concurrency':1,'processes':3,'stock':True,'eligible_quality':True},
        {'name':'original stock confirmation','mean_decode_tps':24.411,
         'interface':'server_completion','concurrency':1,'processes':10,'stock':True,'eligible_quality':True,
         'note':'rounded historical report mean; product publication failed replay repeatability'},
        {'name':'original stock replay','mean_decode_tps':25.383,
         'interface':'server_completion','concurrency':1,'processes':1,'stock':True,'eligible_quality':True,
         'note':'rounded historical report; exceeds original absolute2% gate; not a paired speedup'},
        {'name':'static placement champion','mean_decode_tps':28.13,
         'interface':'llama-cli','concurrency':1,'stock':False,'eligible_quality':False,
         'quality_verdict':quality['verdict'],'ppl_ci95_pct':quality['perplexity']['paired_bootstrap_95_pct']},
        {'name':'concurrent product profile','aggregate_generated_tps':throughput['aggregate_generated_tps'],
         'interface':'continuous_batching','concurrency':4,'stock':False,'eligible_quality':False,
         'note':'aggregate throughput; nondeterministic responses; distinct objective'},
        {'name':'latest paired pristine control','mean_decode_tps':prefetch['direct_mean_tps'],
         'interface':'server_completion','concurrency':1,'processes':10,'stock':True,'eligible_quality':True,
         'note':'current retained control sample, not comparable as matched gain against old experiments'}],
    'decision':'recover strongest eligible pristine stock; never promote static quality failure or aggregate TPS as exact single-request speed',
    'host_environment':capture_host_environment(),
    'historical_cpu_environment':'not sufficiently pinned by GPU-only compiler HardwareIR; no attribution of inter-experiment speed drift'}
atomic_json(root/'history-audit.json',audit)
atomic_json(root/'host-environment.json',audit['host_environment'])
print(json.dumps(audit['host_environment'],indent=2))
