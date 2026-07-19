# Q6 predictive follow-up reproduction commands

All GPU processes are run sequentially. The frozen static state is reconstructed from `static-state.json`; the ten-pair performance benchmark is not rerun.

## Fixed MMLU

Both servers use the Q6 model, context 2048, `-ngl 99`, `--cpu-moe`, 12 threads, one slot, CUDA graphs enabled, and default batch/ubatch/KV settings matching the frozen performance command. Requests are the frozen 100-item zero-shot manifest with temperature 0, seed 42, prompt cache disabled, and grammar-constrained `A-D` output.

```powershell
# OFF: no ExpertFlow environment variables
llama-server.exe -m <Q6.gguf> -c 2048 -ngl 99 --cpu-moe --threads 12 --parallel 1 --port 18431
python scripts/run_q1b_mmlu.py --manifest docs/evidence/q6-quality-preserving/quality-manifest.json --endpoint http://127.0.0.1:18431/completion --output <off-results.json>

# ON: same command and request protocol
$env:LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER='0,1,2,3,4,5,6,7,8,9,15,20'
$env:LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE='1'
llama-server.exe -m <Q6.gguf> -c 2048 -ngl 99 --cpu-moe --threads 12 --parallel 1 --port 18432
python scripts/run_q1b_mmlu.py --manifest docs/evidence/q6-quality-preserving/quality-manifest.json --endpoint http://127.0.0.1:18432/completion --output <on-results.json>
```

The 14 OFF/ON changed items were selected by original manifest index `[9,17,30,39,49,61,62,75,79,86,92,93,94,95]`, written to a no-BOM subset manifest, and rerun with the identical ON server environment. Each repeated item matched selection hash, prediction, token IDs, and content.

## Q6 routing and simulation

```powershell
$env:PYTHONPATH="$PWD\src"
python scripts\collect_q6_routing_workloads.py `
  --ppl-corpus C:\models\expertflow\quality-data\q1b-heldout\wikitext-103-validation.txt `
  --mmlu-manifest docs\evidence\q6-quality-preserving\quality-manifest.json `
  --conversation-corpus C:\models\expertflow\worktrees\expanded-canonical-collection\configs\expanded-canonical-84.json `
  --runtime C:\models\expertflow\runs\canonical-observer-smoke\runtime `
  --model C:\models\gemma-4-26b-a4b-q6\google_gemma-4-26B-A4B-it-Q6_K.gguf `
  --root C:\models\expertflow\runs\q6-predictive-final\stage2-traces\canonical-q6-r2

python scripts\analyze_q6_routing_workloads.py `
  --manifest C:\models\expertflow\runs\q6-predictive-final\stage2-traces\canonical-q6-r2\collection-manifest.json `
  --output docs\evidence\q6-predictive-final\routing-locality.json

python scripts\simulate_q6_hybrid.py `
  --locality docs\evidence\q6-predictive-final\routing-locality.json `
  --profile docs\evidence\q6-placement-final\layer-profile.json `
  --output docs\evidence\q6-predictive-final\cache-simulation.json
```
