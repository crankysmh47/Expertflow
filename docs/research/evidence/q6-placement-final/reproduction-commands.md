# Reproduction commands

Run from the isolated ExpertFlow worktree with the Q6 model and release CUDA runtime paths from `placement-manifest.json`.

```powershell
& .\scripts\run_q6_selected_static_pairs.ps1 `
  -Runtime C:\models\expertflow\builds\llama-q6-placement-final\bin\llama-cli.exe `
  -Model C:\models\gemma-4-26b-a4b-q6\google_gemma-4-26B-A4B-it-Q6_K.gguf `
  -Output C:\models\expertflow\runs\q6-placement-final\stage8-authoritative\matched-10x512 `
  -Pairs 10 -GeneratedTokens 512 -BaselineLayers '' `
  -StaticLayers '0,1,2,3,4,5,6,7,8,9,15,20' `
  -BaselineCudaGraphs on -CudaGraphs on -StaticPrecompute

python scripts\analyze_q6_selected_static.py `
  --input C:\models\expertflow\runs\q6-placement-final\stage8-authoritative\matched-10x512\raw-results.json `
  --output-json C:\models\expertflow\runs\q6-placement-final\stage8-authoritative\matched-10x512\summary.json `
  --output-csv C:\models\expertflow\runs\q6-placement-final\stage8-authoritative\matched-10x512\run-pairs.csv
```

PPL used `llama-perplexity.exe` with `-c 2048 -b 512 -ub 1 --chunks 8 -ngl 99 --cpu-moe --threads 12 --seed 42 --no-warmup`; stock had no ExpertFlow variables and candidate set `LLAMA_EXPERTFLOW_STATIC_ISLAND_LAYER=0,1,2,3,4,5,6,7,8,9,15,20` and `LLAMA_EXPERTFLOW_STATIC_PRECOMPUTE=1`. CUDA graphs were enabled in both modes.

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\analyze_q1b_nll.py `
  --reference C:\models\expertflow\runs\q6-placement-final\stage8-quality\ppl-off\nll.jsonl `
  --candidate C:\models\expertflow\runs\q6-placement-final\stage8-quality\ppl-on\nll.jsonl `
  --block-size 128 --bootstrap-samples 10000 --seed 20260719 --threshold 0.01 `
  --output C:\models\expertflow\runs\q6-placement-final\stage8-quality\ppl-analysis.json
```
