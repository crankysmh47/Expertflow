# Troubleshooting

## Model or binary hash warning

Use the exact Q6_K filename and the compatible patched binaries. `doctor` prints expected and actual hashes. Do not continue with a different model and compare its numbers to the release result.

## CUDA DLL not found

Set `CUDA_PATH` to CUDA 12.8 and put `%CUDA_PATH%\bin` on `PATH`. Keep the llama executable beside its DLLs.

## Server fails during load

Stop other GPU-heavy applications, check `nvidia-smi`, and confirm that the profile's expected peak plus 512 MiB fits. The 262,144 context profile had only 675.418 MiB reserve on the measured machine.

## Output differs under parallel requests

The four-slot benchmark completed without errors but did not reproduce every concurrent response hash. Use `max-performance.json` for the frozen single-request result. Treat the four-slot profile as throughput evidence, not deterministic-output evidence.

## Replay hash failure

Restore the release files from the ZIP. Do not edit `replay-data.json` or `release-state.json`; replay is designed to fail if either changes.
