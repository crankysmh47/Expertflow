# Architecture

```text
GGUF inventory + measured layer profile
                  |
                  v
          ExpertFlow optimizer
                  |
       portable deployment.json
          /               \
 llama-cli runner      llama-server
          \               /
       static CUDA expert banks
       CPU-backed original sources
```

The optimizer consumes model identity, measured per-layer expert work, exact expert-bank bytes, and the GPU budget. The verified Q6 policy selects twelve complete expert banks. The runtime creates identity-mapped CUDA shadow tensors before execution and leaves the original experts CPU-backed.

The deployment manifest is the public boundary. `run` and `serve` resolve model/runtime paths from arguments or environment variables, apply the static layer list, and launch the compatible binary. `compare` and `demo --replay` never infer new performance: they read committed evidence.

The release has no dynamic expert cache. The predictive-cache track ended at simulation because the projected transfer and remapping cost outweighed the memory benefit on this GPU.

Failure is explicit. Hash mismatches, missing binaries, missing measured profiles, and unsupported paths return JSON failure status. The feature remains disabled when the ExpertFlow environment variables are absent.

