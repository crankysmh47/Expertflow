# Third-party notices

ExpertFlow interoperates with llama.cpp, which is distributed under the MIT License. The release includes an ExpertFlow patch series, not an upstream llama.cpp binary.

Gemma model weights are not included. Users must obtain the GGUF and comply with the Gemma terms attached to the original model distribution.

Python development dependencies retain their own licenses. `uv.lock` records the exact package set used for verification.

The local alpha wheel contains ExpertFlow Python code and a runtime identity
manifest. It contains no llama.cpp binaries, CUDA redistributables or model
weights. Bring-your-own and explicitly downloaded native runtimes retain their
upstream licenses. Source and MIT license: https://github.com/ggml-org/llama.cpp.
The tested pristine runtime revision is a7312ae9; historical ExpertFlow patches
are a separate research surface and are not silently applied by local setup.

The optional qualification artifact Qwen2.5-0.5B-Instruct-GGUF is distributed by
Qwen under Apache-2.0: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF.
Granite and Gemma weights likewise retain publisher terms. Downloading a model
does not grant additional rights; check its original distribution. No weights
are redistributed in the alpha artifacts.
