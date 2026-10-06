@echo off
setlocal
call "C:\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39 || exit /b 1
if not exist C:\models\expertflow\builds\llama-cpu-prefetch-20261004\fixture-stock mkdir C:\models\expertflow\builds\llama-cpu-prefetch-20261004\fixture-stock
cl /nologo /EHsc /std:c++17 /I C:/models/expertflow/worktrees/llama-cpu-prefetch-20261004/ggml/include tests/native/cpu_expert_prefetch_parity.cpp /Fo:.superpowers/sdd/cpu-expert-prefetch-20261004/fixture.obj /Fe:C:/models/expertflow/builds/llama-cpu-prefetch-20261004/fixture-stock/cpu-expert-prefetch-fixture.exe /link C:/models/expertflow/builds/llama-a7312ae-cuda128-clean/ggml/src/ggml-base.lib C:/models/expertflow/builds/llama-a7312ae-cuda128-clean/ggml/src/ggml-cpu.lib
