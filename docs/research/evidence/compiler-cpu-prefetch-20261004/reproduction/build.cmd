@echo off
setlocal
call "C:\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39 || exit /b 1
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
cmake -S C:/models/expertflow/worktrees/llama-cpu-prefetch-20261004 -B C:/models/expertflow/builds/llama-cpu-prefetch-20261004 -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU_ALL_VARIANTS=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF || exit /b 1
cmake --build C:/models/expertflow/builds/llama-cpu-prefetch-20261004 --target ggml-cpu -j 12
