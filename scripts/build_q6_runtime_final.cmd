@echo off
setlocal

if "%~2"=="" (
  echo usage: build_q6_runtime_final.cmd LLAMA_SOURCE BUILD_DIR
  exit /b 2
)

if not defined EXPERTFLOW_VCVARSALL set "EXPERTFLOW_VCVARSALL=C:\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat"
if not defined EXPERTFLOW_CUDA_ROOT set "EXPERTFLOW_CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

call "%EXPERTFLOW_VCVARSALL%" x64 -vcvars_ver=14.39 || exit /b 1
set "PATH=%EXPERTFLOW_CUDA_ROOT%\bin;%PATH%"

cmake -S "%~1" -B "%~2" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl.exe ^
  -DCMAKE_CXX_COMPILER=cl.exe ^
  -DBUILD_SHARED_LIBS=ON ^
  -DGGML_CUDA=ON ^
  -DGGML_NATIVE=OFF ^
  -DGGML_BACKEND_DL=ON ^
  -DGGML_CPU_ALL_VARIANTS=OFF ^
  -DCMAKE_CUDA_ARCHITECTURES=120a-real || exit /b 1

cmake --build "%~2" --target llama-cli -j 12
