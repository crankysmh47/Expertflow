@echo off
setlocal
call "C:\BuildTools2022\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.39 || exit /b 1
set "PDL_PROBE_BUILD=C:\models\expertflow\builds\cuda-pdl-audit-20261004"
if not exist "%PDL_PROBE_BUILD%" mkdir "%PDL_PROBE_BUILD%"
set "PDL_CUPTI=C:\models\expertflow\dependencies\cuda-audit-12.8.1\cuda_cupti-windows-x86_64-12.8.90-archive"
set "PDL_CUPTI_HEADERS=%PDL_CUPTI%\include"
if "%~1"=="compatibility" set "PDL_CUPTI=C:\models\expertflow\dependencies\cuda-audit-12.9.1\cuda_cupti-windows-x86_64-12.9.79-archive"
set "PDL_STOCK=C:\models\expertflow\builds\llama-a7312ae-cuda128-clean"
cl /nologo /EHsc /std:c++17 /I C:\models\expertflow\worktrees\llama-q6-placement-final\ggml\include /I "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\include" /I "%PDL_CUPTI_HEADERS%" docs\evidence\compiler-cuda-pdl-20261004\probe.cpp /Fo"%PDL_PROBE_BUILD%\probe.obj" /Fe"%PDL_PROBE_BUILD%\probe.exe" /link "%PDL_STOCK%\ggml\src\ggml-base.lib" "%PDL_STOCK%\ggml\src\ggml.lib" "%PDL_STOCK%\ggml\src\ggml-cuda\ggml-cuda.lib" "%PDL_CUPTI%\lib\cupti.lib"
exit /b %errorlevel%
