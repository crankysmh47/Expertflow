param([Parameter(Mandatory=$true)][string]$Source, [Parameter(Mandatory=$true)][string]$Build)
$ErrorActionPreference = 'Stop'
$upstream = 'a7312ae94f801fc9c6786dc56e38df57b964f697'
$actual = (git -C $Source rev-parse HEAD).Trim()
if ($actual -ne $upstream) { throw "Source must be clean and pinned at $upstream; got $actual" }
if (git -C $Source status --porcelain) { throw 'llama.cpp source is not clean' }
Get-ChildItem (Join-Path $PSScriptRoot '..\patches\llama.cpp\*.patch') | Sort-Object Name | ForEach-Object { git -C $Source am $_.FullName }
cmake -S $Source -B $Build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build $Build --target llama-cli llama-server --config Release
Get-FileHash (Join-Path $Build 'bin\llama-cli.exe') -Algorithm SHA256
Get-FileHash (Join-Path $Build 'bin\llama-server.exe') -Algorithm SHA256
