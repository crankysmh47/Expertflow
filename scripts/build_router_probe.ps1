[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $LlamaCppSource,

    [string] $LlamaRuntime = 'C:\models\expertflow\dependencies\llama-b10002\runtime',

    [string] $BuildDirectory = 'C:\models\expertflow\build\router-probe',

    [ValidateSet('Debug', 'Release', 'RelWithDebInfo')]
    [string] $BuildType = 'Release'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$source = (Resolve-Path -LiteralPath $LlamaCppSource).Path
if (-not (Test-Path -LiteralPath (Join-Path $source 'include\llama.h'))) {
    throw "LlamaCppSource does not contain the pinned llama.cpp headers: $source"
}

$probeSource = Join-Path (Split-Path -Parent $PSScriptRoot) 'native\router_probe'
$runtime = (Resolve-Path -LiteralPath $LlamaRuntime).Path
$build = [IO.Path]::GetFullPath($BuildDirectory)
New-Item -ItemType Directory -Path $build -Force | Out-Null
$imports = Join-Path $build 'imports'
New-Item -ItemType Directory -Path $imports -Force | Out-Null

$gxx = (Get-Command g++.exe -ErrorAction Stop).Source
$dlltool = (Get-Command dlltool.exe -ErrorAction Stop).Source

$definitions = @{
    'llama' = 'llama.def'
    'ggml' = 'ggml.def'
    'ggml-base' = 'ggml-base.def'
}
foreach ($library in $definitions.Keys) {
    $definition = Join-Path $probeSource ('import\' + $definitions[$library])
    $output = Join-Path $imports ("lib$library.dll.a")
    & $dlltool -d $definition -D "$library.dll" -l $output
    if ($LASTEXITCODE -ne 0) {
        throw "import-library generation failed for $library.dll"
    }
}

& cmake.exe `
    -S $probeSource `
    -B $build `
    -G Ninja `
    "-DCMAKE_BUILD_TYPE=$BuildType" `
    "-DCMAKE_CXX_COMPILER=$gxx" `
    "-DLLAMA_CPP_SOURCE=$source" `
    "-DLLAMA_RUNTIME=$runtime" `
    "-DLLAMA_IMPORT_DIR=$imports"
if ($LASTEXITCODE -ne 0) {
    throw "router-probe configure failed with exit code $LASTEXITCODE"
}

& cmake.exe --build $build --config $BuildType
if ($LASTEXITCODE -ne 0) {
    throw "router-probe build failed with exit code $LASTEXITCODE"
}

$binary = Join-Path $runtime 'expertflow-router-probe.exe'
if (-not (Test-Path -LiteralPath $binary)) {
    throw "router-probe build succeeded but no executable was found at $binary"
}

$binary
