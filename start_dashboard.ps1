$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-cu130\Scripts\python.exe'
$app = Join-Path $PSScriptRoot 'dashboard.py'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing isolated Python environment: $python"
}
if (-not (Test-Path -LiteralPath $app)) {
    throw "Missing dashboard source: $app"
}
& $python $app --server-name 127.0.0.1 --server-port 7860
