# Build self-contained kernel scripts and push every kernel under kaggle/.
# Username: $env:KAGGLE_USERNAME, else `kaggle config view`, else ~/.kaggle/kaggle.json.
param([string]$Kernel = "*", [string]$Accelerator = "NvidiaTeslaT4")
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$user = $env:KAGGLE_USERNAME
if (-not $user) {
    $line = kaggle config view 2>$null | Select-String "username:\s*(\S+)"
    if ($line) { $user = $line.Matches[0].Groups[1].Value }
}
if (-not $user -and (Test-Path "$HOME\.kaggle\kaggle.json")) {
    $user = (Get-Content "$HOME\.kaggle\kaggle.json" | ConvertFrom-Json).username
}
if (-not $user) { throw "Kaggle username not found (set KAGGLE_USERNAME)" }
python "$root\scripts\build_kaggle.py"
foreach ($dir in Get-ChildItem "$root\kaggle" -Directory -Filter $Kernel) {
    $meta = "$($dir.FullName)\kernel-metadata.json"
    $json = Get-Content $meta -Raw
    $json = $json -replace '"id":\s*"[^/"]+/', "`"id`": `"$user/"
    $json = $json -replace 'KAGGLE_USERNAME/', "$user/"
    Set-Content $meta $json -NoNewline
    kaggle kernels push -p $dir.FullName --accelerator $Accelerator
    $slug = ($json | ConvertFrom-Json).id
    Write-Host "https://www.kaggle.com/code/$slug"
}
