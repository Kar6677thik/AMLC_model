param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [string]$RunId = 'baseline-001',
    [ValidateSet('learned', 'rules')][string]$Mode = 'learned',
    [string]$Python = 'python',
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path $PSScriptRoot -Parent
$PackageRoot = Join-Path $RepoRoot 'code/business_entity_resolution'
$VenvRoot = Join-Path $RepoRoot '.venv'
$RunPath = Join-Path $RepoRoot "runs/$RunId"
$WorkPath = Join-Path $RepoRoot 'artifacts/baseline'
$OutputPath = Join-Path $RepoRoot "output/$RunId"
$ConfigPath = Join-Path $PackageRoot 'configs/baseline.json'
$DataPath = (Resolve-Path -LiteralPath $Dataset).Path

if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'RunId may contain only letters, digits, underscore, and hyphen.' }
if (-not (Test-Path -LiteralPath (Join-Path $DataPath 'train/train_source1.tsv'))) { throw 'Dataset must contain train/ and test/ folders.' }

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed with exit code $LASTEXITCODE : $Executable $($Arguments -join ' ')" }
}

Set-Location -LiteralPath $RepoRoot
Write-Host 'Hardware inventory (no GPU is required for this baseline):'
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion | Format-Table
Get-PSDrive -PSProvider FileSystem | Select-Object Name, Used, Free | Format-Table

if (-not (Test-Path -LiteralPath (Join-Path $VenvRoot 'Scripts/python.exe'))) {
    Invoke-Checked -Executable $Python -Arguments @('-m', 'venv', $VenvRoot)
}
$Runtime = Join-Path $VenvRoot 'Scripts/python.exe'
if (-not $SkipInstall) {
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'pip', 'install', '-e', $PackageRoot)
}

# Only synthetic fixtures are used here. Stop before real data if tests fail.
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'unittest', 'discover', '-s', (Join-Path $PackageRoot 'tests'), '-v')

if (-not (Test-Path -LiteralPath (Join-Path $RunPath 'training.json'))) {
    if (Test-Path -LiteralPath $RunPath) { throw 'An incomplete training run exists. Use a new RunId; inspect the failed run before removing anything.' }
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'train', '--dataset', $DataPath, '--work', $WorkPath, '--run', $RunPath, '--config', $ConfigPath, '--mode', $Mode)
} else {
    $PreviousRun = Get-Content -Raw -LiteralPath (Join-Path $RunPath 'run.json') | ConvertFrom-Json
    if ($PreviousRun.mode -ne $Mode) { throw 'RunId already belongs to a different model mode.' }
    Write-Host 'Reusing completed training run.'
}
if (-not (Test-Path -LiteralPath (Join-Path $RunPath 'dev_report.json'))) {
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'evaluate', '--work', $WorkPath, '--run', $RunPath)
}
if (-not (Test-Path -LiteralPath (Join-Path $RunPath 'benchmark_test.json'))) {
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'benchmark', '--dataset', $DataPath, '--work', $WorkPath, '--run', $RunPath, '--limit', '1000')
}
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'predict', '--dataset', $DataPath, '--work', $WorkPath, '--run', $RunPath, '--output', $OutputPath)
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'validate', '--work', $WorkPath, '--output', $OutputPath)

Write-Host "Baseline finished. Outputs: $OutputPath"
Write-Host 'Review runs/<RunId>/dev_report.json and output/<RunId>/validation.json before uploading.'
Write-Host 'Run the organizer validator as documented, then package with team member names.'
