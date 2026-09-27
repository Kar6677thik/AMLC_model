param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [string]$RunId = 'optimized-002',
    [ValidateSet('gpu', 'cpu')][string]$Backend = 'gpu',
    [ValidateSet('Evaluate', 'Predict', 'All')][string]$Phase = 'Evaluate',
    [string]$Python = 'python',
    [string]$Work = '',
    [ValidateRange(0.0, 1.0)][double]$MinDevScore = 0.8404988085983199,
    [ValidateRange(0.01, 24.0)][double]$MaxScoringHours = 3.0,
    [switch]$AuditRetrieval,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid RunId.' }
$RepoRoot = Split-Path $PSScriptRoot -Parent
$PackageRoot = Join-Path $RepoRoot 'code/business_entity_resolution'
$VenvRoot = Join-Path $RepoRoot '.venv'
$RunPath = Join-Path $RepoRoot "runs/$RunId"
$OutputPath = Join-Path $RepoRoot "output/$RunId"
$ConfigPath = Join-Path $PackageRoot "configs/optimized-$Backend.json"
$DataPath = (Resolve-Path -LiteralPath $Dataset).Path
if (-not $Work) { $Work = Join-Path $RepoRoot 'artifacts/baseline' }
Set-Location -LiteralPath $RepoRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable $($Arguments -join ' ')" }
}

if (-not (Test-Path -LiteralPath (Join-Path $VenvRoot 'Scripts/python.exe'))) {
    Invoke-Checked -Executable $Python -Arguments @('-m', 'venv', $VenvRoot)
}
$Runtime = Join-Path $VenvRoot 'Scripts/python.exe'
if (-not $SkipInstall) {
    $InstallTarget = if ($Backend -eq 'gpu') { "$PackageRoot[gpu]" } else { $PackageRoot }
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'pip', 'install', '-e', $InstallTarget)
}
# Prevent CPU feature workers from each launching large native thread pools.
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'unittest', 'discover', '-s', (Join-Path $PackageRoot 'tests'), '-v')
if ($Backend -eq 'gpu') {
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'check-gpu')
}

if ($Phase -ne 'Predict') {
    if ($AuditRetrieval) {
        Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'prepare', '--dataset', $DataPath, '--work', $Work, '--config', $ConfigPath)
        $AuditPath = Join-Path $RepoRoot "runs/probes/$RunId-retrieval.json"
        Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'retrieval-audit', '--work', $Work, '--config', $ConfigPath, '--output', $AuditPath, '--limit', '1000')
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RunPath 'training.json'))) {
        Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'train', '--dataset', $DataPath, '--work', $Work, '--run', $RunPath, '--config', $ConfigPath)
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RunPath 'dev_report.json'))) {
        Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'evaluate', '--work', $Work, '--run', $RunPath)
    }
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'benchmark', '--dataset', $DataPath, '--work', $Work, '--run', $RunPath, '--limit', '5000')
}

$Dev = Get-Content -Raw -LiteralPath (Join-Path $RunPath 'dev_report.json') | ConvertFrom-Json
$Bench = Get-Content -Raw -LiteralPath (Join-Path $RunPath 'benchmark_test.json') | ConvertFrom-Json
$ActualRun = Get-Content -Raw -LiteralPath (Join-Path $RunPath 'run.json') | ConvertFrom-Json
$ExpectedBackend = if ($Backend -eq 'gpu') { 'xgboost' } else { 'lightgbm' }
if ($ActualRun.config.model_backend -ne $ExpectedBackend) { throw 'RunId belongs to a different backend. Use a new RunId.' }
Write-Host "Development F0.5: $($Dev.metrics.macro_f05); candidate recall: $($Dev.metrics.candidate_micro_recall)"
Write-Host "Oracle ceiling: $($Dev.metrics.oracle_macro_f05); queries/sec: $($Bench.queries_per_second)"
Write-Host "Reports: $RunPath"

if ($Phase -eq 'Evaluate') {
    Write-Host 'Evaluation phase complete. Review the reports, then rerun with -Phase Predict -SkipInstall.'
    exit 0
}
if ($Dev.metrics.macro_f05 -le $MinDevScore) { throw 'Development score did not beat the baseline. Keep the submitted baseline and inspect the report.' }
if ($null -eq $Bench.conservative_scoring_seconds_2x) { throw 'Benchmark produced no valid throughput estimate.' }
if (($Bench.conservative_scoring_seconds_2x / 3600) -gt $MaxScoringHours) {
    throw "Conservative scoring ETA exceeds $MaxScoringHours hours. Review worker/budget settings and deadline before changing -MaxScoringHours. Export/validation need extra time."
}
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'predict', '--dataset', $DataPath, '--work', $Work, '--run', $RunPath, '--output', $OutputPath)
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'validate', '--work', $Work, '--output', $OutputPath)
Write-Host "Validated outputs: $OutputPath. Run the organizer validator and retain baseline-001 as a fallback."
