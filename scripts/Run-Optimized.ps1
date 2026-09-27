param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [string]$RunId = 'optimized-002',
    [ValidateSet('gpu', 'cpu')][string]$Backend = 'gpu',
    [ValidateSet('Evaluate', 'Predict', 'All')][string]$Phase = 'Evaluate',
    [string]$Python = 'python',
    [string]$Work = '',
    [string]$Config = '',
    [ValidateRange(0.0, 1.0)][double]$MinDevScore = 0.8404988085983199,
    [ValidateRange(0.01, 24.0)][double]$MaxScoringHours = 3.0,
    [ValidateRange(1.0, 3.0)][double]$TimingMargin = 2.0,
    [string]$Deadline = '',
    [ValidateRange(0, 240)][int]$ReserveMinutes = 60,
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
if ($Config) { $ConfigPath = (Resolve-Path -LiteralPath $Config).Path }
$DataPath = (Resolve-Path -LiteralPath $Dataset).Path
if (-not $Work) { $Work = Join-Path $RepoRoot 'artifacts/baseline' }
$Work = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Work)
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

if (Test-Path -LiteralPath (Join-Path $RunPath 'run.json')) {
    $Recorded = Get-Content -Raw -LiteralPath (Join-Path $RunPath 'run.json') | ConvertFrom-Json
    $Requested = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
    foreach ($Property in $Requested.PSObject.Properties) {
        if ($Recorded.config.($Property.Name) -ne $Property.Value) {
            throw "RunId uses a different configuration ($($Property.Name)). Use a new RunId."
        }
    }
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
if ($Dev.metrics.macro_f05 -le $MinDevScore) { throw "Development score did not exceed the required floor ($MinDevScore). Keep the submitted baseline and inspect the report." }
if ($null -eq $Bench.extrapolated_scoring_seconds -or $Bench.extrapolated_scoring_seconds -le 0) { throw 'Benchmark produced no valid throughput estimate.' }
$EstimatedSeconds = $Bench.extrapolated_scoring_seconds * $TimingMargin
if (($EstimatedSeconds / 3600) -gt $MaxScoringHours) {
    throw "Scoring ETA with ${TimingMargin}x margin exceeds $MaxScoringHours hours. Export/validation need extra time."
}
if ($Deadline) {
    $RemainingSeconds = ([DateTimeOffset]::Parse($Deadline) - [DateTimeOffset]::Now).TotalSeconds - $ReserveMinutes * 60
    if ($EstimatedSeconds -gt $RemainingSeconds) {
        throw "Scoring ETA with ${TimingMargin}x margin cannot meet $Deadline while reserving $ReserveMinutes minutes for release. Keep the existing submission."
    }
}
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'predict', '--dataset', $DataPath, '--work', $Work, '--run', $RunPath, '--output', $OutputPath)
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'validate', '--work', $Work, '--output', $OutputPath)
Write-Host "Validated outputs: $OutputPath. Run the organizer validator and retain baseline-001 as a fallback."
