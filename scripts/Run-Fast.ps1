param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [string]$RunId = 'optimized-003',
    [ValidateSet('Probe', 'Evaluate', 'Predict', 'All')][string]$Phase = 'Probe',
    [string]$Python = 'python',
    [string]$Work = '',
    [ValidateRange(1.0, 10000.0)][double]$MinQueriesPerSecond = 150.0,
    [ValidateRange(0.0, 1.0)][double]$MinDevScore = 0.8622980669402457,
    [string]$Deadline = '2026-09-27T23:59:00+05:30',
    [ValidateRange(0, 240)][int]$ReserveMinutes = 60,
    [ValidateRange(1.0, 3.0)][double]$TimingMargin = 1.35,
    [switch]$SkipInstall
)

$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid RunId.' }
$RepoRoot = Split-Path $PSScriptRoot -Parent
$PackageRoot = Join-Path $RepoRoot 'code/business_entity_resolution'
$ReferenceConfig = Join-Path $PackageRoot 'configs/optimized-gpu.json'
$ProbePath = Join-Path $RepoRoot "runs/probes/$RunId-fast"
$SelectedConfig = Join-Path $ProbePath 'selected_config.json'
$ProbeReport = Join-Path $ProbePath 'probe.json'
$DataPath = (Resolve-Path -LiteralPath $Dataset).Path
if (-not $Work) { $Work = Join-Path $RepoRoot 'artifacts/baseline' }
# Resolve caller-relative paths before changing directory.
$Work = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Work)
[DateTimeOffset]::Parse($Deadline) | Out-Null
Set-Location -LiteralPath $RepoRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable $($Arguments -join ' ')" }
}

$VenvRoot = Join-Path $RepoRoot '.venv'
$Runtime = Join-Path $VenvRoot 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Runtime)) {
    Invoke-Checked -Executable $Python -Arguments @('-m', 'venv', $VenvRoot)
}
if (-not $SkipInstall) {
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'pip', 'install', '-e', "${PackageRoot}[gpu]")
}
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'

if ($Phase -eq 'Probe' -or $Phase -eq 'All') {
    if (Test-Path -LiteralPath $ProbePath) { throw 'Probe directory exists. Use Evaluate for a completed probe or a new RunId for a fresh probe.' }
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'unittest', 'discover', '-s', (Join-Path $PackageRoot 'tests'), '-v')
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'prepare', '--dataset', $DataPath, '--work', $Work, '--config', $ReferenceConfig, '--split', 'both')
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'retrieval-probe', '--work', $Work, '--config', $ReferenceConfig,
        '--output', $ProbePath, '--dev-limit', '2000', '--test-limit', '3000', '--min-rate', $MinQueriesPerSecond.ToString([Globalization.CultureInfo]::InvariantCulture))
    Write-Host "Retrieval report: $ProbeReport"
    if ($Phase -eq 'Probe') {
        $Result = Get-Content -Raw -LiteralPath $ProbeReport | ConvertFrom-Json
        if ($Result.recommended) {
            Write-Host "Selected $($Result.recommended). Run the same command with -Phase Evaluate -SkipInstall next."
        } else {
            Write-Host 'No configuration met the speed/quality gates. Send probe.json for review; training has not started.'
        }
        exit 0
    }
}

Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'retrieval-probe', '--work', $Work, '--output', $ProbePath, '--check')
$Probe = Get-Content -Raw -LiteralPath $ProbeReport | ConvertFrom-Json
# These are lightweight pre-training checks. The post-training gate uses a real model benchmark.
$Chosen = @($Probe.results | Where-Object { $_.name -eq $Probe.recommended })[0]
$Remaining = ([DateTimeOffset]::Parse($Deadline) - [DateTimeOffset]::Now).TotalSeconds
$ProbeEstimate = $Chosen.test.extrapolated_retrieval_feature_seconds * $TimingMargin
if ($Phase -ne 'Predict' -and ($ProbeEstimate + $ReserveMinutes * 60 + 1800) -gt $Remaining) {
    throw 'Probe ETA leaves insufficient time for an assumed 30-minute training/evaluation allowance and release buffer. Inspect probe.json and the deadline before proceeding.'
}
$NextPhase = if ($Phase -eq 'All') { 'All' } else { $Phase }
& (Join-Path $PSScriptRoot 'Run-Optimized.ps1') -Dataset $DataPath -RunId $RunId -Backend gpu -Phase $NextPhase `
    -Work $Work -Config $SelectedConfig -MinDevScore $MinDevScore -MaxScoringHours 24 `
    -TimingMargin $TimingMargin -Deadline $Deadline -ReserveMinutes $ReserveMinutes -SkipInstall
if ($LASTEXITCODE -ne 0) { throw 'Training/evaluation/prediction launcher failed.' }
