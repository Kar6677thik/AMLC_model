param(
    [Parameter(Mandatory = $true)][string]$Dataset,
    [string]$StudyId = 'quality-004',
    [ValidateSet('Evaluate', 'Predict', 'All')][string]$Phase = 'Evaluate',
    [string]$SourceRun = '',
    [string]$Work = '',
    [string]$Python = 'python',
    [ValidateRange(1, 12)][int]$Workers = 6,
    [string]$Deadline = '2026-09-27T23:59:00+05:30',
    [ValidateRange(0, 240)][int]$ReserveMinutes = 60,
    [ValidateRange(1.0, 3.0)][double]$TimingMargin = 1.35,
    [switch]$SkipInstall
)
$ErrorActionPreference = 'Stop'
if ($StudyId -notmatch '^[A-Za-z0-9_-]+$') { throw 'Invalid StudyId.' }
$RepoRoot = Split-Path $PSScriptRoot -Parent
$PackageRoot = Join-Path $RepoRoot 'code/business_entity_resolution'
$StudyPath = Join-Path $RepoRoot "runs/$StudyId"
$OutputPath = Join-Path $RepoRoot "output/$StudyId"
$DataPath = (Resolve-Path -LiteralPath $Dataset).Path
if (-not $Work) { $Work = Join-Path $RepoRoot 'artifacts/baseline' }
if (-not $SourceRun) { $SourceRun = Join-Path $RepoRoot 'runs/optimized-002' }
$Work = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Work)
$SourceRun = (Resolve-Path -LiteralPath $SourceRun).Path
[DateTimeOffset]::Parse($Deadline) | Out-Null
Set-Location -LiteralPath $RepoRoot

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed: $Executable $($Arguments -join ' ')" }
}

$VenvRoot = Join-Path $RepoRoot '.venv'
$Runtime = Join-Path $VenvRoot 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $Runtime)) { Invoke-Checked -Executable $Python -Arguments @('-m', 'venv', $VenvRoot) }
if (-not $SkipInstall) { Invoke-Checked -Executable $Runtime -Arguments @('-m', 'pip', 'install', '-e', "${PackageRoot}[gpu]") }
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '1'
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'unittest', 'discover', '-s', (Join-Path $PackageRoot 'tests'), '-v')

if ($Phase -ne 'Predict') {
    Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'quality-sweep', '--dataset', $DataPath, '--work', $Work,
        '--source-run', $SourceRun, '--output', $StudyPath, '--deadline', $Deadline, '--reserve-minutes', "$ReserveMinutes",
        '--margin', $TimingMargin.ToString([Globalization.CultureInfo]::InvariantCulture), '--workers', "$Workers")
    Write-Host "Study report: $StudyPath/quality.json"
    if ($Phase -eq 'Evaluate') {
        Write-Host 'Review quality.json. If a model is selected, run the same command with -Phase Predict -SkipInstall.'
        exit 0
    }
}
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'quality-check', '--output', $StudyPath)
$Study = Get-Content -Raw -LiteralPath (Join-Path $StudyPath 'quality.json') | ConvertFrom-Json
$RunPath = Join-Path $StudyPath $Study.selected
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'predict', '--dataset', $DataPath, '--work', $Work, '--run', $RunPath, '--output', $OutputPath)
Invoke-Checked -Executable $Runtime -Arguments @('-m', 'ber', 'validate', '--work', $Work, '--output', $OutputPath)
Write-Host "Validated outputs: $OutputPath; selected model assets: $RunPath"
