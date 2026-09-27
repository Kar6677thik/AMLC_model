param(
    [ValidateSet('Study', 'Apply', 'Audit')][string]$Phase = 'Study',
    [string]$RunId = 'balanced-005',
    [string]$StudyId = 'refinement-006',
    [string]$Work = '',
    [string]$Original = ''
)

$ErrorActionPreference = 'Stop'
if ($RunId -notmatch '^[A-Za-z0-9_-]+$' -or $StudyId -notmatch '^[A-Za-z0-9_-]+$') {
    throw 'RunId and StudyId must contain only letters, digits, hyphens, or underscores.'
}
$RepoRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $RepoRoot
$Runtime = Join-Path $RepoRoot '.venv/Scripts/python.exe'
$Program = Join-Path $RepoRoot 'code/refinement/refine.py'
$StudyPath = Join-Path $RepoRoot "runs/$StudyId"
$RunPath = Join-Path $RepoRoot "runs/$RunId"
if (-not $Work) { $Work = Join-Path $RepoRoot 'artifacts/baseline' }
if (-not $Original) { $Original = Join-Path $RepoRoot "output/$RunId" }
if (-not (Test-Path -LiteralPath $Runtime)) { throw 'Use the compute PC with the existing .venv.' }

function Invoke-Checked {
    param([string[]]$Arguments)
    & $Runtime @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Refinement command failed: $($Arguments -join ' ')" }
}

Invoke-Checked -Arguments @('-m', 'unittest', 'discover', '-s', 'code/refinement', '-v')
switch ($Phase) {
    'Study' {
        Invoke-Checked -Arguments @($Program, 'study', '--work', $Work, '--run', $RunPath, '--output', $StudyPath)
        $Result = Get-Content -Raw -LiteralPath (Join-Path $StudyPath 'study.json') | ConvertFrom-Json
        Write-Host "Selected: $($Result.selected). Review $StudyPath/study.json."
        if ($Result.selected -eq 'baseline') { Write-Host 'No rule passed the gain gates. Keep the completed original output.' }
    }
    'Apply' {
        Invoke-Checked -Arguments @($Program, 'apply', '--work', $Work, '--run', $RunPath,
            '--original', $Original, '--policy', (Join-Path $StudyPath 'policy.json'),
            '--output', (Join-Path $RepoRoot "output/$StudyId"))
        Write-Host "Review output/$StudyId/refinement.json, run the organizer validator, and use the separate refinement packager."
    }
    'Audit' {
        Invoke-Checked -Arguments @($Program, 'audit', '--work', $Work, '--output', (Join-Path $RepoRoot "runs/$StudyId-audit"))
    }
}
