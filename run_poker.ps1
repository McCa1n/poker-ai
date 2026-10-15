# A project-local venv is checked in the SAME shell that launches the game.
# Missing/incomplete packages are repaired here, not merely in an agent session.
# Forward all arguments, e.g. .\run_poker.ps1 train --epochs 200.
$ErrorActionPreference = 'Stop'
$pokerArguments = @($args)
$pokerEnvironment = Join-Path $PSScriptRoot '.venv'
$pokerPython = Join-Path $pokerEnvironment 'Scripts\python.exe'
$pokerRequirements = Join-Path $PSScriptRoot 'requirements-poker.txt'

function Invoke-PokerProbe {
    param([string]$Executable, [string]$Code)
    # Windows PowerShell 5.1 otherwise treats native stderr as a terminating error.
    $pokerProbePreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $pokerProbeOutput = @(& $Executable -I -B -X utf8 -c $Code $pokerEnvironment 2>&1)
        return @{ Success = ($LASTEXITCODE -eq 0); Details = ($pokerProbeOutput -join "`n") }
    }
    catch { return @{ Success = $false; Details = $_.ToString() } }
    finally { $ErrorActionPreference = $pokerProbePreference }
}

$pokerRuntimeCheck = "import sys,pathlib; assert pathlib.Path(sys.prefix).resolve() == pathlib.Path(sys.argv[1]).resolve(), 'Wrong virtual environment'"
$pokerRuntimeReady = $false
if (Test-Path -LiteralPath $pokerPython -PathType Leaf) {
    $pokerRuntimeReady = (Invoke-PokerProbe $pokerPython $pokerRuntimeCheck).Success
}
if (-not $pokerRuntimeReady) {
    $pokerBasePython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
    if (-not (Test-Path -LiteralPath $pokerBasePython -PathType Leaf)) {
        $pokerCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($pokerCommand) { $pokerBasePython = $pokerCommand.Source }
        else { throw 'Python not found. Install Python 3.12 or newer, then run this script again.' }
    }
    Write-Host "Creating the poker environment: $pokerEnvironment"
    & $pokerBasePython -I -m venv $pokerEnvironment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the poker Python environment.' }
    $pokerRuntimeStatus = Invoke-PokerProbe $pokerPython $pokerRuntimeCheck
    if (-not $pokerRuntimeStatus.Success) { throw "Project Python could not start: $($pokerRuntimeStatus.Details)" }
}

$pokerPackageCheck = @'
import sys, pathlib, numpy, torch
from torch import nn
from torch.func import functional_call
root = pathlib.Path(sys.prefix).resolve()
assert root == pathlib.Path(sys.argv[1]).resolve(), 'Wrong virtual environment'
assert torch.__file__ and pathlib.Path(torch.__file__).resolve().is_relative_to(root), 'PyTorch is outside the project environment'
assert pathlib.Path(numpy.__file__).resolve().is_relative_to(root), 'NumPy is outside the project environment'
x = torch.ones(1, 2, requires_grad=True)
nn.Linear(2, 1)(x).sum().backward()
assert torch.isfinite(x.grad).all(), 'PyTorch gradient check failed'
'@
$pokerPackageStatus = Invoke-PokerProbe $pokerPython $pokerPackageCheck
if (-not $pokerPackageStatus.Success) {
    Write-Host 'Installing/repairing CPU PyTorch and NumPy in poker\.venv. This may take a few minutes.'
    $pokerInstallPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $pokerPython -I -m ensurepip --upgrade
        if ($LASTEXITCODE -ne 0) { throw 'Could not initialize pip in the poker environment.' }
        & $pokerPython -I -m pip --isolated --disable-pip-version-check install --no-user --force-reinstall --no-cache-dir --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu -r $pokerRequirements
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the download error above and run this script again.' }
    }
    finally { $ErrorActionPreference = $pokerInstallPreference }
    $pokerPackageStatus = Invoke-PokerProbe $pokerPython $pokerPackageCheck
    if (-not $pokerPackageStatus.Success) { throw "Dependency check still failed after repair: $($pokerPackageStatus.Details)" }
}

# -E/-s exclude foreign Python environment settings while retaining the script
# directory so poker_env.py and poker_learning.py remain importable.
if ($pokerArguments.Count -eq 0) {
    & $pokerPython -E -s -B -X utf8 -u (Join-Path $PSScriptRoot 'poker_ai.py') --help
}
else { & $pokerPython -E -s -B -X utf8 -u (Join-Path $PSScriptRoot 'poker_ai.py') @pokerArguments }
if ($LASTEXITCODE -ne 0) { throw "Poker program exited with code $LASTEXITCODE" }
