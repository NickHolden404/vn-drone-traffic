# Train in time-boxed pieces: at most -Hours per invocation (stops at the last epoch boundary
# that fits, overrunning by at most one epoch); the next invocation resumes exactly there.
#   powershell -ExecutionPolicy Bypass -File scripts\train.ps1 -Group teachers -Hours 4
#   powershell -ExecutionPolicy Bypass -File scripts\train.ps1 -Status
# Ctrl+C: wait for the process to stop; at most the current epoch is lost (a torn last.pt is
# replaced by last.prev.pt on the next run).
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log.
# Smoke check: $env:CVT_MIN_START_S = "0" lets a tiny -Hours start a piece.
param([string]$Group = "teachers", [double]$Hours = 4, [switch]$Status)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
$env:YOLO_OFFLINE = "1"      # Ultralytics: no DNS probe, telemetry or pip-update check
$env:YOLO_AUTOINSTALL = "0"  # Ultralytics: never pip-install missing packages
function Run([string]$what) {
    Write-Host ">>> $what"
    Invoke-Expression $what
    if ($LASTEXITCODE -ne 0) { throw "failed: $what" }
}
if (-not $Status) {
    if (($Group -eq "teachers" -or $Group -eq "smoke") -and -not (Test-Path "configs/train/teacher.yaml")) {
        Run "uv run --extra ml cvt yolo export --sets VD-train,VD-tune"
        Run "uv run --extra ml cvt yolo yaml --out configs/train/teacher.yaml --train VD-train --val VD-tune"
    }
    Run "uv run --extra ml cvt train run --group $Group --hours $Hours"
}
Run "uv run --extra ml cvt train status"
