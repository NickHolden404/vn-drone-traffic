# Traffic analytics on the full-length drone clips (data/external/uitadrone/videos): detect
# every frame (GPU), track + stabilize, per-clip scale / origin-destination / events / maps (CPU),
# the anomaly evaluation, the rendered overlays and the results page.
#   powershell -ExecutionPolicy Bypass -File scripts\analytics.ps1 -Step detect   # GPU
#   powershell -ExecutionPolicy Bypass -File scripts\analytics.ps1 -Step track    # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\analytics.ps1 -Step analyze  # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\analytics.ps1 -Step eval     # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\analytics.ps1 -Step render   # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\analytics.ps1 -Step report   # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\analytics.ps1                # all, in order
# -Clips a,b restricts detect / track / analyze / render to those clips (file names without
# extension). Every step is resumable per clip: detections and tracks are cached per clip in
# data/cache/analytics, and a finished clip is skipped; a clip interrupted mid-way (e.g. during
# detect) restarts from its first frame. Your own clips are never committed.
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log. -DryRun prints every command without running it.
param([ValidateSet("detect", "track", "analyze", "eval", "render", "report", "all")]
      [string]$Step = "all",
      [string]$Clips = "",
      [switch]$DryRun)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
$env:YOLO_OFFLINE = "1"      # Ultralytics: no DNS probe, telemetry or pip-update check
$env:YOLO_AUTOINSTALL = "0"  # Ultralytics: never pip-install missing packages
function Run([string]$what) {
    Write-Host ">>> $what"
    if ($DryRun) { return }  # -DryRun: print the commands only
    Invoke-Expression $what
    if ($LASTEXITCODE -ne 0) { throw "failed: $what" }
}
$config = "configs/analytics.yaml"
$only = if ($Clips) { " --clips '$Clips'" } else { "" }

if ($Step -eq "detect" -or $Step -eq "all") {
    Run "uv run --extra ml cvt analytics detect --config $config$only"
}
if ($Step -eq "track" -or $Step -eq "all") {
    Run "uv run --extra ml cvt analytics track --config $config$only"
}
if ($Step -eq "analyze" -or $Step -eq "all") {
    Run "uv run --extra ml cvt analytics analyze --config $config$only"
}
if ($Step -eq "eval" -or $Step -eq "all") {
    Run "uv run --extra ml cvt analytics eval --config $config"
}
if ($Step -eq "render" -or $Step -eq "all") {
    Run "uv run --extra ml cvt analytics render --config $config$only"
}
if ($Step -eq "report" -or $Step -eq "all") {
    Run "uv run --extra ml cvt analytics stills --config $config --out reports/demo"
    Run "uv run --extra ml cvt report"
}
