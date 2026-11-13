# Demo video: detect, track (tuned BoT-SORT), stabilize and count on a segment of
# Vietnamese drone footage; writes reports/demo/demo.mp4, three PNG stills and the results-page still
# reports/demo/demo_still.jpg (the middle still, JPEG quality 88) (GPU for detection).
# Needs the tuned tracker (scripts\track.ps1 -Step tune) and a scene file with the counting line(s)
# in pixels of the segment's first frame (see configs\scenes\README.md).
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -Start 120 -Duration 30
# The mp4 is written at the source video's frame rate unless -FpsOut is given (> 0).
# -Out sets the mp4 path (stills go next to it); use data/demo_out/ for third-party videos so
# nothing from them lands under reports/ (git-tracked).
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log.
param([string]$Video = "data/external/uitadrone/videos/DJI_0143.MP4",
      [double]$Start = 60,
      [double]$Duration = 60,
      [string]$Scene = "configs/scenes/demo.yaml",
      [string]$Tracker = "configs/trackers/botsort.tuned.yaml",
      [string]$Model = "src_pseudo-yolo11s-s0",
      [double]$FpsOut = 0,
      [string]$Out = "reports/demo/demo.mp4")
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
if (-not (Test-Path $Video)) { throw "missing ${Video}" }
if (-not (Test-Path $Scene)) { throw "missing ${Scene}: write the counting line(s) first (configs/scenes/README.md)" }
if (-not (Test-Path $Tracker)) { throw "missing ${Tracker}: run scripts/track.ps1 -Step tune first" }
$fps = ""
if ($FpsOut -gt 0) { $fps = "--fps-out $FpsOut" }
Run "uv run --extra ml cvt demo --video '$Video' --start $Start --duration $Duration --scene '$Scene' --tracker '$Tracker' --model $Model --out '$Out' $fps"
