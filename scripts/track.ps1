# Tracking: cache detections once (GPU), tune ByteTrack / BoT-SORT on MOT-tune (CPU),
# then track, evaluate and compare on T-MOT and T-UAVDT (CPU) and regenerate the results tables.
#   powershell -ExecutionPolicy Bypass -File scripts\track.ps1 -Step detect   # GPU
#   powershell -ExecutionPolicy Bypass -File scripts\track.ps1 -Step tune     # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\track.ps1 -Step eval     # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\track.ps1                # all three
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log.
param([ValidateSet("detect", "tune", "eval", "all")][string]$Step = "all",
      [string]$Model = "src_pseudo-yolo11s-s0")
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
$trackers = "bytetrack", "botsort"
$sets = "T-MOT,T-UAVDT"
if ($Step -eq "detect" -or $Step -eq "all") {
    # caches the detections the trackers read, and reports detection mAP on these sets
    Run "uv run --extra ml cvt eval --model $Model --sets MOT-tune,T-MOT,T-UAVDT --imgsz 1280"
}
if ($Step -eq "tune" -or $Step -eq "all") {
    foreach ($t in $trackers) {
        Run "uv run --extra ml cvt track tune --model $Model --tracker-type $t --out configs/trackers/$t.tuned.yaml"
    }
}
if ($Step -eq "eval" -or $Step -eq "all") {
    foreach ($t in $trackers) {
        $cfg = "configs/trackers/$t.tuned.yaml"
        if (-not (Test-Path $cfg)) { throw "missing ${cfg}: run this script with -Step tune first" }
        Run "uv run --extra ml cvt track run --model $Model --tracker $cfg --sets $sets"
        Run "uv run --extra ml cvt track eval --model $Model --tracker $cfg --sets $sets --scenes configs/scenes"
        # per-cell GT vs predicted counts behind the count rows -> reports/counts_<model>+<tracker>.csv
        Run "uv run --extra ml cvt track counts --model $Model --tracker $cfg --sets $sets --scenes configs/scenes"
    }
    Run "uv run --extra ml cvt track compare --a $Model+botsort-tuned --b $Model+bytetrack-tuned --sets $sets"
    Run "uv run --extra ml cvt report"
}
