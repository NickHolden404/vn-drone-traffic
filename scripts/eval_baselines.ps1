# Zero-shot VisDrone YOLO11 baselines (dronefreak n/s/x) at two input sizes.
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts\eval_baselines.ps1                 # T-VD only
#   powershell -ExecutionPolicy Bypass -File scripts\eval_baselines.ps1 -Sets "T-VD,T-VN"  # once UIT-Drone7 images exist
# Predictions are cached in data\cache\preds, so re-running only redoes what changed.
param(
    [string]$Sets = "T-VD"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"

foreach ($v in "n", "s", "x") {
    foreach ($size in 640, 1280) {
        Write-Host "=== dronefreak-yolo11$v @ $size on $Sets ==="
        uv run --extra ml cvt eval --model "dronefreak-yolo11$v" --sets $Sets --imgsz $size
        if ($LASTEXITCODE -ne 0) { throw "eval failed: dronefreak-yolo11$v @ $size" }
    }
}
uv run --extra ml cvt report
if ($LASTEXITCODE -ne 0) { throw "report failed" }
Write-Host "Done. Results appended to reports\results.csv and docs\results.md updated."
