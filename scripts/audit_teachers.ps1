# Teacher audit. Gate and thresholds use VD-tune only; T-VD/T-VN are reported.
# Re-runnable in pieces: predictions are cached, so -Models picks what to (re)do this time.
param([string]$Sets = "VD-tune,T-VD,T-VN",
      [string]$Models = "teacher-yolo11x,teacher-rtdetrx,teacher-yolo11x-tiled,teacher-rtdetrx-tiled,teacher-fused",
      [switch]$SkipCompare)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
function Run([string]$what) {
    Write-Host ">>> $what"
    Invoke-Expression $what
    if ($LASTEXITCODE -ne 0) { throw "failed: $what" }
}
foreach ($m in $Models.Split(",")) {
    Run "uv run --extra ml cvt eval --model $m --sets $Sets --imgsz 1280"
}
if ($SkipCompare) { exit 0 }
Run "uv run --extra ml cvt compare --a teacher-fused --b teacher-yolo11x-tiled --sets VD-tune"
Run "uv run --extra ml cvt compare --a teacher-fused --b teacher-rtdetrx-tiled --sets VD-tune"
Run "uv run --extra ml cvt compare --a teacher-yolo11x-tiled --b teacher-yolo11x --sets VD-tune,T-VN"
Run "uv run --extra ml cvt thresholds --model teacher-fused --set VD-tune --out configs/pseudo_thresholds.yaml"
# per-member thresholds: the disagreement score compares members on their own calibrated scores
Run "uv run --extra ml cvt thresholds --model teacher-yolo11x-tiled --set VD-tune --out configs/thresholds/teacher-yolo11x-tiled.yaml"
Run "uv run --extra ml cvt thresholds --model teacher-rtdetrx-tiled --set VD-tune --out configs/thresholds/teacher-rtdetrx-tiled.yaml"
Run "uv run --extra ml cvt report"
