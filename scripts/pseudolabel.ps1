# Pseudo-label the pool with the gated detector (default teacher-fused), export YOLO data,
# and write the three student dataset yamls (all validate on VD-tune + P-val).
param([string]$Detector = "teacher-fused")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
function Run([string]$what) {
    Write-Host ">>> $what"
    Invoke-Expression $what
    if ($LASTEXITCODE -ne 0) { throw "failed: $what" }
}
$out = "data/pseudo/$Detector"
Run "uv run --extra ml cvt pseudolabel --model $Detector --out $out"
Run "uv run --extra ml cvt yolo export --sets P-train,P-val --pseudo $out"
$p = "P-train-$Detector"; $v = "VD-tune,P-val-$Detector"
Run "uv run --extra ml cvt yolo yaml --out configs/train/src.yaml --train VD-train --val $v"
Run "uv run --extra ml cvt yolo yaml --out configs/train/src_pseudo.yaml --train VD-train,$p --val $v"
Run "uv run --extra ml cvt yolo yaml --out configs/train/pseudo.yaml --train $p --val $v"
