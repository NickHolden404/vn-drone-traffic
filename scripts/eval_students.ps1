# Evaluate students on the frozen sets and run the pre-declared paired comparisons.
param([string]$Seeds = "0,1,2")
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
function Run([string]$what) {
    Write-Host ">>> $what"
    Invoke-Expression $what
    if ($LASTEXITCODE -ne 0) { throw "failed: $what" }
}
$names = @()
foreach ($mix in "src", "src_pseudo", "pseudo") { foreach ($size in "n", "s", "m") { $names += "$mix-yolo11$size-s0" } }
foreach ($seed in $Seeds.Split(",")) { if ($seed -ne "0") { $names += "src-yolo11s-s$seed", "src_pseudo-yolo11s-s$seed" } }
$done = @{ "dronefreak-yolo11s" = $true }  # baseline, always available
foreach ($n in $names) {
    # exit 0 only when the job is trained and finalised; non-zero is expected, so not via Run
    & uv run --extra ml cvt train done --name $n
    if ($LASTEXITCODE -ne 0) { Write-Host "skip ${n}: not trained"; continue }
    $done[$n] = $true
    Run "uv run --extra ml cvt eval --model $n --sets T-VD,T-VN --imgsz 1280"
}
function ComparePair([string]$a, [string]$b) {
    if (-not ($done[$a] -and $done[$b])) { Write-Host "skip compare ${a} vs ${b}: not both trained"; return }
    Run "uv run --extra ml cvt compare --a $a --b $b"
}
foreach ($seed in $Seeds.Split(",")) { ComparePair "src_pseudo-yolo11s-s$seed" "src-yolo11s-s$seed" }
ComparePair "src-yolo11s-s0" "dronefreak-yolo11s"
ComparePair "pseudo-yolo11s-s0" "src-yolo11s-s0"
Run "uv run --extra ml cvt report"
