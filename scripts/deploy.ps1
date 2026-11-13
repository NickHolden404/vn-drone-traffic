# Deployment: export the TensorRT engines (GPU), benchmark latency (GPU), evaluate every
# grid point on T-VN and T-VD (GPU), then build the Pareto table / plot and the results page (CPU).
#   powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Step export   # + register
#   powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Step bench
#   powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Step eval
#   powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1 -Step report
#   powershell -ExecutionPolicy Bypass -File scripts\deploy.ps1                 # all four
#   ... -Step bench -Only "src_pseudo-yolo11n-s0@640-fp16,src_pseudo-yolo11n-s0@640-int8"
# -Only limits export / bench / eval to those registry names (names: `cvt deploy list`). Every step
# is resumable: up-to-date engines are skipped, benchmarked points are skipped, and `cvt eval` reads
# its prediction cache. Close other GPU applications while benchmarking (gpu_mem_mb is device-wide).
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log.
param([ValidateSet("export", "bench", "eval", "report", "all")][string]$Step = "all",
      [string[]]$Only = @())
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
$sel = @($Only | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
$models = "configs/models.yaml,configs/models_deploy.yaml"

# every grid point as "name imgsz" (the grid lives in configs/deploy.yaml only)
function Get-Points {
    $lines = @(& uv run --extra ml cvt deploy list)
    if ($LASTEXITCODE -ne 0) { throw "failed: uv run --extra ml cvt deploy list" }
    $points = @()
    foreach ($line in $lines) {
        $parts = ([string]$line).Trim() -split "\s+"
        if ($parts.Count -ne 2) { throw "unexpected line from cvt deploy list: '$line'" }
        $points += , @($parts[0], $parts[1])
    }
    if ($sel.Count -gt 0) {
        $known = @($points | ForEach-Object { $_[0] })
        foreach ($o in $sel) {
            if ($known -notcontains $o) { throw "-Only ${o}: not a grid point ($($known -join ', '))" }
        }
        $points = @($points | Where-Object { $sel -contains $_[0] })
    }
    return , $points
}

if ($Step -eq "export" -or $Step -eq "all") {
    if ($sel.Count -eq 0) {
        Run "uv run --extra ml cvt deploy export"
    } else {
        foreach ($p in (Get-Points)) {
            $name = $p[0]
            if ($name -like "*fp32*" -or $name -like "*+tiles*") {
                Write-Host "--- ${name}: not an engine (FP32 reference or tiled point), nothing to export"
            } else {
                Run "uv run --extra ml cvt deploy export --only '$name'"
            }
        }
    }
    Run "uv run --extra ml cvt deploy register"
}
if ($Step -eq "bench" -or $Step -eq "all") {
    if (-not (Test-Path "configs/models_deploy.yaml")) {
        throw "missing configs/models_deploy.yaml: run this script with -Step export first"
    }
    if ($sel.Count -eq 0) {
        Run "uv run --extra ml cvt deploy bench"
    } else {
        Run "uv run --extra ml cvt deploy bench --only '$($sel -join ',')'"
    }
}
if ($Step -eq "eval" -or $Step -eq "all") {
    if (-not (Test-Path "configs/models_deploy.yaml")) {
        throw "missing configs/models_deploy.yaml: run this script with -Step export first"
    }
    foreach ($p in (Get-Points)) {
        # tiled points: full-frame 1280 (the tile size is part of the registry entry)
        Run "uv run --extra ml cvt eval --model '$($p[0])' --sets T-VN,T-VD --imgsz $($p[1]) --models $models"
    }
}
if ($Step -eq "report" -or $Step -eq "all") {
    Run "uv run --extra ml cvt deploy report"
    Run "uv run --extra ml cvt report"
}
