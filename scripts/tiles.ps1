# Batched tile engines: export the batch-32 640-px tile engines (GPU), time the static
# batch-1 tiled points and the batched ones in ONE session (GPU, a few minutes), evaluate
# the batched points on T-VN and T-VD (GPU), then the results page (CPU).
#   powershell -ExecutionPolicy Bypass -File scripts\tiles.ps1 -Step export   # GPU ~1 h
#   powershell -ExecutionPolicy Bypass -File scripts\tiles.ps1 -Step bench    # GPU ~10 min
#   powershell -ExecutionPolicy Bypass -File scripts\tiles.ps1 -Step eval     # GPU ~40 min
#   powershell -ExecutionPolicy Bypass -File scripts\tiles.ps1 -Step report   # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\tiles.ps1                # all, in order
# bench writes reports/latency_tiles.csv (not reports/latency.csv): the reference full-frame point,
# the 7 static tiled points and the 7 batched ones, all timed in the same session so they compare.
# Points already in that file are skipped; to re-time everything in one session, delete
# reports/latency_tiles.csv first. Close other GPU applications and let the machine idle while
# benchmarking. export / eval are resumable (up-to-date engines and cached predictions are skipped).
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log. -DryRun prints every GPU command without running it.
param([ValidateSet("export", "bench", "eval", "report", "all")][string]$Step = "all",
      [switch]$DryRun)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
$env:YOLO_OFFLINE = "1"      # Ultralytics: no DNS probe, telemetry or pip-update check
$env:YOLO_AUTOINSTALL = "0"  # Ultralytics: never pip-install missing packages
function Run([string]$what) {
    Write-Host ">>> $what"
    if ($DryRun) { return }  # -DryRun: print the commands only (helper listings still run, CPU)
    Invoke-Expression $what
    if ($LASTEXITCODE -ne 0) { throw "failed: $what" }
}
# stdout lines of a cvt helper command, failing loudly
function Get-Lines([string]$what) {
    $lines = @(Invoke-Expression "uv run --extra ml $what")
    if ($LASTEXITCODE -ne 0) { throw "failed: uv run --extra ml $what" }
    return , @($lines | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ })
}
# the tiled points ("name imgsz" lines) of a grid. Get-Lines returns its array as ONE pipeline
# object (the leading comma), so assign it before piping, or the filter sees the whole list.
function Get-Tiled([string]$config) {
    $all = Get-Lines "cvt deploy list --config $config"
    return , @($all | Where-Object { $_ -like "*+tiles*" })
}
$out = "reports/latency_tiles.csv"
$reference = "src_pseudo-yolo11m-s0@1280-fp16"  # configs/tiles.yaml `reference`
# grid config -> registry yamls; static batch-1 grids first, then their batched counterparts
$grids = @(
    @("configs/deploy.yaml", "configs/models.yaml,configs/models_deploy.yaml"),
    @("configs/deploy_al_fallback.yaml", "configs/models.yaml,configs/models_al.yaml,configs/models_deploy_al_fallback.yaml"),
    @("configs/deploy_tiles.yaml", "configs/models.yaml,configs/models_deploy_tiles.yaml"),
    @("configs/deploy_tiles_al.yaml", "configs/models.yaml,configs/models_al.yaml,configs/models_deploy_tiles_al.yaml")
)
$batched = @($grids[2], $grids[3])

if ($Step -eq "export" -or $Step -eq "all") {
    # the 1280 full-frame engines already exist (skipped when up to date); new: <model>@640-<prec>-b32
    Run "uv run --extra ml cvt deploy export --config configs/deploy_tiles.yaml --models configs/models.yaml"
    Run "uv run --extra ml cvt deploy register --config configs/deploy_tiles.yaml --models configs/models.yaml"
    Run "uv run --extra ml cvt deploy export --config configs/deploy_tiles_al.yaml --models configs/models.yaml,configs/models_al.yaml"
    Run "uv run --extra ml cvt deploy register --config configs/deploy_tiles_al.yaml --models configs/models.yaml,configs/models_al.yaml"
}
if ($Step -eq "bench" -or $Step -eq "all") {
    foreach ($g in $grids) {
        $f = ($g[1] -split ",")[-1]
        if (-not (Test-Path $f) -and -not $DryRun) { throw "missing ${f}: run this script with -Step export first" }
    }
    $first = $true
    foreach ($g in $grids) {
        $tiled = Get-Tiled $g[0]
        $names = @($tiled | ForEach-Object { ($_ -split "\s+")[0] })
        if ($first) { $names = @($reference) + $names; $first = $false }
        $only = $names -join ","
        Run "uv run --extra ml cvt deploy bench --config $($g[0]) --models $($g[1]) --only '$only' --out $out"
    }
}
if ($Step -eq "eval" -or $Step -eq "all") {
    foreach ($g in $batched) {
        $tiled = Get-Tiled $g[0]
        foreach ($line in $tiled) {
            $p = $line -split "\s+"
            if ($p.Count -ne 2) { throw "unexpected line from cvt deploy list: '$line'" }
            Run "uv run --extra ml cvt eval --model '$($p[0])' --sets T-VN,T-VD --imgsz $($p[1]) --models $($g[1])"
        }
    }
}
if ($Step -eq "report" -or $Step -eq "all") {
    Run "uv run --extra ml cvt report"
}
