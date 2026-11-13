# Active learning: score the pool with the fused teacher (GPU), prepare the runs (CPU),
# train them in pieces (GPU), evaluate them (GPU), paired comparisons (CPU), the headline TensorRT
# engines (GPU), then the curve plot / CSV and the results page (CPU).
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step score     # GPU ~2 h, cached
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step prepare   # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step train -Hours 6   # repeat
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step train -Hours 6 -Group al_upper
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step eval      # GPU ~5 h
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step compare   # CPU (cached)
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step engine    # GPU ~20 min
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step fallback  # GPU ~1-1.5 h
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1 -Step report    # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\al.ps1                 # all, in order
# train: the groups al_curve, al_variance, al_upper, al_headline in that order (or only -Group),
# sharing one -Hours budget; it stops at the first group that does not finish in this invocation,
# so run it again until `cvt train status --queue configs/train_queue_al.yaml` shows every job done.
# fallback: the primary run at 1536 px and tiled (pre-declared secondary results, configs/al.yaml
# `fallbacks`; configs/deploy_al_fallback.yaml), benchmarked and evaluated on T-VN and T-VD.
# eval / compare / engine / fallback refuse runs that have not finished training (their last.pt would be mid-run).
# Every step is resumable: scoring and eval read their prediction caches, training resumes at an
# epoch boundary, up-to-date engines and benchmarked points are skipped.
# -Step all skips `score` once data/pseudo/teacher-fused-udpool/meta.json exists (-Step score forces it).
# NOTE: re-running `prepare` appends a second copy of every reveal to
# reports/al_reveals.csv (the log is per reveal; the generated files are rewritten identically),
# so -Step all runs prepare only while configs/train_queue_al.yaml does not exist.
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log.
param([ValidateSet("score", "prepare", "train", "eval", "compare", "engine", "fallback", "report", "all")]
      [string]$Step = "all",
      [double]$Hours = 4.0,
      [ValidateSet("", "al_curve", "al_variance", "al_upper", "al_headline")][string]$Group = "")
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
$queue = "configs/train_queue_al.yaml"
$models = "configs/models.yaml,configs/models_al.yaml"
$deploy = "configs/deploy_al.yaml"
$deployModels = "configs/models_deploy_al.yaml"  # never configs/models_deploy.yaml (deployment grid)
$fallback = "configs/deploy_al_fallback.yaml"
$fallbackModels = "configs/models_deploy_al_fallback.yaml"  # not with $deployModels (shared 1280 engine)
$sets = "T-VN,T-VD"

# stdout lines of a cvt helper command (list / comparisons), failing loudly
function Get-Lines([string]$what) {
    $lines = @(Invoke-Expression "uv run --extra ml $what")
    if ($LASTEXITCODE -ne 0) { throw "failed: uv run --extra ml $what" }
    return , @($lines | ForEach-Object { ([string]$_).Trim() } | Where-Object { $_ })
}
function Assert-Trained([string]$name) {
    & uv run --extra ml cvt train done --name $name --queue $queue
    if ($LASTEXITCODE -ne 0) { throw "${name}: not finished training (run this script with -Step train)" }
}

if ($Step -eq "score") {
    # tiled teachers on the 9,352 pool frames; only the per-frame `disagreement` is used
    Run "uv run --extra ml cvt pseudolabel --model teacher-fused --sets UD-pool --out data/pseudo/teacher-fused-udpool"
} elseif ($Step -eq "all") {
    if (Test-Path "data/pseudo/teacher-fused-udpool/meta.json") {
        Write-Host "--- data/pseudo/teacher-fused-udpool/meta.json exists: score skipped (-Step score re-runs it)"
    } else {
        Run "uv run --extra ml cvt pseudolabel --model teacher-fused --sets UD-pool --out data/pseudo/teacher-fused-udpool"
    }
}
if ($Step -eq "prepare") {
    # re-running re-logs every reveal in reports/al_reveals.csv (see NOTE above)
    Run "uv run --extra ml cvt al prepare"
} elseif ($Step -eq "all") {
    if (Test-Path $queue) {
        Write-Host "--- ${queue} exists: prepare skipped (-Step prepare regenerates it and re-logs every reveal)"
    } else {
        Run "uv run --extra ml cvt al prepare"
    }
}
if ($Step -eq "train" -or $Step -eq "all") {
    if (-not (Test-Path $queue)) { throw "missing ${queue}: run this script with -Step prepare first" }
    $groups = @("al_curve", "al_variance", "al_upper", "al_headline")
    if ($Group) { $groups = @($Group) }
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $finished = $true
    foreach ($g in $groups) {
        $left = [Math]::Max(0.0, $Hours - $clock.Elapsed.TotalHours)
        $h = $left.ToString("0.###", [Globalization.CultureInfo]::InvariantCulture)
        # not captured: the training output streams to the console
        Run "uv run --extra ml cvt train run --queue $queue --group $g --hours $h"
        & uv run --extra ml cvt train done --group $g --queue $queue
        if ($LASTEXITCODE -ge 2) { throw "failed: cvt train done --group $g --queue $queue" }
        if ($LASTEXITCODE -eq 1) {
            Write-Host "--- ${g} not finished: run -Step train again to continue"
            $finished = $false
            break
        }
    }
    if (-not $finished) {
        if ($Step -eq "all") { Write-Host "--- stopping before eval: training is not finished" }
        exit 0
    }
}
if ($Step -eq "eval" -or $Step -eq "all") {
    foreach ($name in (Get-Lines "cvt al list")) {
        Assert-Trained $name
        Run "uv run --extra ml cvt eval --model '$name' --sets $sets --imgsz 1280 --models $models"
    }
}
if ($Step -eq "compare" -or $Step -eq "all") {
    $alRuns = Get-Lines "cvt al list"
    foreach ($line in (Get-Lines "cvt al comparisons")) {
        $ab = $line -split "\s+"
        if ($ab.Count -ne 2) { throw "unexpected line from cvt al comparisons: '$line'" }
        # starting students (src_pseudo-*) are not AL runs; AL runs must have finished training
        foreach ($n in $ab) { if ($alRuns -contains $n) { Assert-Trained $n } }
        Run "uv run --extra ml cvt compare --a '$($ab[0])' --b '$($ab[1])' --sets $sets --models $models"
    }
}
if ($Step -eq "engine" -or $Step -eq "all") {
    $points = Get-Lines "cvt deploy list --config $deploy"
    foreach ($line in $points) { Assert-Trained (($line -split "\s+")[0] -split "@")[0] }
    Run "uv run --extra ml cvt deploy export --config $deploy --models $models"
    Run "uv run --extra ml cvt deploy register --config $deploy --models $models --out $deployModels"
    Run "uv run --extra ml cvt deploy bench --config $deploy --models $models,$deployModels"
    foreach ($line in $points) {
        $p = $line -split "\s+"
        if ($p.Count -ne 2) { throw "unexpected line from cvt deploy list: '$line'" }
        Run "uv run --extra ml cvt eval --model '$($p[0])' --sets $sets --imgsz $($p[1]) --models $models,$deployModels"
    }
}
if ($Step -eq "fallback" -or $Step -eq "all") {
    Assert-Trained "al-m-variety-800"
    # exports the 640 / 1536 engines (the 1280 headline engine is skipped when up to date)
    Run "uv run --extra ml cvt deploy export --config $fallback --models $models"
    Run "uv run --extra ml cvt deploy register --config $fallback --models $models --out $fallbackModels"
    $fb = Get-Lines "cvt al fallbacks"
    $only = ($fb | ForEach-Object { ($_ -split "\s+")[0] }) -join ","
    Run "uv run --extra ml cvt deploy bench --config $fallback --models $models,$fallbackModels --only '$only'"
    foreach ($line in $fb) {
        $p = $line -split "\s+"
        if ($p.Count -ne 2) { throw "unexpected line from cvt al fallbacks: '$line'" }
        Run "uv run --extra ml cvt eval --model '$($p[0])' --sets $sets --imgsz $($p[1]) --models $models,$fallbackModels"
    }
}
if ($Step -eq "report" -or $Step -eq "all") {
    Run "uv run --extra ml cvt al report"
    Run "uv run --extra ml cvt report"
}
