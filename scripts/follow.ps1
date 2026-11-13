# Target lock and virtual-gimbal follow on our own GT lock tests: detector features of the
# cached m@1536 detections (GPU), the pre-declared lock and gimbal grids on MOT-tune (CPU), the
# frozen evaluation of the 5 lock variants and the gimbal on T-MOT and T-UAVDT (CPU), the same on
# the public benchmarks UAVDT-S, UAV123-veh and UAV20L-veh (detection GPU, evaluation CPU), then
# the results page (CPU).
#   powershell -ExecutionPolicy Bypass -File scripts\follow.ps1 -Step features  # GPU ~30-60 min
#   powershell -ExecutionPolicy Bypass -File scripts\follow.ps1 -Step sot-detect  # GPU
#   powershell -ExecutionPolicy Bypass -File scripts\follow.ps1 -Step tune      # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\follow.ps1 -Step eval      # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\follow.ps1 -Step sot-eval  # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\follow.ps1 -Step report    # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\follow.ps1                 # all, in order
# features needs the m@1536 detection caches of scripts\track3b.ps1 -Step detect; it is resumable
# (sets whose feature file is up to date are skipped). tune tracks MOT-tune with botsort-tuned3b
# first if those tracks are missing, writes configs/follow.tuned.yaml and
# reports/follow_grid_{lock,gimbal}.csv; eval uses the frozen configs/follow.tuned.yaml.
# sot-detect runs the m@1536 engine, then the .pt feature pass, on every frame of the benchmark
# videos declared in configs/follow_sot.yaml (data/external/uavdt_sot, data/external/uav123) and
# caches them in data/cache/sot (resumable per video). sot-eval (frozen params) tracks with GMC,
# checks UAVDT-S against the training frames (pHash; matched sequences are dropped, pairs in
# reports/follow_sot_leakage.csv) and appends rows UAVDT-S, UAV123-veh, UAV20L-veh.
# demo (needs configs/follow.tuned.yaml and a GPU): click a vehicle at second -Frame of -Video, lock
# it (reacq variant) and follow it with the virtual gimbal -> -Out mp4 (+ command log .csv beside it).
# Your own clips are never committed. Example: follow.ps1 -Step demo -Video clip.mp4 -Click 1920,1080
#   -Frame 5 -Seconds 20 -Out reports/demo/follow.mp4 (-Click auto, or car/motorbike/truck/bus, picks it)
# demo-all (GPU, ~3-5 min per clip): every clip in -Videos (folders or files, comma-separated), a
# -Seconds segment (default 60 here) from its middle, one follow video per vehicle class (a different vehicle each,
# picked automatically) -> data/demo_out/follow/<clip>_<class>.mp4 + summary.csv. Resumable.
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log. -DryRun prints every command without running it.
param([ValidateSet("features", "sot-detect", "tune", "eval", "sot-eval", "report", "demo", "demo-all", "all")]
      [string]$Step = "all",
      [string]$Video = "",
      [string]$Click = "",
      [double]$Frame = 0,
      [double]$Seconds = 20,
      [string]$Out = "reports/demo/follow.mp4",
      [string]$Videos = "data/external/uitadrone/videos",
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
$det = "src_pseudo-yolo11m-s0@1536-fp16"  # configs/follow.yaml `detector` (TensorRT engine)
$weights = "src_pseudo-yolo11m-s0"         # its .pt weights: the P3 feature map
$models = "configs/models.yaml,configs/models_deploy.yaml"
$config = "configs/follow.yaml"
$tuned = "configs/follow.tuned.yaml"
$sot = "configs/follow_sot.yaml"
$sotSets = "UAVDT-S,UAV123-veh,UAV20L-veh"

if ($Step -eq "features" -or $Step -eq "all") {
    Run "uv run --extra ml cvt follow features --weights-model $weights --boxes-of '$det' --imgsz 1536 --sets MOT-tune,T-MOT,T-UAVDT --min-score 0.1 --models $models"
}
if ($Step -eq "sot-detect" -or $Step -eq "all") {
    # engine detections + P3 features on every frame of the declared benchmark videos
    Run "uv run --extra ml cvt follow sot-detect --config $config --sot $sot --models $models --weights-model $weights --imgsz 1536 --min-score 0.1 --sets $sotSets"
}
if ($Step -eq "tune" -or $Step -eq "all") {
    # MOT-tune only: 18 lock points (lock-features success AUC), 12 gimbal points (GT-driven)
    Run "uv run --extra ml cvt follow tune --config $config --models $models --out $tuned"
}
if ($Step -eq "eval" -or $Step -eq "all") {
    if (-not (Test-Path $tuned) -and -not $DryRun) { throw "missing ${tuned}: run this script with -Step tune first" }
    Run "uv run --extra ml cvt follow eval --config $config --models $models --tuned $tuned --sets T-MOT,T-UAVDT"
}
if ($Step -eq "sot-eval" -or $Step -eq "all") {
    if (-not (Test-Path $tuned) -and -not $DryRun) { throw "missing ${tuned}: run this script with -Step tune first" }
    Run "uv run --extra ml cvt follow sot-eval --config $config --sot $sot --models $models --tuned $tuned --sets $sotSets"
}
if ($Step -eq "report" -or $Step -eq "all") {
    Run "uv run --extra ml cvt report"
}
if ($Step -eq "demo") {
    if (-not $Video -or -not $Click) { throw "-Step demo needs -Video <clip> and -Click x,y" }
    if (-not (Test-Path $tuned) -and -not $DryRun) { throw "missing ${tuned}: run this script with -Step tune first" }
    Run "uv run --extra ml cvt follow demo --video '$Video' --click '$Click' --frame $Frame --seconds $Seconds --out '$Out' --config $config --tuned $tuned --models $models"
}
if ($Step -eq "demo-all") {
    if (-not (Test-Path $tuned) -and -not $DryRun) { throw "missing ${tuned}: run this script with -Step tune first" }
    Run "uv run --extra ml cvt follow demo-all --videos '$Videos' --seconds $(if ($PSBoundParameters.ContainsKey("Seconds")) { $Seconds } else { 60 }) --out-dir data/demo_out/follow --config $config --tuned $tuned --models $models"
}
