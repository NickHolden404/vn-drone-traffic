# Tracking re-tune: cache the m@1536 engine's detections (GPU), tune ByteTrack / BoT-SORT on
# MOT-tune with grid 3b for both detectors (CPU), then track, evaluate (counting rules 1 and 2) and
# compare the 2 x 2 ablation on T-MOT and T-UAVDT (CPU) and regenerate the results tables.
#   powershell -ExecutionPolicy Bypass -File scripts\track3b.ps1 -Step detect   # GPU ~30-60 min
#   powershell -ExecutionPolicy Bypass -File scripts\track3b.ps1 -Step tune     # CPU, hours
#   powershell -ExecutionPolicy Bypass -File scripts\track3b.ps1 -Step eval     # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\track3b.ps1 -Step report   # CPU
#   powershell -ExecutionPolicy Bypass -File scripts\track3b.ps1                # all, in order
# Rows (detector x tracker x config): detectors s@1280 (original) and m@1536 (TensorRT engine);
# configs = originally tuned (configs/trackers/<t>.tuned.yaml) and grid 3b tuned on that detector
# (configs/trackers/<t>.tuned3b.<detector>.yaml). The 2 original rows (s@1280 + originally tuned) keep
# their tracks; eval re-evaluates them so they gain the rule-2 count metrics.
# Don't pipe this script's output (2>&1 | Tee-Object aborts under PowerShell 5.1); use
# Start-Transcript if you want a log. -DryRun prints every command without running it.
param([ValidateSet("detect", "tune", "eval", "report", "all")][string]$Step = "all",
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
$d1 = "src_pseudo-yolo11s-s0"            # original detector (imgsz 1280)
$d2 = "src_pseudo-yolo11m-s0@1536-fp16"  # TensorRT engine (registry imgsz 1536)
# detector -> registry yamls
$detectors = @(
    @($d1, "configs/models.yaml"),
    @($d2, "configs/models.yaml,configs/models_deploy.yaml")
)
$trackers = "bytetrack", "botsort"
$sets = "T-MOT,T-UAVDT"

if ($Step -eq "detect" -or $Step -eq "all") {
    # caches the detections the trackers read, and reports detection mAP on these sets
    Run "uv run --extra ml cvt eval --model '$d2' --sets MOT-tune,T-MOT,T-UAVDT --imgsz 1536 --models configs/models.yaml,configs/models_deploy.yaml"
}
if ($Step -eq "tune" -or $Step -eq "all") {
    # grid 3b (192 points) on MOT-tune only; writes reports/tracker_grid_<t>_3b_<detector>.csv
    foreach ($d in $detectors) {
        foreach ($t in $trackers) {
            Run "uv run --extra ml cvt track tune --model '$($d[0])' --tracker-type $t --grid 3b --models $($d[1]) --out 'configs/trackers/$t.tuned3b.$($d[0]).yaml'"
        }
    }
}
if ($Step -eq "eval" -or $Step -eq "all") {
    foreach ($d in $detectors) {
        foreach ($t in $trackers) {
            foreach ($cfg in @("configs/trackers/$t.tuned.yaml", "configs/trackers/$t.tuned3b.$($d[0]).yaml")) {
                if (-not (Test-Path $cfg) -and -not $DryRun) { throw "missing ${cfg}: run this script with -Step tune first" }
                $common = "--model '$($d[0])' --tracker '$cfg' --sets $sets --models $($d[1])"
                $plan3Row = ($d[0] -eq $d1) -and ($cfg -eq "configs/trackers/$t.tuned.yaml")
                # original rows: their tracks exist (cvt track eval refuses stale ones); only re-evaluate
                if (-not $plan3Row) { Run "uv run --extra ml cvt track run $common" }
                Run "uv run --extra ml cvt track eval $common --scenes configs/scenes"
                # per-cell GT vs predicted counts -> reports/counts_<model>+<tracker>.csv (rule 1), counts2_ (rule 2)
                Run "uv run --extra ml cvt track counts $common --scenes configs/scenes"
            }
        }
    }
    # m@1536 + grid 3b vs original (s@1280 + originally tuned): different detectors, same GT sequences
    foreach ($t in $trackers) {
        Run "uv run --extra ml cvt track compare --a '$d2+$t-tuned3b' --b '$d1+$t-tuned' --sets $sets --cross-detector"
    }
    Run "uv run --extra ml cvt track compare --a '$d2+botsort-tuned3b' --b '$d2+bytetrack-tuned3b' --sets $sets"
}
if ($Step -eq "report" -or $Step -eq "all") {
    Run "uv run --extra ml cvt report"
}
