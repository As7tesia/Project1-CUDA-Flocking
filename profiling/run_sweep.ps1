<#
.SYNOPSIS
  Run the profiled boids exe over a list of boid counts and collect one CSV per run.

.DESCRIPTION
  The exe must already be built (Release) with PROFILE 1 in src/main.cpp and with the
  VISUALIZE / UNIFORM_GRID / COHERENT_GRID / blockSize / cell-width settings you want
  to measure. This script does not rebuild; it just runs the exe once per boid count
  with cwd = build/ (shaders are loaded by relative path), waits for it to exit on
  its own, and moves <label>.csv into profiling/raw/ (or -RawDir).

  Label convention (the analysis script parses it):
      <mode>_vis<0|1>_bs<blockSize>_cw<cellWidthMultiplier>_n<N>
  e.g. coherent_vis0_bs128_cw2_n50000

.EXAMPLE
  .\run_sweep.ps1 -Tag coherent_vis0_bs128_cw2 -Sizes 5000,10000,20000,50000,100000,200000,500000,1000000
  .\run_sweep.ps1 -Tag naive_vis0_bs128_cw2 -Sizes 5000,10000,20000,50000,100000 -StepsFor @{100000=500}
#>
param(
    [Parameter(Mandatory)][string]$Tag,
    [Parameter(Mandatory)][int[]]$Sizes,
    [int]$Steps = 3000,
    [hashtable]$StepsFor = @{},
    [int]$Warmup = 200,
    [int]$TimeoutSec = 900,
    [switch]$Force,
    [string]$RawDir = ''   # where the CSVs go; default profiling/raw
)

$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$exe  = Join-Path $root 'build\bin\Release\cis5650_boids.exe'
$cwd  = Join-Path $root 'build'
$raw  = if ($RawDir) { $RawDir } else { Join-Path $PSScriptRoot 'raw' }
New-Item -ItemType Directory -Force $raw | Out-Null

if (-not (Test-Path $exe)) { throw "exe not found: $exe (build Release first)" }

$util = (& nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits).Trim()
if ([int]$util -gt 10) {
    Write-Warning "GPU is already at $util% utilization before the sweep. Minimize other GPU windows for clean numbers."
} else {
    Write-Host "GPU idle check: $util% utilization"
}

Write-Host ("{0,-36} {1,6} {2,8} {3,10} {4,10} {5,10}" -f 'label','steps','wall s','step ms','frame ms','fps')
foreach ($n in $Sizes) {
    $label = "${Tag}_n$n"
    $dest  = Join-Path $raw "$label.csv"
    if ((Test-Path $dest) -and -not $Force) { Write-Host ("{0,-36} skip, already have it" -f $label); continue }

    $s = if ($StepsFor.ContainsKey($n)) { [int]$StepsFor[$n] } else { $Steps }
    $log = Join-Path $cwd "$label.log"
    $t0 = Get-Date
    $p = Start-Process -FilePath $exe -ArgumentList @($label, $n, $s) -WorkingDirectory $cwd -PassThru `
         -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    if (-not $p.WaitForExit($TimeoutSec * 1000)) {
        Stop-Process -Id $p.Id -Force
        Write-Warning "$label timed out after $TimeoutSec s, killed"
        continue
    }
    $wall = ((Get-Date) - $t0).TotalSeconds
    $src = Join-Path $cwd "$label.csv"
    if ($p.ExitCode -ne 0 -or -not (Test-Path $src)) {
        Write-Warning "$label failed (exit $($p.ExitCode)); see $log"
        continue
    }
    Move-Item $src $dest -Force
    Remove-Item $log, "$log.err" -ErrorAction SilentlyContinue

    # quick feedback: mean step / frame time over the rows after warm-up
    $rows = Get-Content $dest | Where-Object { $_ -notmatch '^(#|step,)' } | Select-Object -Skip $Warmup
    if ($rows.Count -gt 0) {
        $stepMs  = ($rows | ForEach-Object { [double]($_ -split ',')[1] } | Measure-Object -Average).Average
        $frameMs = ($rows | ForEach-Object { [double]($_ -split ',')[2] } | Measure-Object -Average).Average
        Write-Host ("{0,-36} {1,6} {2,8:n1} {3,10:n4} {4,10:n4} {5,10:n1}" -f $label, $s, $wall, $stepMs, $frameMs, (1000.0 / $frameMs))
    } else {
        Write-Host ("{0,-36} {1,6} {2,8:n1}  (fewer rows than warm-up)" -f $label, $s, $wall)
    }
}
