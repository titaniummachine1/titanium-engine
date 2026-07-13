# Full movegen / perft / lazy-seal measurement battery (post wall-pipeline optimizations).
param(
    [string]$OutDir = "runs/movegen_measure_$(Get-Date -Format 'yyyyMMdd_HHmmss')",
    [switch]$SkipLazySeal,
    [switch]$SkipPerft5NoTt
)

$ErrorActionPreference = "Continue"
$EngineRoot = Split-Path $PSScriptRoot -Parent
Set-Location $EngineRoot

$env:RUSTFLAGS = "-C target-cpu=native"
$RunRoot = Join-Path $EngineRoot $OutDir
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

$Summary = [ordered]@{
    timestamp = (Get-Date -Format "o")
    rustflags = $env:RUSTFLAGS
    out_dir   = $OutDir
    steps     = @()
}

function Add-Step([string]$Name, [string]$Status, [hashtable]$Data = @{}) {
    $row = [ordered]@{ name = $Name; status = $Status }
    foreach ($k in $Data.Keys) { $row[$k] = $Data[$k] }
    $Summary.steps += $row
    $color = if ($Status -eq "PASS") { "Green" } elseif ($Status -eq "FAIL") { "Red" } else { "Yellow" }
    Write-Host "[$Status] $Name" -ForegroundColor $color
}

function Run-Capture([string]$Name, [string]$LogName, [scriptblock]$Block) {
    $logPath = Join-Path $RunRoot $LogName
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $exit = 0
    try {
        $output = & $Block 2>&1
        $output | ForEach-Object { "$_" } | Set-Content -Encoding utf8 $logPath
        $output | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { $exit = $LASTEXITCODE }
    } catch {
        "$_" | Add-Content $logPath
        Write-Host "ERROR: $_" -ForegroundColor Red
        $exit = 1
    }
    $sw.Stop()
    $status = if ($exit -eq 0) { "PASS" } else { "FAIL" }
    Add-Step $Name $status @{ log = $LogName; wall_s = [math]::Round($sw.Elapsed.TotalSeconds, 2); exit = $exit }
    return $exit
}

Write-Host "Measurement output: $RunRoot`n"

$failures = 0

$failures += Run-Capture "build_titanium" "01_build_titanium.log" {
    cargo build --release -p titanium
}

$failures += Run-Capture "build_search_bench" "02_build_search_bench.log" {
    cargo build --release -p titanium --bin search_bench
}

$Titanium = Join-Path $EngineRoot "target\release\titanium.exe"

$failures += Run-Capture "test_movegen" "03_test_movegen.log" {
    cargo test -p titanium --release movegen::
}

$benches = @(
    @{ name = "perft_pawn_modes";   log = "10_perft_pawn_modes.log" },
    @{ name = "perft_full_compare"; log = "11_perft_full_compare.log" },
    @{ name = "tt_speedup";         log = "12_tt_speedup.log" },
    @{ name = "flood_modes";        log = "13_flood_modes.log" },
    @{ name = "perft_pawn_only";    log = "14_perft_pawn_only.log" },
    @{ name = "path_bfs";           log = "15_path_bfs.log"; extra = @("--", "--noplot") },
    @{ name = "cat_build";          log = "16_cat_build.log"; extra = @("--", "--noplot") }
)

foreach ($b in $benches) {
    $extra = if ($b.extra) { $b.extra } else { @() }
    $failures += Run-Capture "bench_$($b.name)" $b.log {
        cargo bench --bench $($b.name) @extra
    }
}

# ── perft-bench: production engine (TT on, topo flood-skip) ─────────────────
$failures += Run-Capture "perft5_tt_topo" "20_perft5_tt_topo.log" {
    Remove-Item Env:TITANIUM_BENCH -ErrorAction SilentlyContinue
    Remove-Item Env:TITANIUM_WALL_FLOOD_SKIP -ErrorAction SilentlyContinue
    & $Titanium perft-bench 5
}

if (-not $SkipPerft5NoTt) {
    $failures += Run-Capture "perft5_no_tt_topo" "21_perft5_no_tt_topo.log" {
        Remove-Item Env:TITANIUM_BENCH -ErrorAction SilentlyContinue
        Remove-Item Env:TITANIUM_WALL_FLOOD_SKIP -ErrorAction SilentlyContinue
        & $Titanium perft-bench --no-tt 5
    }
}

# ── flood-skip A/B at perft(4) no-TT (o1-lut, bench flag path) ───────────────
$failures += Run-Capture "perft4_no_tt_topo" "22_perft4_no_tt_topo.log" {
    Remove-Item Env:TITANIUM_BENCH -ErrorAction SilentlyContinue
    Remove-Item Env:TITANIUM_WALL_FLOOD_SKIP -ErrorAction SilentlyContinue
    & $Titanium perft-bench --no-tt 4
}

$failures += Run-Capture "perft4_no_tt_anchor" "23_perft4_no_tt_anchor.log" {
    $env:TITANIUM_BENCH = "1"
    $env:TITANIUM_WALL_FLOOD_SKIP = "anchor"
    & $Titanium perft-bench --no-tt 4
}
Remove-Item Env:TITANIUM_BENCH -ErrorAction SilentlyContinue
Remove-Item Env:TITANIUM_WALL_FLOOD_SKIP -ErrorAction SilentlyContinue

# ── A/B/C/D lazy-seal parity battery ───────────────────────────────────────
if (-not $SkipLazySeal) {
    $lazyOut = Join-Path $RunRoot "lazy_seal_abcd"
    New-Item -ItemType Directory -Force -Path $lazyOut | Out-Null
    $failures += Run-Capture "lazy_seal_abcd" "30_lazy_seal_abcd.log" {
        Push-Location $EngineRoot
        try {
            & "$EngineRoot\scripts\lazy_seal_abcd_battery.ps1" -OutDir $lazyOut
        } finally {
            Pop-Location
        }
    }
    if (Test-Path (Join-Path $lazyOut "summary.csv")) {
        Copy-Item (Join-Path $lazyOut "summary.csv") (Join-Path $RunRoot "lazy_seal_summary.csv")
    }
}

# ── Aggregate key metrics ────────────────────────────────────────────────────
$metrics = [ordered]@{}

function Parse-PerftBenchLine([string]$Line) {
    if ($Line -match "perft_bench depth=(\d+) nodes=(\d+) threads=(\d+) wall_flood_skip=(\S+) time_s=([\d.]+) nps=([\d.]+)") {
        return [ordered]@{
            depth = [int]$Matches[1]; nodes = [int64]$Matches[2]; threads = [int]$Matches[3]
            flood_skip = $Matches[4]; time_s = [double]$Matches[5]; nps = [double]$Matches[6]
        }
    }
    return $null
}

foreach ($log in @("20_perft5_tt_topo.log", "21_perft5_no_tt_topo.log", "22_perft4_no_tt_topo.log", "23_perft4_no_tt_anchor.log")) {
    $p = Join-Path $RunRoot $log
    if (Test-Path $p) {
        $line = Get-Content $p | Where-Object { $_ -match "perft_bench" } | Select-Object -Last 1
        if ($line) {
            $key = ($log -replace '\.log$','')
            $metrics[$key] = Parse-PerftBenchLine $line
        }
    }
}

$fullCompare = Join-Path $RunRoot "11_perft_full_compare.log"
if (Test-Path $fullCompare) {
    $tt5 = Get-Content $fullCompare | Where-Object { $_ -match '^\| d5 \|' } | Select-Object -First 1
    if ($tt5 -match '\| d5 \| (\d+) \| (\S+) \| ([\d.]+)') {
        $metrics["perft5_tt_o1_regression_gate"] = [ordered]@{
            nodes = [int64]$Matches[1]; correct = $Matches[2]; time_s = [double]$Matches[3]
        }
    }
}

$Summary.metrics = $metrics
$Summary.failures = $failures
$summaryPath = Join-Path $RunRoot "summary.json"
$Summary | ConvertTo-Json -Depth 6 | Set-Content -Encoding utf8 $summaryPath

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "MEASUREMENT COMPLETE" -ForegroundColor Cyan
Write-Host "Output: $RunRoot" -ForegroundColor Cyan
Write-Host "Failures: $failures" -ForegroundColor $(if ($failures -eq 0) { "Green" } else { "Red" })
Write-Host "Summary: $summaryPath" -ForegroundColor Cyan

if ($metrics.Count -gt 0) {
    Write-Host "`nKey perft metrics:" -ForegroundColor Yellow
    $metrics.GetEnumerator() | ForEach-Object {
        Write-Host "  $($_.Key): $($_.Value | ConvertTo-Json -Compress)"
    }
}

exit [int]($failures -gt 0)
