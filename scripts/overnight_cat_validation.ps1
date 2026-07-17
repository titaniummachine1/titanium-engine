# Overnight CAT validation: oracle, parity, extended NPS.
$ErrorActionPreference = "Continue"
$LogDir = Join-Path (Split-Path $PSScriptRoot -Parent) "scripts\overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Log = Join-Path $LogDir "overnight_$Stamp.log"

function Log($msg) {
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $Log -Value $line
    Write-Output $line
}

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:TITANIUM_BENCH_ENGINE = "titanium-v17"
$env:RUSTFLAGS = "-C target-cpu=native"
$Bin = ".\target\release\search_bench.exe"

Log "=== BUILD ==="
cargo build --release --bin search_bench 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "build_$Stamp.log") | Out-Null
if ($LASTEXITCODE -ne 0) { Log "BUILD FAILED"; exit 1 }

Log "=== ORACLE TEST ==="
cargo test -p titanium --lib wall_incr_no_edge_cut_implies_unchanged_pawn_distance_oracle 2>&1 | Tee-Object -FilePath (Join-Path $LogDir "oracle_$Stamp.log")
if ($LASTEXITCODE -ne 0) { Log "ORACLE FAILED"; exit 1 }
Log "ORACLE OK"

function Bench-Profile($sec, $pos, $extraEnv) {
    Remove-Item Env:TITANIUM_CAT_CHILD_DIST_REUSE,Env:TITANIUM_REFRESH_AB_SKIP -ErrorAction SilentlyContinue
    Remove-Item Env:TITANIUM_CAT_NO_EDGE_SKIP -ErrorAction SilentlyContinue
    foreach ($k in $extraEnv.Keys) { Set-Item -Path "Env:$k" -Value $extraEnv[$k] }
    $out = cmd /c "`"$Bin`" profile --sec $sec --position $pos --threads 1 --full 2>&1"
    ($out -split "`n" | Where-Object { $_ -match '^\{"bench_type"' }) | Select-Object -Last 1 | ConvertFrom-Json
}

function Bench-Depth($pos, $depth, $extraEnv) {
    Remove-Item Env:TITANIUM_CAT_CHILD_DIST_REUSE,Env:TITANIUM_REFRESH_AB_SKIP -ErrorAction SilentlyContinue
    Remove-Item Env:TITANIUM_CAT_NO_EDGE_SKIP -ErrorAction SilentlyContinue
    foreach ($k in $extraEnv.Keys) { Set-Item -Path "Env:$k" -Value $extraEnv[$k] }
    $out = cmd /c "`"$Bin`" depth --position $pos --threads 1 --full --depth $depth 2>&1"
    ($out -split "`n" | Where-Object { $_ -match '^\{"bench_type"' }) | Select-Object -Last 1 | ConvertFrom-Json
}

Log "=== PARITY SPOT (A off vs default C on) ==="
$parity = @(
    @{pos="startpos";d=14}, @{pos="wall-maze";d=12}, @{pos="endgame-c5";d=12}
)
foreach ($c in $parity) {
    $a = Bench-Depth $c.pos $c.d @{ "TITANIUM_CAT_NO_EDGE_SKIP" = "0" }
    $b = Bench-Depth $c.pos $c.d @{}
    $ok = ($a.nodes -eq $b.nodes -and $a.move -eq $b.move -and $a.score -eq $b.score)
    Log ("parity {0} d={1} ok={2} A=({3},{4},{5}) B=({6},{7},{8})" -f $c.pos,$c.d,$ok,$a.nodes,$a.move,$a.score,$b.nodes,$b.move,$b.score)
}

Log "=== 20 PAIRED 30s startpos A(off) vs C(on) ==="
$aNps = @(); $cNps = @()
for ($p = 1; $p -le 20; $p++) {
    $order = if ($p % 2 -eq 1) { @("A","C") } else { @("C","A") }
    foreach ($v in $order) {
        $env = if ($v -eq "A") { @{ "TITANIUM_CAT_NO_EDGE_SKIP" = "0" } } else { @{} }
        $r = Bench-Profile 30 "startpos" $env
        if ($v -eq "A") { $aNps += [double]$r.nps } else { $cNps += [double]$r.nps }
        Log ("30s pair={0} {1} nodes={2} nps={3} depth={4}" -f $p,$v,$r.nodes,[int]$r.nps,$r.depth)
    }
}
$sA = $aNps | Sort-Object; $sC = $cNps | Sort-Object
$medA = $sA[9]; $medC = $sC[9]
$pct = 100.0 * ($cNps | ForEach-Object { $_ } | ForEach-Object -Begin {$i=0} -Process { 100*($cNps[$i]/$aNps[$i]-1); $i++ })
$meanPct = ($pct | Measure-Object -Average).Average
Log ("30s summary median_A={0} median_C={1} median_gain={2:N2}% mean_gain={3:N2}%" -f [int]$medA,[int]$medC,(100*($medC/$medA-1)),$meanPct)

Log "=== 10 PAIRED 30s wall-maze ==="
$aNps2 = @(); $cNps2 = @()
for ($p = 1; $p -le 10; $p++) {
    $order = if ($p % 2 -eq 1) { @("A","C") } else { @("C","A") }
    foreach ($v in $order) {
        $env = if ($v -eq "A") { @{ "TITANIUM_CAT_NO_EDGE_SKIP" = "0" } } else { @{} }
        $r = Bench-Profile 30 "wall-maze" $env
        if ($v -eq "A") { $aNps2 += [double]$r.nps } else { $cNps2 += [double]$r.nps }
        Log ("wall-maze pair={0} {1} nodes={2} nps={3}" -f $p,$v,$r.nodes,[int]$r.nps)
    }
}
$sA2 = $aNps2 | Sort-Object; $sC2 = $cNps2 | Sort-Object
Log ("wall-maze summary median_A={0} median_C={1} gain={2:N2}%" -f [int]$sA2[4],[int]$sC2[4],(100*($sC2[4]/$sA2[4]-1)))

Log "=== DONE log=$Log ==="
