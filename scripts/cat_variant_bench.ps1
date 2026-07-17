# CAT path-LMR variant validation: parity, instrumented metrics, paired NPS.
param(
    [int]$PairedRuns = 15,
    [switch]$SkipNps,
    [switch]$SkipParity,
    [switch]$SkipInstr,
    [switch]$InstrOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location (Split-Path $PSScriptRoot -Parent)
$env:TITANIUM_BENCH_ENGINE = "titanium-v17"
$ReleaseBin = ".\target\release\search_bench.exe"
$InstrBin = $ReleaseBin

function Clear-VariantEnv {
    Remove-Item Env:TITANIUM_CAT_CHILD_DIST_REUSE -ErrorAction SilentlyContinue
    Remove-Item Env:TITANIUM_CAT_NO_EDGE_SKIP -ErrorAction SilentlyContinue
    Remove-Item Env:TITANIUM_REFRESH_AB_SKIP -ErrorAction SilentlyContinue
}

function Set-Variant($Name) {
    Clear-VariantEnv
    switch ($Name) {
        "A" { $env:TITANIUM_CAT_NO_EDGE_SKIP = "0" }
        "B" { $env:TITANIUM_CAT_CHILD_DIST_REUSE = "1" }
        "C" { }
        "D" {
            $env:TITANIUM_CAT_CHILD_DIST_REUSE = "1"
        }
        default { throw "unknown variant $Name" }
    }
}

function Invoke-DepthBench($Pos, $Depth) {
    $out = cmd /c "`"$ReleaseBin`" depth --position $Pos --threads 1 --full --depth $Depth 2>&1"
    $line = ($out -split "`n" | Where-Object { $_ -match '^\{"bench_type"' }) | Select-Object -Last 1
    if (-not $line) { throw "no JSON from depth bench pos=$Pos depth=$Depth" }
    $line | ConvertFrom-Json
}

function Invoke-ProfileBench($Sec) {
    $out = cmd /c "`"$ReleaseBin`" profile --sec $Sec --position startpos --threads 1 --full 2>&1"
    $line = ($out -split "`n" | Where-Object { $_ -match '^\{"bench_type"' }) | Select-Object -Last 1
    if (-not $line) { throw "no JSON from profile bench" }
    $line | ConvertFrom-Json
}

function Invoke-InstrBench($Sec) {
    $out = cmd /c "`"$InstrBin`" instr --sec $Sec --position startpos --threads 1 --full 2>&1"
    $profile = ($out -split "`n" | Where-Object { $_ -match '^\{"bench_type":"profile"' }) | Select-Object -Last 1
    $instr = ($out -split "`n" | Where-Object { $_ -match '"refresh_dist_calls"' }) | Select-Object -Last 1
    if (-not $instr) { throw "no instr JSON" }
    [PSCustomObject]@{
        profile = if ($profile) { $profile | ConvertFrom-Json } else { $null }
        instr   = $instr | ConvertFrom-Json
    }
}

Write-Output "=== PARITY (fast suite) ==="
$parityFailures = @()
if (-not $SkipParity) {
    $parityCases = @(
        @{ pos = "startpos"; depth = 12 },
        @{ pos = "startpos"; depth = 14 },
        @{ pos = "startpos"; depth = 16 },
        @{ pos = "c3h-midgame"; depth = 12 },
        @{ pos = "low-wall"; depth = 12 },
        @{ pos = "wall-maze"; depth = 10 },
        @{ pos = "wall-maze"; depth = 12 },
        @{ pos = "dense-maze"; depth = 10 },
        @{ pos = "endgame-c5"; depth = 10 },
        @{ pos = "endgame-c5"; depth = 12 }
    )
    $baseline = @{}
    foreach ($case in $parityCases) {
        $key = "$($case.pos)|$($case.depth)"
        foreach ($v in @("A", "B", "C", "D")) {
            Set-Variant $v
            $r = Invoke-DepthBench $case.pos $case.depth
            Write-Output ("parity {0} {1} d={2} nodes={3} move={4} score={5}" -f $v, $case.pos, $case.depth, $r.nodes, $r.move, $r.score)
            if ($v -eq "A") {
                $baseline[$key] = $r
            }
            elseif ($r.nodes -ne $baseline[$key].nodes -or $r.move -ne $baseline[$key].move -or $r.score -ne $baseline[$key].score) {
                $parityFailures += "$v mismatch $key"
            }
        }
    }
    if ($parityFailures.Count -eq 0) {
        Write-Output "PARITY_OK $($parityCases.Count) cases x B/C/D vs A"
    }
    else {
        $parityFailures | ForEach-Object { Write-Output $_ }
    }
}
else {
    Write-Output "PARITY_SKIPPED"
}

Write-Output "=== INSTRUMENTED 10s startpos (A/B/C/D) ==="
$instrRows = @()
if (-not $SkipInstr) {
    foreach ($v in @("A", "B", "C", "D")) {
        Set-Variant $v
        $r = Invoke-InstrBench 10
        $cat = $r.instr.cat_path_lmr
        $catSite = ($r.instr.refresh_sites | Where-Object { $_.label -eq "cat_path_lmr" } | Select-Object -First 1)
        $row = [PSCustomObject]@{
            variant            = $v
            nodes              = $r.instr.search_nodes
            refresh_dist_calls = $r.instr.refresh_dist_calls
            cat_path_lmr_calls = if ($catSite) { $catSite.calls } else { 0 }
            cat_refloods       = if ($catSite) { $catSite.reflood } else { 0 }
            no_edge_skip       = $cat.no_edge_skip
            edge_test_calls    = $cat.edge_test_calls
            dup_avoided        = $cat.dup_avoided
        }
        $instrRows += $row
        Write-Output ("instr {0} nodes={1} refresh={2} cat_calls={3} cat_reflood={4} no_edge_skip={5} dup_avoided={6}" -f $row.variant, $row.nodes, $row.refresh_dist_calls, $row.cat_path_lmr_calls, $row.cat_refloods, $row.no_edge_skip, $row.dup_avoided)
    }
}
else {
    Write-Output "INSTR_SKIPPED"
}

if (-not $SkipNps -and -not $InstrOnly) {
    Write-Output "=== PAIRED 10s NPS ($PairedRuns pairs: A vs B, A vs C, A vs D) ==="
    function Run-Paired($Label, $VariantB) {
        $aNps = New-Object System.Collections.Generic.List[double]
        $bNps = New-Object System.Collections.Generic.List[double]
        $aNodes = New-Object System.Collections.Generic.List[uint64]
        $bNodes = New-Object System.Collections.Generic.List[uint64]
        for ($pair = 1; $pair -le $PairedRuns; $pair++) {
            $order = if ($pair % 2 -eq 1) { @("A", $VariantB) } else { @($VariantB, "A") }
            foreach ($v in $order) {
                Set-Variant $v
                $r = Invoke-ProfileBench 10
                if ($v -eq "A") {
                    $aNps.Add([double]$r.nps)
                    $aNodes.Add([uint64]$r.nodes)
                }
                else {
                    $bNps.Add([double]$r.nps)
                    $bNodes.Add([uint64]$r.nodes)
                }
                Write-Output ("nps {0} pair={1} variant={2} nodes={3} nps={4} depth={5} move={6}" -f $Label, $pair, $v, $r.nodes, [int]$r.nps, $r.depth, $r.move)
            }
        }
        $pairedPct = for ($i = 0; $i -lt $PairedRuns; $i++) { 100.0 * ($bNps[$i] / $aNps[$i] - 1.0) }
        $sortedA = $aNps | Sort-Object
        $sortedB = $bNps | Sort-Object
        $medA = $sortedA[[int]($PairedRuns / 2)]
        $medB = $sortedB[[int]($PairedRuns / 2)]
        $meanPct = ($pairedPct | Measure-Object -Average).Average
        $stdevPct = [Math]::Sqrt((($pairedPct | ForEach-Object { ($_ - $meanPct) * ($_ - $meanPct) } | Measure-Object -Sum).Sum / ($PairedRuns - 1)))
        $ci = 1.96 * $stdevPct / [Math]::Sqrt($PairedRuns)
        Write-Output ("summary {0} median_A_nps={1} median_B_nps={2} median_gain_pct={3:N2} mean_gain_pct={4:N2} ci95=[{5:N2},{6:N2}]" -f $Label, [int]$medA, [int]$medB, (100.0 * ($medB / $medA - 1)), $meanPct, ($meanPct - $ci), ($meanPct + $ci))
    }
    Run-Paired "A_vs_B" "B"
    Run-Paired "A_vs_C" "C"
    Run-Paired "A_vs_D" "D"
}

$outPath = Join-Path $PSScriptRoot "cat_variant_bench_results.txt"
@(
    "parity_failures=$($parityFailures.Count)",
    ($instrRows | Format-Table -AutoSize | Out-String)
) | Set-Content $outPath
Write-Output "Wrote $outPath"
