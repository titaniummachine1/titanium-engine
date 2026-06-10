# DEPRECATED — do not use unattended loop.
# Agent runs manual chunks: analyze -> tweak -> next chunk.
# See benchmark/overnight/MANUAL.md
Write-Host "overnight_loop.ps1 is disabled. Use manual chunks:" -ForegroundColor Yellow
Write-Host "  node benchmark/overnight_iterate.mjs --resume --steps 1 --workers 6 --probe-games 12 --no-confirm"
exit 1
