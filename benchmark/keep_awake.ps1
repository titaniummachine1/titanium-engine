# Prevent sleep while overnight tournament runs (AC power).
Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SleepBlock {
    [DllImport("kernel32.dll", CharSet=CharSet.Auto, SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$ES_CONTINUOUS = 0x80000000
$ES_SYSTEM_REQUIRED = 0x00000001
$ES_DISPLAY_REQUIRED = 0x00000002

Write-Host "keep_awake: blocking sleep until parent exits (pid=$PID)"

while ($true) {
    [SleepBlock]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED -bor $ES_DISPLAY_REQUIRED) | Out-Null
    Start-Sleep -Seconds 45
}
