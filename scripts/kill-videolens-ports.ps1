# 测试/运维专用 — 一键清理 VideoLens 占用的端口
# 用法: powershell -ExecutionPolicy Bypass -File scripts/kill-videolens-ports.ps1
$ports = 3000, 2024, 9800, 9801, 9802
foreach ($p in $ports) {
    $line = netstat -ano | findstr ":$p " | findstr LISTENING
    if ($line) {
        $procId = ($line -split '\s+')[-1]
        Write-Host "Killing port $p PID $procId"
        taskkill /F /PID $procId
    } else {
        Write-Host "Port $p free"
    }
}
