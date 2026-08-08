# Detect current WLAN IP, re-sign local cert with mkcert to fixed path certs/cert.pem + certs/key.pem.
# SAN: localhost / 127.0.0.1 / ::1 / hostname / hostname.local / current WLAN IP.
# Called by start.bat before services launch so Android collaborators can use IP directly.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".tools\mkcert.exe")) {
    Write-Host "[renew-cert] .tools\mkcert.exe missing, skip" -ForegroundColor Yellow
    exit 0
}

$wlanIp = $null
$wlan = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object { ($_.InterfaceAlias -match "WLAN|Wi-Fi|Wireless") -and ($_.PrefixOrigin -in "Dhcp","Manual") } |
    Select-Object -First 1
if ($wlan) {
    $wlanIp = $wlan.IPAddress
} else {
    $eth = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { ($_.InterfaceAlias -match "Ethernet") -and ($_.PrefixOrigin -in "Dhcp","Manual") } |
        Select-Object -First 1
    if ($eth) { $wlanIp = $eth.IPAddress }
}

$sans = @("localhost", "127.0.0.1", "::1", $env:COMPUTERNAME, "$($env:COMPUTERNAME).local")
if ($wlanIp) {
    $sans += $wlanIp
    Write-Host "[renew-cert] WLAN IP: $wlanIp"
} else {
    Write-Host "[renew-cert] No WLAN/Ethernet IP, hostname only" -ForegroundColor Yellow
}

if (-not (Test-Path "certs")) { New-Item -ItemType Directory -Path "certs" | Out-Null }

$sanStr = $sans -join ", "
Write-Host "[renew-cert] signing cert.pem/key.pem (SAN: $sanStr)"

$mkArgs = @("-cert-file", "certs\cert.pem", "-key-file", "certs\key.pem") + $sans
& ".\.tools\mkcert.exe" @mkArgs 2>&1 | ForEach-Object { Write-Host "  $_" }
if ($LASTEXITCODE -ne 0) {
    Write-Host "[renew-cert] mkcert failed" -ForegroundColor Red
    exit 1
}
Write-Host "[renew-cert] OK" -ForegroundColor Green

# Ensure hosts entry for <hostname>.local, else browser can't resolve https://<hostname>.local
# (cert SAN includes it, but DNS won't). Needs admin; non-admin just warns, doesn't block start.
# Only IPv4 127.0.0.1: IPv6 ::1 lookup on Windows localhost is slow (~2s timeout).
$hostsPath = "$env:WINDIR\System32\drivers\etc\hosts"
$hostEntry = "$($env:COMPUTERNAME.ToLower()).local"
if (-not (Select-String -Path $hostsPath -Pattern $hostEntry -SimpleMatch -Quiet -ErrorAction SilentlyContinue)) {
    try {
        Add-Content -Path $hostsPath -Value "127.0.0.1 $hostEntry" -ErrorAction Stop
        Write-Host "[renew-cert] added hosts: 127.0.0.1 $hostEntry" -ForegroundColor Green
    } catch {
        Write-Host "[renew-cert] cannot write hosts (run as admin to enable https://$hostEntry)" -ForegroundColor Yellow
    }
} else {
    Write-Host "[renew-cert] hosts already has $hostEntry"
}
