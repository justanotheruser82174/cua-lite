$scriptFolder = "C:\oem"
$pythonDir = "$env:LOCALAPPDATA\Programs\Python\Python310"
$pythonExecutable = Join-Path $pythonDir "python.exe"
$pythonScript = Join-Path $scriptFolder "server\main.py"
$caddyExecutable = "C:\Users\Docker\caddy_windows_amd64.exe"
$pythonWinDir = Join-Path $pythonDir "Lib\site-packages\pythonwin"
$pywin32SystemDir = Join-Path $pythonDir "Lib\site-packages\pywin32_system32"

$env:PATH = "$pythonDir;$pythonDir\Scripts;$pythonWinDir;$pywin32SystemDir;$env:PATH"

Write-Host "Running the Caddy reverse proxy from port 9222 to port 1337"
Start-Process `
    -WindowStyle Hidden `
    -FilePath $caddyExecutable `
    -ArgumentList @("reverse-proxy", "--from", ":9222", "--to", ":1337")

Write-Host "Running the WinArena server on port 5000"
Start-Process `
    -WindowStyle Hidden `
    -WorkingDirectory (Join-Path $scriptFolder "server") `
    -FilePath $pythonExecutable `
    -ArgumentList @($pythonScript, "--port", "5000", "--log_file", "C:\oem\server-5000.log")
