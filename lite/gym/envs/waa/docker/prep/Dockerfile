# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=windowsarena/winarena@sha256:869afffe384f0e0356dbfcdf78486611e31d62e692544cbfd9a6c08606f07440
FROM ${BASE_IMAGE}

COPY on-logon.ps1 /oem/on-logon.ps1

RUN python3 - <<'PY'
import json
from pathlib import Path

tools_path = Path("/oem/tools_config.json")
tools = json.loads(tools_path.read_text(encoding="utf-8"))
tools["LibreOffice"]["mirrors"] = [
    "https://downloadarchive.documentfoundation.org/libreoffice/old/24.8.2.1/win/x86_64/LibreOffice_24.8.2.1_Win_x86-64.msi"
]
tools_path.write_text(json.dumps(tools, indent=4) + "\n", encoding="utf-8")

server_path = Path("/oem/server/main.py")
server = server_path.read_text(encoding="utf-8")

replacements = {
    'return pyautogui.position().x, pyautogui.position().y': (
        'position = pyautogui.position()\n'
        '    return jsonify({"x": position.x, "y": position.y})'
    ),
    'if isinstance(command, str) and not shell:\n        command = shlex.split(command)': (
        'if isinstance(command, str) and not shell:\n'
        '        command = shlex.split(command)\n\n'
        '    if not shell and command and command[0].lower() in {"python", "python.exe"}:\n'
        '        command[0] = sys.executable'
    ),
    "if 'google-chrome' in command and _get_machine_architecture() == 'arm':": (
        "if (\n"
        "            user_platform == 'Windows'\n"
        "            and command\n"
        "            and str(command[0]).lower() in {'chrome', 'chrome.exe'}\n"
        "            and any(str(arg).startswith('--remote-debugging-port=') for arg in command)\n"
        "            and not any(str(arg).startswith('--user-data-dir=') for arg in command)\n"
        "        ):\n"
        "            command.append(\n"
        "                r'--user-data-dir=C:\\Users\\Docker\\AppData\\Local\\Google\\Chrome\\WAA-Remote'\n"
        "            )\n\n"
        "        if 'google-chrome' in command and _get_machine_architecture() == 'arm':"
    ),
    'app.run(debug=True, host="0.0.0.0", port=args.port)': (
        'app.run(debug=False, host="0.0.0.0", port=args.port)'
    ),
}
for old, new in replacements.items():
    if old not in server:
        raise RuntimeError(f"expected WAA server source fragment not found: {old!r}")
    server = server.replace(old, new, 1)

server_path.write_text(server, encoding="utf-8")

setup_path = Path("/oem/setup.ps1")
setup = setup_path.read_text(encoding="utf-8")
pywinauto_install = 'Install-PythonPackages -Package "pywinauto"'
pywin32_setup = r'''Install-PythonPackages -Package "pywinauto"
# pywin32 312 dropped mfc140u.dll from its wheel. WAA imports win32ui at
# server startup, so retain the last wheel that includes the required DLL.
Install-PythonPackages -Package "pywin32==311" -Arguments "--force-reinstall"
& python -c "import win32ui"
if ($LASTEXITCODE -ne 0) {
    Write-Error "pywin32 validation failed: win32ui could not be imported."
    exit 1
}'''
if pywinauto_install not in setup:
    raise RuntimeError("WAA pywinauto installation command not found in setup.ps1")
setup = setup.replace(pywinauto_install, pywin32_setup, 1)

chrome_marker = "\n# - LibreOffice"
chrome_profile_setup = r'''

# Chrome 136+ disables remote debugging for the default user data directory.
# WAA tasks require CDP while its evaluators read the historical default paths.
# Keep one physical profile at a non-default path and expose the old path as a
# directory junction so both assumptions remain true.
$chromeDataParent = Join-Path $env:LOCALAPPDATA "Google\Chrome"
$chromeDefaultData = Join-Path $chromeDataParent "User Data"
$chromeRemoteData = Join-Path $chromeDataParent "WAA-Remote"
New-Item -ItemType Directory -Path $chromeDataParent -Force | Out-Null
if ((Test-Path $chromeDefaultData) -and -not (Test-Path $chromeRemoteData)) {
    Move-Item -Path $chromeDefaultData -Destination $chromeRemoteData
}
New-Item -ItemType Directory -Path $chromeRemoteData -Force | Out-Null
if (Test-Path $chromeDefaultData) {
    Remove-Item -Path $chromeDefaultData -Recurse -Force
}
New-Item -ItemType Junction `
    -Path $chromeDefaultData `
    -Target $chromeRemoteData | Out-Null

# - LibreOffice'''
if chrome_marker not in setup:
    raise RuntimeError("WAA Chrome setup block terminator not found")
setup = setup.replace(chrome_marker, chrome_profile_setup, 1)

libreoffice_marker = "}\n\n# - VLC"
libreoffice_warmup = r'''}

# LibreOffice's MSI defers Writer components until the first GUI launch. On a
# fresh Windows user it can also open a release URL, which blocks on the
# Windows 11 protocol chooser when no https default has been accepted yet.
# Complete both one-time paths before sealing the golden image.
$libreOfficeExecutable = "C:\Program Files\LibreOffice\program\soffice.exe"
if (Test-Path $libreOfficeExecutable) {
    Write-Host "Warming up LibreOffice Writer..."
    $libreOfficeProfile = Join-Path $env:APPDATA "LibreOffice\4\user"
    New-Item -ItemType Directory -Path $libreOfficeProfile -Force | Out-Null
    @'
<?xml version="1.0" encoding="UTF-8"?>
<oor:items xmlns:oor="http://openoffice.org/2001/registry">
  <item oor:path="/org.openoffice.Office.Jobs/Jobs/org.openoffice.Office.Jobs:Job['UpdateCheck']/Arguments">
    <prop oor:name="AutoCheckEnabled" oor:op="fuse"><value>false</value></prop>
  </item>
</oor:items>
'@ | Set-Content `
        -Path (Join-Path $libreOfficeProfile "registrymodifications.xcu") `
        -Encoding UTF8

    Start-Process `
        -FilePath $libreOfficeExecutable `
        -ArgumentList @("--writer", "--nologo", "--nofirststartwizard", "--norestore")
    $libreOfficeDeadline = (Get-Date).AddMinutes(15)
    $libreOfficeReadySince = $null
    do {
        # The protocol chooser is infrastructure setup, not part of a task.
        Get-Process -Name OpenWith -ErrorAction SilentlyContinue |
            Stop-Process -Force -ErrorAction SilentlyContinue
        $libreOfficeProcesses = Get-Process -Name "soffice*" -ErrorAction SilentlyContinue
        $libreOfficeWindow = $libreOfficeProcesses |
            Where-Object { $_.MainWindowHandle -ne 0 } |
            Select-Object -First 1
        if ($libreOfficeWindow) {
            if (-not $libreOfficeReadySince) {
                $libreOfficeReadySince = Get-Date
            } elseif (((Get-Date) - $libreOfficeReadySince).TotalSeconds -ge 15) {
                break
            }
        } else {
            $libreOfficeReadySince = $null
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $libreOfficeDeadline)
    if (
        -not $libreOfficeWindow -or
        -not $libreOfficeReadySince -or
        ((Get-Date) - $libreOfficeReadySince).TotalSeconds -lt 15
    ) {
        Stop-Process -Name "soffice*" -Force -ErrorAction SilentlyContinue
        Write-Error "LibreOffice Writer warm-up timed out."
        exit 1
    }
    $libreOfficeWindow.CloseMainWindow() | Out-Null
    Start-Sleep -Seconds 10
    Stop-Process -Name "soffice*" -Force -ErrorAction SilentlyContinue
}

# - VLC'''
if libreoffice_marker not in setup:
    raise RuntimeError("WAA LibreOffice setup block terminator not found")
setup = setup.replace(libreoffice_marker, libreoffice_warmup, 1)
setup_path.write_text(setup, encoding="utf-8")
PY
