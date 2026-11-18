<#
.SYNOPSIS
  Install ollama-compat from GitHub Releases (Windows).

.DESCRIPTION
  irm https://raw.githubusercontent.com/shra26/ollama-compatibility-checker/main/scripts/install.ps1 | iex

  To pass parameters, save the script first, or set $env:VERSION / $env:INSTALL_DIR.
  Downloads only the release zip and checksums.txt from github.com, verifies the
  SHA-256 before extracting, and sends no telemetry.

.PARAMETER Version
  Release to install, e.g. v0.1.1 or 0.1.1 (default: latest, or $env:VERSION).

.PARAMETER InstallDir
  Destination (default: $env:LOCALAPPDATA\Programs\ollama-compat, or $env:INSTALL_DIR).

.PARAMETER AddToPath
  Add the install directory to your user PATH (otherwise only a hint is printed).
#>
[CmdletBinding()]
param(
  [string]$Version = $env:VERSION,
  [string]$InstallDir = $env:INSTALL_DIR,
  [switch]$AddToPath
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12 } catch { }

$Repo = 'shra26/ollama-compatibility-checker'
$Bin = 'ollama-compat'
if (-not $InstallDir) { $InstallDir = Join-Path $env:LOCALAPPDATA 'Programs\ollama-compat' }

switch ($env:PROCESSOR_ARCHITECTURE) {
  'AMD64' { $arch = 'amd64' }
  'ARM64' { $arch = 'arm64' }
  default { throw "Unsupported architecture: $($env:PROCESSOR_ARCHITECTURE)" }
}

if (-not $Version) {
  $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ 'User-Agent' = 'ollama-compat-installer' }
  $Version = $rel.tag_name
  if (-not $Version) { throw 'Could not determine the latest release (set -Version vX.Y.Z)' }
}
if (-not $Version.StartsWith('v')) { $Version = "v$Version" }
$num = $Version.Substring(1)

$archive = "${Bin}_${num}_windows_${arch}.zip"
$base = "https://github.com/$Repo/releases/download/$Version"
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("ollama-compat-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null

try {
  Write-Host "Installing $Bin $Version (windows/$arch)"
  Invoke-WebRequest -Uri "$base/$archive" -OutFile (Join-Path $tmp $archive) -UseBasicParsing
  Invoke-WebRequest -Uri "$base/checksums.txt" -OutFile (Join-Path $tmp 'checksums.txt') -UseBasicParsing

  $line = Get-Content (Join-Path $tmp 'checksums.txt') |
    Where-Object { ($_ -split '\s+')[1] -in @($archive, "*$archive") } | Select-Object -First 1
  if (-not $line) { throw "No checksum listed for $archive" }
  $expected = ($line -split '\s+')[0].ToLowerInvariant()
  $actual = (Get-FileHash -Algorithm SHA256 -Path (Join-Path $tmp $archive)).Hash.ToLowerInvariant()
  if ($expected -ne $actual) {
    throw "Checksum mismatch for $archive (expected $expected, got $actual); nothing installed"
  }
  Write-Host 'Checksum OK'

  $extract = Join-Path $tmp 'x'
  Expand-Archive -Path (Join-Path $tmp $archive) -DestinationPath $extract
  $exe = Join-Path $extract "$Bin.exe"
  if (-not (Test-Path $exe)) { throw "$Bin.exe not found in archive" }

  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  Copy-Item -Force $exe (Join-Path $InstallDir "$Bin.exe")
  Write-Host "Installed $(Join-Path $InstallDir "$Bin.exe")"
}
finally {
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$onPath = ($userPath -split ';') -contains $InstallDir
if (-not $onPath) {
  if ($AddToPath) {
    [Environment]::SetEnvironmentVariable('Path', ($userPath.TrimEnd(';') + ';' + $InstallDir), 'User')
    Write-Host "Added $InstallDir to your user PATH. Open a new terminal to use $Bin."
  } else {
    Write-Host "Note: $InstallDir is not on your PATH. Re-run with -AddToPath to add it to your user PATH, or run:"
    Write-Host "  [Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ';$InstallDir', 'User')"
  }
}
