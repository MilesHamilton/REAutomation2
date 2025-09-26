# Graphviz Portable Setup Script
# This script helps set up Graphviz without admin privileges

param(
    [switch]$DownloadOnly,
    [switch]$SetupPath
)

# Configuration
$toolsDir = "$env:USERPROFILE\Tools"
$graphvizDir = "$toolsDir\graphviz"
$graphvizBin = "$graphvizDir\bin"
$downloadUrl = "https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/14.0.0/windows_10_cmake_Release_graphviz-install-14.0.0-win64.zip"
$zipFile = "$toolsDir\graphviz-14.0.0-win64.zip"

Write-Host "=== Graphviz Portable Setup ===" -ForegroundColor Green
Write-Host "Target directory: $graphvizDir" -ForegroundColor Yellow

# Create tools directory
if (!(Test-Path $toolsDir)) {
    New-Item -ItemType Directory -Force -Path $toolsDir | Out-Null
    Write-Host "Created tools directory: $toolsDir" -ForegroundColor Green
}

# Function to download Graphviz
function Download-Graphviz {
    Write-Host "Downloading Graphviz..." -ForegroundColor Yellow
    
    try {
        # Use System.Net.WebClient for download with progress
        $webClient = New-Object System.Net.WebClient
        
        # Add progress handler
        Register-ObjectEvent -InputObject $webClient -EventName "DownloadProgressChanged" -Action {
            $percent = $Event.SourceEventArgs.ProgressPercentage
            Write-Progress -Activity "Downloading Graphviz" -Status "$percent% Complete" -PercentComplete $percent
        } | Out-Null
        
        $webClient.DownloadFile($downloadUrl, $zipFile)
        $webClient.Dispose()
        
        Write-Progress -Activity "Downloading Graphviz" -Completed
        Write-Host "Download completed: $zipFile" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Download failed: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Please download manually from: https://graphviz.org/download/" -ForegroundColor Yellow
        return $false
    }
}

# Function to extract Graphviz
function Extract-Graphviz {
    if (!(Test-Path $zipFile)) {
        Write-Host "ZIP file not found: $zipFile" -ForegroundColor Red
        return $false
    }
    
    Write-Host "Extracting Graphviz..." -ForegroundColor Yellow
    
    try {
        # Remove existing directory if it exists
        if (Test-Path $graphvizDir) {
            Remove-Item -Recurse -Force $graphvizDir
        }
        
        # Extract using .NET
        Add-Type -AssemblyName System.IO.Compression.FileSystem
        [System.IO.Compression.ZipFile]::ExtractToDirectory($zipFile, $toolsDir)
        
        # The extracted folder might have a different name, find it
        $extractedFolders = Get-ChildItem -Path $toolsDir -Directory | Where-Object { $_.Name -like "*graphviz*" }
        
        if ($extractedFolders.Count -gt 0) {
            $extractedFolder = $extractedFolders[0]
            if ($extractedFolder.Name -ne "graphviz") {
                Rename-Item -Path $extractedFolder.FullName -NewName "graphviz"
            }
        }
        
        Write-Host "Extraction completed to: $graphvizDir" -ForegroundColor Green
        return $true
    }
    catch {
        Write-Host "Extraction failed: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Function to setup PATH
function Setup-Path {
    if (!(Test-Path "$graphvizBin\dot.exe")) {
        Write-Host "Graphviz executables not found in: $graphvizBin" -ForegroundColor Red
        return $false
    }
    
    Write-Host "Setting up PATH..." -ForegroundColor Yellow
    
    # Get current user PATH
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    
    if ($userPath -like "*$graphvizBin*") {
        Write-Host "Graphviz already in PATH" -ForegroundColor Green
        return $true
    }
    
    # Add to PATH
    $newPath = if ($userPath) { "$userPath;$graphvizBin" } else { $graphvizBin }
    [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
    
    Write-Host "Added to PATH: $graphvizBin" -ForegroundColor Green
    Write-Host "Please restart your terminal for PATH changes to take effect" -ForegroundColor Yellow
    
    return $true
}

# Function to test installation
function Test-Installation {
    Write-Host "Testing installation..." -ForegroundColor Yellow
    
    # Test direct execution
    try {
        $dotExe = "$graphvizBin\dot.exe"
        if (Test-Path $dotExe) {
            $result = & $dotExe -V 2>&1
            Write-Host "Graphviz version: $result" -ForegroundColor Green
            return $true
        } else {
            Write-Host "dot.exe not found at: $dotExe" -ForegroundColor Red
            return $false
        }
    }
    catch {
        Write-Host "Failed to execute dot.exe: $($_.Exception.Message)" -ForegroundColor Red
        return $false
    }
}

# Main execution
if ($DownloadOnly) {
    Download-Graphviz
    exit
}

if ($SetupPath) {
    Setup-Path
    Test-Installation
    exit
}

# Full setup process
Write-Host "Starting full Graphviz setup..." -ForegroundColor Green

# Step 1: Download
if (!(Test-Path $zipFile)) {
    if (!(Download-Graphviz)) {
        Write-Host "Setup failed at download step" -ForegroundColor Red
        exit 1
    }
}

# Step 2: Extract
if (!(Test-Path "$graphvizBin\dot.exe")) {
    if (!(Extract-Graphviz)) {
        Write-Host "Setup failed at extraction step" -ForegroundColor Red
        exit 1
    }
}

# Step 3: Setup PATH
if (!(Setup-Path)) {
    Write-Host "Setup failed at PATH configuration step" -ForegroundColor Red
    exit 1
}

# Step 4: Test
if (!(Test-Installation)) {
    Write-Host "Setup completed but testing failed" -ForegroundColor Yellow
    Write-Host "Try restarting your terminal and running: dot -V" -ForegroundColor Yellow
} else {
    Write-Host "=== Setup completed successfully! ===" -ForegroundColor Green
}

# Cleanup
if (Test-Path $zipFile) {
    Remove-Item $zipFile -Force
    Write-Host "Cleaned up download file" -ForegroundColor Green
}

Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Restart your terminal" -ForegroundColor White
Write-Host "2. Test with: dot -V" -ForegroundColor White
Write-Host "3. Generate project dependencies with:" -ForegroundColor White
Write-Host "   pydeps src --max-bacon 2 --cluster -o project_dependencies.svg" -ForegroundColor Cyan
