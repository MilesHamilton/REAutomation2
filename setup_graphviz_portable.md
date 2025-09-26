# Graphviz Portable Installation Guide (Method 1)

## Step-by-Step Implementation

### Step 1: Download Graphviz Portable Version

1. **Download the ZIP archive:**

   - URL: https://graphviz.org/download/
   - File: `graphviz-14.0.0 (64-bit) ZIP archive [sha256]`
   - Direct link: https://gitlab.com/api/v4/projects/4207231/packages/generic/graphviz-releases/14.0.0/windows_10_cmake_Release_graphviz-install-14.0.0-win64.exe.zip

2. **Alternative download locations:**
   - GitHub releases: https://github.com/xflr6/graphviz/releases
   - Official GitLab: https://gitlab.com/graphviz/graphviz/-/releases

### Step 2: Extract to User Directory

1. **Create a tools directory:**

   ```
   C:\Users\mhamilton\Tools\
   ```

2. **Extract the ZIP file to:**

   ```
   C:\Users\mhamilton\Tools\graphviz\
   ```

3. **Verify the structure looks like:**
   ```
   C:\Users\mhamilton\Tools\graphviz\
   ├── bin\
   │   ├── dot.exe
   │   ├── neato.exe
   │   ├── twopi.exe
   │   └── ... (other executables)
   ├── lib\
   ├── share\
   └── ... (other directories)
   ```

### Step 3: Add to User PATH

1. **Open Environment Variables:**

   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Click "Environment Variables..." button
   - In the "User variables" section (NOT System variables)

2. **Edit PATH variable:**
   - Select "Path" in User variables
   - Click "Edit..."
   - Click "New"
   - Add: `C:\Users\mhamilton\Tools\graphviz\bin`
   - Click "OK" on all dialogs

### Step 4: Verify Installation

Open a new Command Prompt or PowerShell and run:

```bash
dot -V
```

Expected output:

```
dot - graphviz version 14.0.0 (...)
```

### Step 5: Test with Your Project

Navigate to your project directory and test:

```bash
cd "C:\Users\mhamilton\OneDrive - Credence Management Solutions LLC\Projects\REAutomation2"
pydeps src --max-bacon 2 --cluster -o project_dependencies.svg
```

## Troubleshooting

### If `dot -V` doesn't work:

1. **Restart your terminal** - PATH changes require a new session
2. **Check PATH manually:**

   ```bash
   echo $env:PATH
   ```

   Look for your graphviz path in the output

3. **Test direct execution:**
   ```bash
   "C:\Users\mhamilton\Tools\graphviz\bin\dot.exe" -V
   ```

### If pydeps still fails:

1. **Check Python can find dot:**

   ```python
   import subprocess
   subprocess.run(['dot', '-V'])
   ```

2. **Set GRAPHVIZ_DOT environment variable:**
   - Add new user environment variable:
   - Name: `GRAPHVIZ_DOT`
   - Value: `C:\Users\mhamilton\Tools\graphviz\bin\dot.exe`

## Alternative: PowerShell Setup Script

Save this as `setup_graphviz.ps1` and run it:

```powershell
# Create tools directory
$toolsDir = "$env:USERPROFILE\Tools"
$graphvizDir = "$toolsDir\graphviz"
New-Item -ItemType Directory -Force -Path $toolsDir

Write-Host "Tools directory created at: $toolsDir"
Write-Host "Please download and extract Graphviz ZIP to: $graphvizDir"
Write-Host "Then run this script again to update PATH"

# Check if graphviz is already extracted
if (Test-Path "$graphvizDir\bin\dot.exe") {
    # Add to user PATH
    $userPath = [Environment]::GetEnvironmentVariable("PATH", "User")
    $graphvizBin = "$graphvizDir\bin"

    if ($userPath -notlike "*$graphvizBin*") {
        $newPath = "$userPath;$graphvizBin"
        [Environment]::SetEnvironmentVariable("PATH", $newPath, "User")
        Write-Host "Added $graphvizBin to user PATH"
        Write-Host "Please restart your terminal and run: dot -V"
    } else {
        Write-Host "Graphviz already in PATH"
    }
} else {
    Write-Host "Graphviz not found. Please extract the ZIP file to $graphvizDir first."
}
```

## Next Steps After Installation

Once Graphviz is working, you can generate your project visualizations:

```bash
# Basic dependency graph
pydeps src --max-bacon 2 --cluster -o project_dependencies.svg

# Module-specific graphs
pydeps src/agents --max-bacon 1 -o agents_dependencies.svg
pydeps src/voice --max-bacon 1 -o voice_dependencies.svg
pydeps src/llm --max-bacon 1 -o llm_dependencies.svg
pydeps src/database --max-bacon 1 -o database_dependencies.svg

# External dependencies
pydeps src --externals -o external_dependencies.svg
```

The generated SVG files can be opened in any web browser or image viewer.
