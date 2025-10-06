# REAutomation2 Codebase Visualization Guide

## Overview

This guide provides multiple free tools and methods to visualize and understand your REAutomation2 codebase structure, dependencies, and architecture.

## 🚀 Quick Start: Graphviz Installation Alternatives

### Method 1: Direct Download (Recommended)

1. **Download directly from Graphviz website:**
   - Go to: https://graphviz.org/download/
   - Download Windows installer: `graphviz-14.0.0-win64.exe`
   - Run as Administrator
   - Add to PATH: `C:\Program Files\Graphviz\bin`

### Method 2: Scoop Package Manager

```bash
# Install Scoop first (if not installed)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex

# Install Graphviz
scoop install graphviz
```

### Method 3: Conda/Mamba

```bash
conda install -c conda-forge graphviz
# or
mamba install -c conda-forge graphviz
```

### Method 4: Portable Version

1. Download portable Graphviz from: https://graphviz.org/download/
2. Extract to a folder (e.g., `C:\Tools\graphviz`)
3. Add `C:\Tools\graphviz\bin` to your PATH environment variable

### Method 5: WSL (Windows Subsystem for Linux) ✅ INSTALLED

```bash
# In WSL terminal
sudo apt update
sudo apt install graphviz

# Verify installation
dot -V
# Should output: dot - graphviz version 2.43.0 (0)
```

**Status**: ✅ **Graphviz is successfully installed in your WSL environment**
- Version: 2.43.0
- All Graphviz tools available: dot, neato, circo, fdp, sfdp, twopi
- Virtual environment created with pydeps installed
- Sample dependency graphs generated successfully

**Note for WSL Users**: The error message about `/usr/bin/wslview` not found is normal and can be ignored. This just means the SVG files won't auto-open, but they are created successfully and can be viewed manually.

## 📊 Visualization Tools for Your Codebase

### 1. Pydeps (Already Installed)

**Best for:** Module dependencies and import relationships

```bash
# Basic dependency graph
pydeps src --max-bacon 2 --cluster -o project_dependencies.svg

# Focus on specific modules
pydeps src/agents --max-bacon 1 -o agents_dependencies.svg
pydeps src/voice --max-bacon 1 -o voice_dependencies.svg
pydeps src/llm --max-bacon 1 -o llm_dependencies.svg

# Text-based analysis (no Graphviz needed)
pydeps src --show-deps --no-output
pydeps src --show-cycles --no-output

# External dependencies only
pydeps src --externals --no-output
```

### 2. Pyreverse (Part of Pylint)

**Best for:** UML class diagrams and inheritance hierarchies

```bash
# Install pylint (includes pyreverse)
pip install pylint

# Generate UML diagrams
pyreverse -o png -p REAutomation2 src/
pyreverse -o svg -p REAutomation2-agents src/agents/
pyreverse -o svg -p REAutomation2-voice src/voice/
```

### 3. Code2flow

**Best for:** Function call flows and execution paths

```bash
# Install code2flow
pip install code2flow

# Generate flowcharts
code2flow src/ -o codeflow.png
code2flow src/agents/ -o agents_flow.png
code2flow src/voice/ -o voice_flow.png
```

### 4. Py2puml

**Best for:** PlantUML diagrams from Python code

```bash
# Install py2puml
pip install py2puml

# Generate PlantUML files
py2puml src/ src.puml
py2puml src/agents/ agents.puml
py2puml src/database/ database.puml
```

### 5. Sourcetrail (Interactive Explorer)

**Best for:** Interactive code navigation and exploration

1. Download from: https://github.com/CoatiSoftware/Sourcetrail/releases
2. Create a new Python project
3. Point to your `src/` directory
4. Index and explore interactively

### 6. Snakeviz (For Performance Analysis)

**Best for:** Profiling and performance visualization

```bash

```
