#!/usr/bin/env python3
"""
Simple script to check Python imports and syntax
"""
import ast
import os
import sys
from pathlib import Path


def check_file(file_path):
    """Check a Python file for syntax errors"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse AST to check for syntax errors
        ast.parse(content)
        print(f"✓ {file_path} - OK")
        return True

    except SyntaxError as e:
        print(f"✗ {file_path} - Syntax Error: {e}")
        return False
    except Exception as e:
        print(f"✗ {file_path} - Error: {e}")
        return False


def main():
    """Check all Python files in src directory"""
    src_path = Path("src")
    if not src_path.exists():
        print("src directory not found")
        return

    python_files = list(src_path.rglob("*.py"))

    if not python_files:
        print("No Python files found in src/")
        return

    print(f"Checking {len(python_files)} Python files...")
    print()

    success_count = 0
    for py_file in python_files:
        if check_file(py_file):
            success_count += 1

    print()
    print(f"Results: {success_count}/{len(python_files)} files passed")

    if success_count == len(python_files):
        print("All files passed syntax check!")
    else:
        print("Some files have issues - see above")


if __name__ == "__main__":
    main()