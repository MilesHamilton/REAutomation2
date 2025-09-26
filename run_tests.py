#!/usr/bin/env python3
"""
Comprehensive test runner for REAutomation2 infrastructure

This script provides different test execution modes:
- Unit tests: Fast, isolated component tests
- Integration tests: Database and service integration tests
- E2E tests: Full end-to-end pipeline tests
- Performance tests: Load and performance validation
- All tests: Complete test suite

Usage:
    python run_tests.py --unit                 # Run unit tests only
    python run_tests.py --integration          # Run integration tests only
    python run_tests.py --e2e                  # Run E2E tests only
    python run_tests.py --all                  # Run all tests
    python run_tests.py --coverage             # Run with coverage report
    python run_tests.py --verbose              # Verbose output
"""
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_command(cmd, description, verbose=False):
    """Run a command and track execution time"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {description}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=False  # Don't raise on non-zero exit
        )

        execution_time = time.time() - start_time

        if verbose or result.returncode != 0:
            if result.stdout:
                print("STDOUT:", result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)

        print(f"✅ {description} - {execution_time:.2f}s" if result.returncode == 0
              else f"❌ {description} FAILED - {execution_time:.2f}s")

        return result.returncode == 0

    except FileNotFoundError as e:
        print(f"❌ {description} FAILED - Command not found: {e}")
        return False
    except Exception as e:
        print(f"❌ {description} FAILED - {e}")
        return False


def check_dependencies():
    """Check that required testing dependencies are available"""
    print("🔍 Checking test dependencies...")

    required_packages = [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "pytest-mock"
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r tests/requirements.txt")
        return False

    print("✅ All test dependencies available")
    return True

def install_test_dependencies():
    """Install test dependencies"""
    cmd = [sys.executable, "-m", "pip", "install", "-r", "tests/requirements.txt"]
    return run_command(cmd, "Installing test dependencies")


def run_unit_tests(verbose=False, coverage=False):
    """Run unit tests"""
    cmd = [sys.executable, "-m", "pytest", "tests/unit/", "-v"]

    if coverage:
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    if not verbose:
        cmd.append("--tb=short")

    return run_command(cmd, "Unit Tests", verbose)

def run_integration_tests(verbose=False):
    """Run integration tests"""
    cmd = [sys.executable, "-m", "pytest", "tests/integration/", "-v"]

    if not verbose:
        cmd.append("--tb=short")

    # Integration tests might need more time
    cmd.extend(["--timeout=60"])

    return run_command(cmd, "Integration Tests", verbose)

def run_e2e_tests(verbose=False):
    """Run end-to-end tests"""
    cmd = [sys.executable, "-m", "pytest", "tests/e2e/", "-v"]

    if not verbose:
        cmd.append("--tb=short")

    # E2E tests need even more time
    cmd.extend(["--timeout=120"])

    return run_command(cmd, "End-to-End Tests", verbose)

def run_performance_tests(verbose=False):
    """Run performance tests"""
    # Performance tests are marked with @pytest.mark.performance
    cmd = [sys.executable, "-m", "pytest", "-m", "performance", "-v"]

    if not verbose:
        cmd.append("--tb=short")

    cmd.extend(["--timeout=180"])  # Performance tests can take longer

    return run_command(cmd, "Performance Tests", verbose)

def run_all_tests(verbose=False):
    """Run all tests"""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v"]

    if not verbose:
        cmd.append("--tb=short")

    return run_command(cmd, "All Tests", verbose)

def run_with_coverage():
    """Run tests with detailed coverage report"""
    cmd = [
        sys.executable, "-m", "pytest", "tests/",
        "--cov=src",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=70"
    ]
    return run_command(cmd, "Tests with Coverage Analysis", True)

def run_linting():
    """Run code linting"""
    success = True

    # Try flake8 first
    flake8_cmd = [sys.executable, "-m", "flake8", "src/", "tests/"]
    if not run_command(flake8_cmd, "Linting (flake8)", False):
        # Fallback to pycodestyle if flake8 not available
        pycodestyle_cmd = [sys.executable, "-m", "pycodestyle", "src/", "tests/"]
        success = run_command(pycodestyle_cmd, "Linting (pycodestyle)", False)

    return success


def main():
    parser = argparse.ArgumentParser(description="REAutomation2 Test Runner")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests only")
    parser.add_argument("--e2e", action="store_true", help="Run E2E tests only")
    parser.add_argument("--performance", action="store_true", help="Run performance tests only")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--fast", action="store_true", help="Run unit + integration tests (CI mode)")
    parser.add_argument("--coverage", action="store_true", help="Run with coverage report")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--lint", action="store_true", help="Run linting only")
    parser.add_argument("--install-deps", action="store_true", help="Install test dependencies")

    args = parser.parse_args()

    # If no specific test type specified, run all
    if not any([args.unit, args.integration, args.e2e, args.performance,
                args.fast, args.lint, args.coverage, args.install_deps]):
        args.all = True

    print("🧪 REAutomation2 Test Suite")
    print("=" * 50)

    # Install dependencies if requested
    if args.install_deps:
        success = install_test_dependencies()
        if not success:
            sys.exit(1)
        return

    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)

    start_time = time.time()
    results = []

    # Run requested test suites
    if args.lint or args.all:
        results.append(("Linting", run_linting()))

    if args.unit or args.all or args.fast:
        results.append(("Unit Tests", run_unit_tests(args.verbose, args.coverage)))

    if args.integration or args.all or args.fast:
        results.append(("Integration Tests", run_integration_tests(args.verbose)))

    if args.e2e or args.all:
        results.append(("E2E Tests", run_e2e_tests(args.verbose)))

    if args.performance or args.all:
        results.append(("Performance Tests", run_performance_tests(args.verbose)))

    if args.coverage and not any([args.unit, args.integration, args.e2e]):
        results.append(("Coverage Analysis", run_with_coverage()))

    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for _, success in results if success)
    total = len(results)

    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)

    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20} {status}")

    print(f"\nResults: {passed}/{total} test suites passed")
    print(f"Total execution time: {total_time:.2f}s")

    if passed == total:
        print("\n🎉 All tests passed!")
        if args.coverage or any([args.unit, args.integration, args.e2e]) and args.coverage:
            print("\n📊 Coverage report available at: htmlcov/index.html")
        sys.exit(0)
    else:
        print(f"\n💥 {total - passed} test suite(s) failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
