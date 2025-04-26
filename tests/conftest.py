"""Pytest configuration file with custom reporting."""

import pytest


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom reporting after test session completion."""
    print("\n=== Test Summary ===")
    passed = len([i for i in terminalreporter.stats.get('passed', [])])
    failed = len([i for i in terminalreporter.stats.get('failed', [])])
    skipped = len([i for i in terminalreporter.stats.get('skipped', [])])
    
    print(f"✓ Passed: {passed} tests")
    if failed:
        print(f"✗ Failed: {failed} tests")
    if skipped:
        print(f"- Skipped: {skipped} tests")
    
    # Print specific test names that passed
    if passed:
        print("\nPassed Tests:")
        for i in terminalreporter.stats.get('passed', []):
            print(f"  ✓ {i.nodeid}") 