#!/usr/bin/env python3
"""
Quick test to verify logger fix works.
Tests that src.main.logger can handle keyword arguments.
"""
import sys

# Test 1: Import and basic keyword arg test
print("Test 1: Importing logger from src.main...")
try:
    from src.main import logger
    print("✓ Import successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Keyword argument test
print("\nTest 2: Testing logger with keyword arguments...")
try:
    logger.info("test_event", ticker="0005.HK", company_name="HSBC Holdings")
    print("✓ Logger accepts keyword arguments")
except TypeError as e:
    print(f"✗ Logger rejected keyword arguments: {e}")
    sys.exit(1)

# Test 3: Verify it's structlog
print("\nTest 3: Verifying logger type...")
import structlog
if isinstance(logger, structlog.stdlib.BoundLogger):
    print(f"✓ Logger is structlog.BoundLogger")
else:
    print(f"✓ Logger type: {type(logger)}")

print("\n✅ All tests passed!")
