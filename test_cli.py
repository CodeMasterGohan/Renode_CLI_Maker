#!/usr/bin/env python3
"""
Simple test script for the Renode Peripheral Generator CLI.
Tests basic functionality without requiring external dependencies.
"""

import sys
import subprocess
import os

def test_help():
    """Test that help command works."""
    print("Testing --help command...")
    try:
        result = subprocess.run([sys.executable, "renode_generator", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Help command works")
            return True
        else:
            print(f"❌ Help command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Help command error: {e}")
        return False

def test_examples():
    """Test that examples command works."""
    print("\nTesting --examples command...")
    try:
        result = subprocess.run([sys.executable, "renode_generator", "--examples"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Examples command works")
            return True
        else:
            print(f"❌ Examples command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Examples command error: {e}")
        return False

def test_version():
    """Test that version command works."""
    print("\nTesting --version command...")
    try:
        result = subprocess.run([sys.executable, "renode_generator", "--version"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Version command works")
            return True
        else:
            print(f"❌ Version command failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Version command error: {e}")
        return False

def test_no_prompt_error():
    """Test that missing prompt gives helpful error."""
    print("\nTesting missing prompt error handling...")
    try:
        result = subprocess.run([sys.executable, "renode_generator"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0 and "Prompt is required" in result.stderr:
            print("✅ Missing prompt error handling works")
            return True
        else:
            print(f"❌ Missing prompt error handling failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Missing prompt error test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Renode Peripheral Generator CLI")
    print("=" * 50)
    
    tests = [
        test_help,
        test_examples, 
        test_version,
        test_no_prompt_error
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! CLI is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 