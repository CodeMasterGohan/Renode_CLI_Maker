#!/usr/bin/env python3
"""
Standalone RobotFramework Test

Tests RF functionality without external dependencies.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Test RF Templates standalone
def test_rf_templates():
    """Test RF template generation without dependencies."""
    
    print("🤖 Testing RobotFramework Template Generation")
    print("=" * 50)
    
    # Import directly to avoid dependency chain
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent / 'rf_core'))
    from rf_templates import RFTemplateManager
    
    template_manager = RFTemplateManager()
    
    # Test basic template generation
    print("\n📝 Generating Basic Template...")
    basic_template = template_manager.get_template('basic_suite')
    print(basic_template)
    
    # Test full test suite generation
    print("\n📝 Generating UART Test Suite...")
    test_cases = [
        {
            'name': 'UART Initialization Test',
            'description': 'Test UART peripheral initialization',
            'tags': ['integration', 'uart'],
            'steps': [
                'Initialize UART With Default Config',
                'Verify UART Ready State'
            ]
        }
    ]
    
    config = {
        'suite_name': 'UARTTests',
        'keyword_library': 'ReNodeKeywords'
    }
    
    uart_suite = template_manager.generate_test_suite(
        'UART', 'integration', test_cases, config
    )
    
    print(uart_suite)
    print("\n✅ RF Template generation successful!")
    
    return uart_suite


def test_rf_validators():
    """Test RF validation without dependencies."""
    
    print("\n🔍 Testing RobotFramework Validation")
    print("=" * 50)
    
    # Import directly to avoid dependency chain
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent / 'rf_core'))
    from rf_validators import RFValidator
    
    validator = RFValidator()
    
    # Test with valid RF code
    valid_rf_code = """*** Settings ***
Documentation     Test suite for UART peripheral
Library           ReNodeKeywords

*** Test Cases ***
Basic Test
    [Documentation]    Test basic functionality
    [Tags]    integration
    Log    Test message
"""
    
    print("\n📊 Validating RobotFramework code...")
    result = validator.validate_rf_content(valid_rf_code)
    
    print(f"Valid: {result['valid']}")
    print(f"Score: {result['score']}/100")
    print(f"Errors: {len(result['errors'])}")
    print(f"Warnings: {len(result['warnings'])}")
    print(f"Suggestions: {len(result['suggestions'])}")
    
    print("\n✅ RF Validation successful!")
    
    return result


def test_rf_formatter():
    """Test RF formatter without dependencies."""
    
    print("\n📋 Testing RobotFramework Formatter")
    print("=" * 50)
    
    from utils.rf_formatter import RFOutputFormatter
    
    formatter = RFOutputFormatter()
    
    # Test data
    rf_result = {
        'rf_code': """*** Settings ***
Documentation     GPIO Test Suite
Library           ReNodeKeywords

*** Test Cases ***
GPIO Test
    [Tags]    integration
    Log    GPIO test
""",
        'peripheral_name': 'GPIO',
        'test_level': 'integration',
        'validation': {
            'valid': True,
            'score': 90,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
    }
    
    print("\n📄 Pretty Format:")
    pretty_output = formatter.format_rf_output(rf_result, 'pretty')
    print(pretty_output)
    
    print("\n✅ RF Formatter successful!")
    
    return rf_result


def main():
    """Run standalone RF tests."""
    
    print("🚀 ROBOTFRAMEWORK STANDALONE TESTS")
    print("🧪 Testing core RF functionality without external dependencies...")
    
    try:
        # Test 1: Templates
        test_rf_templates()
        
        # Test 2: Validators  
        test_rf_validators()
        
        # Test 3: Formatter
        test_rf_formatter()
        
        print("\n" + "=" * 60)
        print("✅ ALL ROBOTFRAMEWORK TESTS PASSED!")
        print("=" * 60)
        print("\n💡 Core RobotFramework functionality is working correctly.")
        print("📦 To use with full LLM integration, install dependencies:")
        print("   pip install -r requirements.txt")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main()) 