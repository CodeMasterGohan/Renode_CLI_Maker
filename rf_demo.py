#!/usr/bin/env python3
"""
RobotFramework Demo Script

This script demonstrates the RobotFramework test generation capabilities
of the Renode Peripheral Generator.
"""

import sys
import os
import traceback
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from core.config import AppConfig, ConfigManager
    from utils.status import StatusReporter
    from rf_core.rf_application import RFGeneratorCLI
    from rf_core.rf_templates import RFTemplateManager
    from rf_core.rf_validators import RFValidator
    from utils.rf_formatter import RFOutputFormatter
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available")
    sys.exit(1)


def demo_rf_templates():
    """Demonstrate RobotFramework template generation."""
    print("=" * 60)
    print("ROBOTFRAMEWORK TEMPLATE DEMO")
    print("=" * 60)
    
    template_manager = RFTemplateManager()
    
    # Generate a basic test suite
    print("\n🤖 Generating UART Integration Test Suite...")
    test_cases = [
        {
            'name': 'UART Initialization Test',
            'description': 'Test UART peripheral initialization',
            'tags': ['integration', 'uart'],
            'steps': [
                'Initialize UART With Default Config',
                'Verify UART Ready State',
                'Check UART Configuration Registers'
            ]
        },
        {
            'name': 'UART Data Transmission Test',
            'description': 'Test UART data transmission',
            'tags': ['integration', 'uart', 'transmission'],
            'steps': [
                'Configure UART For Transmission',
                'Send Test Data    Hello, Renode!',
                'Wait For Transmission Complete',
                'Verify Data Transmitted Successfully'
            ]
        }
    ]
    
    config = {
        'suite_name': 'UARTIntegrationTests',
        'keyword_library': 'ReNodeKeywords',
        'include_setup': True,
        'include_teardown': True
    }
    
    rf_code = template_manager.generate_test_suite(
        'UART', 'integration', test_cases, config
    )
    
    print(rf_code)
    
    return rf_code


def demo_rf_validation():
    """Demonstrate RobotFramework validation."""
    print("\n" + "=" * 60)
    print("ROBOTFRAMEWORK VALIDATION DEMO")
    print("=" * 60)
    
    validator = RFValidator()
    
    # Test with a simple RF code
    sample_rf_code = """*** Settings ***
Documentation     Test suite for UART peripheral
Library           ReNodeKeywords
Suite Setup       Setup Test Environment
Suite Teardown    Teardown Test Environment

*** Variables ***
${PERIPHERAL_NAME}    UART
${TIMEOUT}           30s

*** Test Cases ***
Basic UART Test
    [Documentation]    Test basic UART functionality
    [Tags]    integration    uart
    Initialize UART
    Verify UART Status
    Send Test Data
    Verify Data Received

*** Keywords ***
Initialize UART
    [Documentation]    Initialize UART peripheral
    Start Renode Simulation
    Configure UART Settings
    Log    UART initialized successfully
"""
    
    print("\n🔍 Validating RobotFramework Test Code...")
    validation_result = validator.validate_rf_content(sample_rf_code)
    
    print(f"Valid: {validation_result['valid']}")
    print(f"Quality Score: {validation_result['score']}/100")
    
    if validation_result['errors']:
        print(f"Errors: {len(validation_result['errors'])}")
        for error in validation_result['errors']:
            print(f"  ❌ {error}")
    
    if validation_result['warnings']:
        print(f"Warnings: {len(validation_result['warnings'])}")
        for warning in validation_result['warnings']:
            print(f"  ⚠️  {warning}")
    
    if validation_result['suggestions']:
        print(f"Suggestions: {len(validation_result['suggestions'])}")
        for suggestion in validation_result['suggestions']:
            print(f"  💡 {suggestion}")
    
    return validation_result


def demo_rf_formatter():
    """Demonstrate RobotFramework output formatting."""
    print("\n" + "=" * 60)
    print("ROBOTFRAMEWORK FORMATTER DEMO")
    print("=" * 60)
    
    formatter = RFOutputFormatter()
    
    # Create sample RF result
    rf_result = {
        'rf_code': """*** Settings ***
Documentation     GPIO Controller Test Suite
Library           ReNodeKeywords

*** Test Cases ***
GPIO Basic Test
    [Documentation]    Test basic GPIO functionality
    [Tags]    integration    gpio
    Initialize GPIO Controller
    Set Pin Direction    0    OUTPUT
    Set Pin Value    0    HIGH
    Verify Pin Value    0    HIGH
""",
        'peripheral_name': 'GPIO',
        'test_level': 'integration',
        'validation': {
            'valid': True,
            'score': 85,
            'errors': [],
            'warnings': ['Consider adding more test documentation'],
            'suggestions': ['Add setup and teardown keywords']
        }
    }
    
    print("\n📋 Pretty Format:")
    pretty_output = formatter.format_rf_output(rf_result, 'pretty')
    print(pretty_output)
    
    print("\n📋 JSON Format:")
    json_output = formatter.format_rf_output(rf_result, 'json')
    print(json_output)
    
    return rf_result


def demo_rf_with_mock_llm():
    """Demonstrate RobotFramework generation with mock LLM."""
    print("\n" + "=" * 60)
    print("ROBOTFRAMEWORK GENERATION DEMO (Mock Mode)")
    print("=" * 60)
    
    try:
        # Create a minimal config for demo
        config = AppConfig(
            mode='robotframework',
            llm={
                'provider': 'ollama',
                'model': 'llama3',
                'host': 'http://localhost:11434'
            },
            milvus={
                'uri': 'localhost:19530',
                'collections': {
                    'manual': 'pacer_documents',
                    'examples': 'pacer_renode_peripheral_examples',
                    'rf_examples': 'robotframework_test_examples',
                    'rf_docs': 'robotframework_documentation'
                }
            },
            robotframework={
                'enabled': True,
                'test_levels': ['integration'],
                'keyword_library': 'ReNodeKeywords',
                'output_dir': 'demo_tests'
            },
            cache={'enabled': False}  # Disable cache for demo
        )
        
        status_reporter = StatusReporter()
        status_reporter.setup(verbose=True, quiet=False, debug=False)
        
        print("\n🤖 This would normally connect to LLM and Milvus...")
        print("For demo purposes, we'll show template-based generation:")
        
        # Create RF generator
        rf_generator = RFGeneratorCLI(config, status_reporter)
        
        # Generate using templates (fallback mode)
        print("\n📝 Generating SPI Controller Test Suite...")
        rf_code = rf_generator.template_manager.generate_test_suite(
            'SPI', 'integration', [], {
                'suite_name': 'SPIControllerTests',
                'keyword_library': 'ReNodeKeywords'
            }
        )
        
        print("✅ Generated RobotFramework Test Suite:")
        print("-" * 50)
        print(rf_code)
        
        # Validate the generated code
        validation_result = rf_generator.validator.validate_rf_content(rf_code)
        print(f"\n📊 Validation Score: {validation_result['score']}/100")
        
        return {
            'rf_code': rf_code,
            'validation': validation_result,
            'peripheral_name': 'SPI',
            'test_level': 'integration'
        }
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        if "--debug" in sys.argv:
            print(traceback.format_exc())
        return None


def main():
    """Run all RobotFramework demos."""
    print("🚀 RENODE ROBOTFRAMEWORK GENERATOR DEMO")
    print("🤖 Testing RobotFramework functionality...")
    
    try:
        # Demo 1: Template Generation
        demo_rf_templates()
        
        # Demo 2: Validation
        demo_rf_validation()
        
        # Demo 3: Output Formatting
        demo_rf_formatter()
        
        # Demo 4: Full Generation (Mock Mode)
        demo_rf_with_mock_llm()
        
        print("\n" + "=" * 60)
        print("✅ ALL ROBOTFRAMEWORK DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\n💡 To use with real LLM and Milvus:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Start Milvus: docker run milvusdb/milvus")
        print("   3. Run: python renode_generator \"UART peripheral\" --mode robotframework")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        if "--debug" in sys.argv:
            print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main()) 