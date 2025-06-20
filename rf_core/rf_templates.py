"""
RobotFramework Templates Module

This module provides templates and patterns for generating RobotFramework test suites
for Renode peripheral testing.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path


class RFTemplateManager:
    """Manages RobotFramework test templates and patterns."""
    
    def __init__(self):
        self.templates = self._initialize_templates()
        
    def _initialize_templates(self) -> Dict[str, str]:
        """Initialize the RF test templates."""
        return {
            "basic_suite": self._get_basic_suite_template(),
            "peripheral_test": self._get_peripheral_test_template(),
            "integration_test": self._get_integration_test_template(),
            "unit_test": self._get_unit_test_template(),
            "system_test": self._get_system_test_template()
        }
    
    def get_template(self, template_name: str) -> str:
        """Get a specific template by name."""
        return self.templates.get(template_name, "")
    
    def generate_test_suite(self, peripheral_name: str, test_level: str, 
                          test_cases: List[Dict[str, Any]], 
                          config: Optional[Dict[str, Any]] = None) -> str:
        """Generate a complete RobotFramework test suite."""
        
        config = config or {}
        suite_name = config.get('suite_name', f'{peripheral_name}Tests')
        keyword_library = config.get('keyword_library', 'ReNodeKeywords')
        
        # Build the test suite
        suite_parts = []
        
        # Header and settings
        suite_parts.append(self._generate_suite_header(suite_name, peripheral_name))
        suite_parts.append(self._generate_settings_section(keyword_library))
        
        # Variables section
        if config.get('include_variables', True):
            suite_parts.append(self._generate_variables_section(peripheral_name))
        
        # Test cases
        suite_parts.append(self._generate_test_cases_section(test_cases, test_level))
        
        # Keywords section  
        if config.get('include_keywords', True):
            suite_parts.append(self._generate_keywords_section(peripheral_name, test_level))
        
        return '\n\n'.join(suite_parts)
    
    def _generate_suite_header(self, suite_name: str, peripheral_name: str) -> str:
        """Generate the test suite header."""
        return f"""*** Settings ***
Documentation     Test suite for {peripheral_name} peripheral
...               This suite tests the functionality of the {peripheral_name} 
...               peripheral implementation in Renode.
Suite Setup       Setup Test Environment
Suite Teardown    Teardown Test Environment
Test Tags         {peripheral_name.lower()}    renode    peripheral"""
    
    def _generate_settings_section(self, keyword_library: str) -> str:
        """Generate the settings section."""
        return f"""
Library           {keyword_library}
Library           Collections
Library           String
Resource          renode_common.resource
Variables         test_config.py"""
    
    def _generate_variables_section(self, peripheral_name: str) -> str:
        """Generate the variables section."""
        peripheral_lower = peripheral_name.lower()
        return f"""*** Variables ***
${{PERIPHERAL_NAME}}     {peripheral_name}
${{RENODE_SCRIPT}}       {peripheral_lower}_test.resc
${{EXPECTED_REGISTERS}}  Create List  
${{TEST_DATA_FILE}}      {peripheral_lower}_test_data.json
${{TIMEOUT}}             30s"""
    
    def _generate_test_cases_section(self, test_cases: List[Dict[str, Any]], test_level: str) -> str:
        """Generate the test cases section."""
        cases = ["*** Test Cases ***"]
        
        for case in test_cases:
            case_name = case.get('name', 'Unknown Test')
            case_description = case.get('description', '')
            case_steps = case.get('steps', [])
            case_tags = case.get('tags', [test_level])
            
            case_text = [f"{case_name}"]
            if case_description:
                case_text.append(f"    [Documentation]    {case_description}")
            if case_tags:
                case_text.append(f"    [Tags]    {' '.join(case_tags)}")
            
            # Add test steps
            for step in case_steps:
                case_text.append(f"    {step}")
            
            if not case_steps:
                # Add default steps
                case_text.extend([
                    "    Start Renode With Script    ${RENODE_SCRIPT}",
                    "    Verify Peripheral Initialization    ${PERIPHERAL_NAME}",
                    "    Perform Basic Operations",
                    "    Verify Expected Behavior"
                ])
            
            cases.append('\n'.join(case_text))
        
        # Add default test cases if none provided
        if not test_cases:
            cases.extend(self._get_default_test_cases(test_level))
        
        return '\n\n'.join(cases)
    
    def _generate_keywords_section(self, peripheral_name: str, test_level: str) -> str:
        """Generate the keywords section."""
        keywords = [
            "*** Keywords ***",
            self._get_setup_keyword(),
            self._get_teardown_keyword(),
            self._get_peripheral_keywords(peripheral_name, test_level)
        ]
        return '\n\n'.join(keywords)
    
    def _get_default_test_cases(self, test_level: str) -> List[str]:
        """Get default test cases based on test level."""
        if test_level == "unit":
            return [
                """Register Read Write Test
    [Documentation]    Test basic register read/write operations
    [Tags]    unit    registers
    Write Register    0x00    0x12345678
    ${value}=    Read Register    0x00
    Should Be Equal    ${value}    0x12345678""",
                
                """Reset Functionality Test
    [Documentation]    Test peripheral reset functionality
    [Tags]    unit    reset
    Reset Peripheral
    Verify Reset State
    Check Default Register Values"""
            ]
        elif test_level == "integration":
            return [
                """Peripheral Initialization Test
    [Documentation]    Test peripheral initialization and configuration
    [Tags]    integration    initialization
    Initialize Peripheral With Default Config
    Verify Peripheral Ready State
    Check Configuration Registers""",
                
                """Interrupt Handling Test
    [Documentation]    Test interrupt generation and handling
    [Tags]    integration    interrupts
    Enable Interrupts
    Trigger Test Condition
    Wait For Interrupt    timeout=5s
    Verify Interrupt Status
    Clear Interrupt"""
            ]
        else:  # system
            return [
                """End To End Functionality Test
    [Documentation]    Complete system test of peripheral functionality
    [Tags]    system    e2e
    Setup Complete System
    Configure All Components
    Execute Full Operation Sequence
    Verify System Behavior
    Check All Expected Outputs""",
                
                """Performance Test
    [Documentation]    Test peripheral performance characteristics
    [Tags]    system    performance
    Setup Performance Test Environment
    Execute Performance Test Sequence
    Measure Performance Metrics
    Verify Performance Requirements"""
            ]
    
    def _get_setup_keyword(self) -> str:
        """Get the setup keyword."""
        return """Setup Test Environment
    [Documentation]    Setup the test environment for peripheral testing
    Initialize Renode Environment
    Load Renode Configuration
    Set Test Variables
    Log    Test environment setup complete"""
    
    def _get_teardown_keyword(self) -> str:
        """Get the teardown keyword."""
        return """Teardown Test Environment
    [Documentation]    Clean up test environment
    Stop Renode Simulation
    Clean Up Test Files
    Reset Test Variables
    Log    Test environment cleanup complete"""
    
    def _get_peripheral_keywords(self, peripheral_name: str, test_level: str) -> str:
        """Get peripheral-specific keywords."""
        return f"""Initialize Peripheral With Default Config
    [Documentation]    Initialize {peripheral_name} with default configuration
    Start Renode With Script    ${{RENODE_SCRIPT}}
    Wait For Peripheral Ready
    Log    {peripheral_name} initialized successfully

Verify Peripheral Ready State
    [Documentation]    Verify {peripheral_name} is in ready state
    ${{status}}=    Get Peripheral Status
    Should Be Equal    ${{status}}    READY
    Log    {peripheral_name} is ready

Perform Basic Operations
    [Documentation]    Perform basic {peripheral_name} operations
    Execute Basic Operation Sequence
    Verify Operation Results
    Log    Basic operations completed successfully

Verify Expected Behavior
    [Documentation]    Verify {peripheral_name} behaves as expected
    Check Expected Outputs
    Validate State Transitions
    Verify Error Handling
    Log    Expected behavior verified"""
    
    def _get_basic_suite_template(self) -> str:
        """Get basic suite template."""
        return """*** Settings ***
Documentation     Basic RobotFramework test suite template
Library           ReNodeKeywords
Suite Setup       Setup Test Environment
Suite Teardown    Teardown Test Environment

*** Variables ***
${TIMEOUT}        30s

*** Test Cases ***
Basic Test
    [Documentation]    Basic test case template
    Log    This is a basic test

*** Keywords ***
Setup Test Environment
    Log    Setting up test environment

Teardown Test Environment
    Log    Cleaning up test environment"""
    
    def _get_peripheral_test_template(self) -> str:
        """Get peripheral-specific test template."""
        return """*** Settings ***
Documentation     Peripheral test template for Renode
Library           ReNodeKeywords
Resource          renode_common.resource

*** Test Cases ***
Peripheral Initialization Test
    Initialize Peripheral
    Verify Peripheral State

*** Keywords ***
Initialize Peripheral
    Start Renode Simulation
    Configure Peripheral

Verify Peripheral State
    Check Peripheral Status
    Validate Configuration"""
    
    def _get_integration_test_template(self) -> str:
        """Get integration test template."""
        return """*** Settings ***
Documentation     Integration test template
Test Tags         integration

*** Test Cases ***
Integration Test
    Setup Integration Environment
    Execute Integration Scenario
    Verify Integration Results"""
    
    def _get_unit_test_template(self) -> str:
        """Get unit test template."""
        return """*** Settings ***
Documentation     Unit test template
Test Tags         unit

*** Test Cases ***
Unit Test
    Execute Unit Test Scenario
    Verify Unit Test Results"""
    
    def _get_system_test_template(self) -> str:
        """Get system test template."""
        return """*** Settings ***
Documentation     System test template
Test Tags         system

*** Test Cases ***
System Test
    Setup System Test Environment
    Execute System Test Scenario
    Verify System Test Results""" 