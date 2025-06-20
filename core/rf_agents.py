"""
RobotFramework-specific Agent classes for test generation.

This module provides specialized agents for generating RobotFramework test suites
for Renode peripherals using the same multi-agent architecture.
"""

import time
from typing import Dict, Any, List, Optional, Callable

from .agents import BaseAgent
from .clients import LanguageModelClient, MilvusClient
from .exceptions import GenerationError


class RFPlanningAgent(BaseAgent):
    """Agent responsible for planning RobotFramework test strategy."""

    async def _execute_impl(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a plan for RobotFramework test implementation."""
        
        prompt = task.get('prompt', '')
        test_level = task.get('test_level', 'integration')
        
        # Search for relevant RF examples if Milvus is available
        context_docs = []
        if self.milvus_client:
            try:
                rf_docs = await self.milvus_client.search('rf_docs', prompt, limit=3)
                rf_examples = await self.milvus_client.search('rf_examples', prompt, limit=2)
                context_docs = rf_docs + rf_examples
            except Exception:
                pass  # Continue without context if search fails
        
        planning_context = f"""
You are a RobotFramework test planning agent for Renode peripheral testing.
Given a peripheral description, create a comprehensive test strategy.

Focus on:
1. Understanding the peripheral's key functionality
2. Identifying test scenarios for the specified test level ({test_level})
3. Planning test data requirements
4. Considering Renode-specific testing aspects
5. Defining test coverage areas

Reference RF documentation and examples:
{chr(10).join(context_docs[:500]) if context_docs else "No specific examples available"}
"""

        planning_prompt = f"""
Create a detailed RobotFramework test plan for: {prompt}

Test Level: {test_level}

Provide a structured test plan including:
1. Test objectives and scope
2. Test scenarios to cover
3. Required test data
4. Setup and teardown requirements
5. Expected test cases with descriptions

Format as a structured plan with clear sections.
"""

        response = await self.llm_client.generate(planning_prompt, planning_context)
        
        # Parse the response into a structured plan
        plan = self._parse_planning_response(response, test_level)
        
        return {
            'plan': plan,
            'test_level': test_level,
            'test_scenarios': plan.get('scenarios', []),
            'test_cases': plan.get('test_cases', [])
        }
    
    def _parse_planning_response(self, response: str, test_level: str) -> Dict[str, Any]:
        """Parse the planning response into structured data."""
        # Simple parsing - in production, this would be more sophisticated
        lines = response.split('\n')
        
        plan = {
            'objectives': [],
            'scenarios': [],
            'test_cases': [],
            'requirements': [],
            'test_level': test_level
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect sections
            if 'objective' in line.lower():
                current_section = 'objectives'
            elif 'scenario' in line.lower():
                current_section = 'scenarios'
            elif 'test case' in line.lower():
                current_section = 'test_cases'
            elif 'requirement' in line.lower():
                current_section = 'requirements'
            elif line.startswith('-') or line.startswith('*'):
                if current_section and current_section in plan:
                    plan[current_section].append(line[1:].strip())
        
        # Generate default test cases if none found
        if not plan['test_cases']:
            plan['test_cases'] = self._generate_default_test_cases(test_level)
        
        return plan
    
    def _generate_default_test_cases(self, test_level: str) -> List[Dict[str, Any]]:
        """Generate default test cases based on test level."""
        if test_level == "unit":
            return [
                {
                    'name': 'Register Read Write Test',
                    'description': 'Test basic register read/write operations',
                    'tags': ['unit', 'registers'],
                    'steps': [
                        'Write Register    0x00    0x12345678',
                        '${value}=    Read Register    0x00',
                        'Should Be Equal    ${value}    0x12345678'
                    ]
                }
            ]
        elif test_level == "integration":
            return [
                {
                    'name': 'Peripheral Initialization Test',
                    'description': 'Test peripheral initialization and configuration',
                    'tags': ['integration', 'initialization'],
                    'steps': [
                        'Initialize Peripheral With Default Config',
                        'Verify Peripheral Ready State',
                        'Check Configuration Registers'
                    ]
                }
            ]
        else:  # system
            return [
                {
                    'name': 'End To End Functionality Test',
                    'description': 'Complete system test of peripheral functionality',
                    'tags': ['system', 'e2e'],
                    'steps': [
                        'Setup Complete System',
                        'Configure All Components',
                        'Execute Full Operation Sequence',
                        'Verify System Behavior'
                    ]
                }
            ]


class RFCodingAgent(BaseAgent):
    """Agent responsible for generating RobotFramework test code."""

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        """Generate RobotFramework test code."""
        
        prompt = task.get('prompt', '')
        plan = task.get('plan', {})
        test_level = task.get('test_level', 'integration')
        test_cases = task.get('test_cases', [])
        
        # Extract peripheral name from prompt
        peripheral_name = self._extract_peripheral_name(prompt)
        
        # Search for relevant RF examples if Milvus is available
        context_docs = []
        if self.milvus_client:
            try:
                rf_examples = await self.milvus_client.search('rf_examples', f"{peripheral_name} {test_level}", limit=3)
                context_docs = rf_examples
            except Exception:
                pass
        
        coding_context = f"""
You are a RobotFramework test code generation agent specialized in creating Renode peripheral tests.

Generate complete, working RobotFramework test suites following these guidelines:
1. Use proper RobotFramework syntax and structure
2. Include comprehensive test documentation
3. Use appropriate Renode-specific keywords
4. Follow RobotFramework best practices
5. Include proper setup and teardown
6. Add meaningful test tags

Reference RF examples:
{chr(10).join(context_docs[:500]) if context_docs else "No specific examples available"}

Test Level: {test_level}
"""

        coding_prompt = f"""
Generate a complete RobotFramework test suite for: {prompt}

Test Level: {test_level}
Peripheral Name: {peripheral_name}

Requirements:
- Complete RobotFramework .robot file
- Proper *** Settings ***, *** Variables ***, *** Test Cases ***, and *** Keywords *** sections
- Include setup and teardown procedures
- Add comprehensive documentation
- Use appropriate test tags
- Include error handling

Provide only the RobotFramework test code, properly formatted and ready to use.
"""

        response = await self.llm_client.generate(coding_prompt, coding_context)
        return response
    
    def _extract_peripheral_name(self, prompt: str) -> str:
        """Extract peripheral name from prompt."""
        # Simple extraction - look for common peripheral types
        prompt_lower = prompt.lower()
        peripherals = ['uart', 'spi', 'i2c', 'gpio', 'timer', 'adc', 'dac', 'pwm', 'dma', 'usb']
        
        for peripheral in peripherals:
            if peripheral in prompt_lower:
                return peripheral.upper()
        
        # Fallback to first word that might be a peripheral name
        words = prompt.split()
        for word in words:
            if len(word) > 2 and word.isalpha():
                return word.capitalize()
        
        return "GenericPeripheral"


class RFReviewingAgent(BaseAgent):
    """Agent responsible for reviewing and improving RobotFramework test quality."""

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        """Review and improve the generated RobotFramework test code."""
        
        code = task.get('code', '')
        prompt = task.get('prompt', '')
        test_level = task.get('test_level', 'integration')
        
        review_context = """
You are a RobotFramework test review agent specialized in improving test quality.
Review the provided RobotFramework test code and improve it for:

1. RobotFramework syntax and best practices
2. Test completeness and coverage
3. Code organization and readability
4. Proper documentation and comments
5. Error handling and robustness
6. Renode-specific testing patterns

Provide the improved version of the code with all issues fixed.
"""

        review_prompt = f"""
Review and improve this RobotFramework test code for: {prompt}

Current code:
{code}

Test Level: {test_level}

Please provide an improved version that addresses:
- RobotFramework syntax and structure issues
- Test quality and completeness
- Documentation and readability
- Best practices compliance

Return only the improved RobotFramework code.
"""

        response = await self.llm_client.generate(review_prompt, review_context)
        return response


class RFAccuracyAgent(BaseAgent):
    """Agent responsible for verifying RobotFramework test accuracy and compliance."""

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        """Verify and validate the RobotFramework test code for accuracy."""
        
        code = task.get('code', '')
        prompt = task.get('prompt', '')
        test_level = task.get('test_level', 'integration')
        
        accuracy_context = f"""
You are an accuracy verification agent for RobotFramework test implementations.
Verify that the test code correctly implements the requested test functionality.

Focus on:
1. Correctness of test logic and flow
2. Proper RobotFramework syntax and structure
3. Accurate test coverage for the test level
4. Appropriate use of Renode-specific features
5. Test reliability and maintainability

Test Level: {test_level}
"""

        accuracy_prompt = f"""
Verify and validate this RobotFramework test code for: {prompt}

Test code:
{code}

Test Level: {test_level}

Please verify:
1. Test logic correctness for {test_level} level testing  
2. Proper test coverage and completeness
3. RobotFramework syntax accuracy
4. Renode integration correctness
5. Test maintainability and reliability

Provide the final verified and corrected RobotFramework code.
"""

        response = await self.llm_client.generate(accuracy_prompt, accuracy_context)
        return response


class RFRoutingAgent(BaseAgent):
    """Routing agent that orchestrates the RobotFramework test generation workflow."""

    def __init__(self, llm_client: LanguageModelClient, milvus_client: MilvusClient, max_iterations: int = 3):
        super().__init__(llm_client, milvus_client)
        self.max_iterations = max_iterations
        self.current_iteration = 0
        
        # Initialize RF-specific agents
        self.planning_agent = RFPlanningAgent(llm_client, milvus_client)
        self.coding_agent = RFCodingAgent(llm_client, milvus_client)
        self.reviewing_agent = RFReviewingAgent(llm_client, milvus_client)
        self.accuracy_agent = RFAccuracyAgent(llm_client, milvus_client)

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        """Execute the RobotFramework test generation workflow."""
        
        prompt = task.get('prompt', '')
        test_level = task.get('test_level', 'integration')
        status_callback = task.get('status_callback', lambda msg, it=None: None)
        
        status_callback("Starting RobotFramework test generation...", 1)
        
        # Step 1: Planning
        status_callback("Planning test strategy...", 1)
        planning_result = await self.planning_agent.execute({
            'prompt': prompt,
            'test_level': test_level
        })
        
        # Step 2: Initial code generation
        status_callback("Generating RobotFramework test code...", 2)
        code = await self.coding_agent.execute({
            'prompt': prompt,
            'plan': planning_result.get('plan', {}),
            'test_level': test_level,
            'test_cases': planning_result.get('test_cases', [])
        })
        
        # Step 3: Iterative improvement
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration + 1
            
            status_callback(f"Reviewing and improving code (iteration {self.current_iteration})...", 
                          self.current_iteration + 2)
            
            # Review the code
            reviewed_code = await self.reviewing_agent.execute({
                'code': code,
                'prompt': prompt,
                'test_level': test_level
            })
            
            # Check for stability (stop if no significant changes)
            if self._is_code_stable(code, reviewed_code):
                code = reviewed_code
                break
                
            code = reviewed_code
        
        # Step 4: Final accuracy check
        status_callback("Performing final accuracy verification...", self.max_iterations + 3)
        final_code = await self.accuracy_agent.execute({
            'code': code,
            'prompt': prompt,
            'test_level': test_level
        })
        
        status_callback("RobotFramework test generation completed!", self.max_iterations + 4)
        
        return final_code
    
    def _is_code_stable(self, old_code: str, new_code: str) -> bool:
        """Check if the code has stabilized (minimal changes)."""
        # Simple stability check - compare lengths and common sections
        len_diff = abs(len(new_code) - len(old_code)) / max(len(old_code), 1)
        return len_diff < 0.1  # Less than 10% change
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for the routing agent."""
        base_metrics = super().get_metrics()
        base_metrics.update({
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "planning_agent_metrics": self.planning_agent.get_metrics(),
            "coding_agent_metrics": self.coding_agent.get_metrics(),
            "reviewing_agent_metrics": self.reviewing_agent.get_metrics(),
            "accuracy_agent_metrics": self.accuracy_agent.get_metrics()
        })
        return base_metrics
