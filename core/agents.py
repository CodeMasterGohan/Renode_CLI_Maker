"""
Agent classes for the multi-agent peripheral generation system.
"""

import time
from typing import Dict, Any, List, Optional, Callable

from .clients import LanguageModelClient, MilvusClient
from .exceptions import GenerationError


class BaseAgent:
    """Base class for all agents in the system."""

    def __init__(self, llm_client: LanguageModelClient, milvus_client: Optional[MilvusClient] = None):
        self.llm_client = llm_client
        self.milvus_client = milvus_client
        self.execution_count = 0
        self.total_execution_time = 0.0

    async def execute(self, task: Dict[str, Any]) -> Any:
        """Execute the agent's task."""
        start_time = time.time()
        try:
            result = await self._execute_impl(task)
            self.execution_count += 1
            self.total_execution_time += time.time() - start_time
            return result
        except Exception as e:
            raise GenerationError(f"Agent {self.__class__.__name__} failed: {e}") from e

    async def _execute_impl(self, task: Dict[str, Any]) -> Any:
        """Implementation-specific execution logic."""
        raise NotImplementedError("Subclasses must implement _execute_impl")

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics for this agent."""
        return {
            "execution_count": self.execution_count,
            "total_execution_time": self.total_execution_time,
            "avg_execution_time": self.total_execution_time / max(1, self.execution_count)
        }


class PlanningAgent(BaseAgent):
    """Agent responsible for planning the peripheral generation approach."""

    async def _execute_impl(self, task: Dict[str, Any]) -> List[str]:
        """Generate a plan for peripheral implementation."""
        
        prompt = task.get('prompt', '')
        
        planning_context = """
You are a planning agent for Renode peripheral generation. Given a user request for a peripheral,
create a structured plan of implementation steps. Focus on:
1. Understanding the peripheral requirements
2. Identifying key features and interfaces
3. Planning the C# class structure
4. Considering Renode-specific requirements

Return a list of specific steps to implement the peripheral.
"""
        
        planning_prompt = f"""
Create a detailed implementation plan for: {prompt}

Provide a structured approach with specific steps for creating a Renode peripheral.
Each step should be actionable and specific to Renode peripheral development.
"""

        response = await self.llm_client.generate(planning_prompt, planning_context)
        
        # Parse the response into steps (simplified parsing)
        steps = [step.strip() for step in response.split('\n') if step.strip() and not step.startswith('#')]
        return steps[:10]  # Limit to 10 steps


class CodingAgent(BaseAgent):
    """Agent responsible for generating C# peripheral code."""

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        """Generate C# peripheral code."""
        
        prompt = task.get('prompt', '')
        plan = task.get('plan', [])
        
        # Search for relevant examples if Milvus is available
        context_docs = []
        if self.milvus_client:
            try:
                # Search in both manual and examples collections
                manual_docs = await self.milvus_client.search('manual', prompt, limit=3)
                example_docs = await self.milvus_client.search('examples', prompt, limit=2)
                context_docs = manual_docs + example_docs
            except Exception:
                pass  # Continue without context if search fails
        
        coding_context = f"""
You are a C# code generation agent specialized in creating Renode peripheral implementations.

Generate complete, working C# code for Renode peripherals. Follow these guidelines:
1. Use Renode's peripheral base classes and interfaces
2. Implement proper register handling and memory mapping
3. Include comprehensive logging and error handling
4. Follow Renode naming conventions and patterns
5. Add XML documentation comments

Reference documentation and examples:
{chr(10).join(context_docs[:500])}  # Limit context size
"""

        implementation_plan = '\n'.join(f"- {step}" for step in plan) if plan else ""
        
        coding_prompt = f"""
Generate a complete C# peripheral implementation for: {prompt}

Implementation plan:
{implementation_plan}

Requirements:
- Complete C# class inheriting from appropriate Renode base class
- Proper register definitions and handling
- Interrupt support where applicable
- Comprehensive logging
- XML documentation

Provide only the C# code, properly formatted and ready to use.
"""

        response = await self.llm_client.generate(coding_prompt, coding_context)
        return response


class ReviewingAgent(BaseAgent):
    """Agent responsible for reviewing and improving generated code."""

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        """Review and improve the generated code."""
        
        code = task.get('code', '')
        prompt = task.get('prompt', '')
        
        review_context = """
You are a code review agent specialized in Renode peripheral implementations.
Review the provided C# code and identify improvements for:
1. Code quality and best practices
2. Renode-specific requirements and patterns
3. Error handling and robustness
4. Documentation completeness
5. Performance considerations

Provide the improved version of the code with all issues fixed.
"""

        review_prompt = f"""
Review and improve this Renode peripheral code for: {prompt}

Original code:
{code}

Please provide an improved version that addresses any issues with:
- Renode peripheral patterns and conventions
- Error handling and validation
- Code structure and organization
- Documentation and comments
- Performance and efficiency

Return only the improved C# code.
"""

        response = await self.llm_client.generate(review_prompt, review_context)
        return response


class AccuracyAgent(BaseAgent):
    """Agent responsible for verifying code accuracy and compliance."""

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        """Verify and validate the generated code for accuracy."""
        
        code = task.get('code', '')
        prompt = task.get('prompt', '')
        
        # Search for verification examples if available
        verification_docs = []
        if self.milvus_client:
            try:
                verification_docs = await self.milvus_client.search('manual', f"verification {prompt}", limit=2)
            except Exception:
                pass
        
        accuracy_context = f"""
You are an accuracy verification agent for Renode peripheral implementations.
Verify that the code correctly implements the requested peripheral functionality.

Focus on:
1. Correctness of peripheral behavior
2. Proper register implementation
3. Accurate interrupt handling
4. Compliance with Renode requirements
5. Functionality completeness

Verification references:
{chr(10).join(verification_docs)}
"""

        accuracy_prompt = f"""
Verify and validate this Renode peripheral implementation for: {prompt}

Code to verify:
{code}

Check for:
- Functional correctness
- Renode compliance
- Complete implementation of requested features
- Proper error handling
- Documentation accuracy

Provide the final, verified version of the code with any necessary corrections.
"""

        response = await self.llm_client.generate(accuracy_prompt, accuracy_context)
        return response


class RoutingAgent(BaseAgent):
    """Agent responsible for orchestrating the overall generation workflow."""

    def __init__(self, llm_client: LanguageModelClient, milvus_client: MilvusClient, max_iterations: int = 3):
        super().__init__(llm_client, milvus_client)
        self.max_iterations = max_iterations
        self.current_iteration = 0
        
        # Initialize sub-agents
        self.planning_agent = PlanningAgent(llm_client, milvus_client)
        self.coding_agent = CodingAgent(llm_client, milvus_client)
        self.reviewing_agent = ReviewingAgent(llm_client, milvus_client)
        self.accuracy_agent = AccuracyAgent(llm_client, milvus_client)

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        """Execute the complete peripheral generation workflow."""
        
        prompt = task.get('prompt', '')
        status_callback = task.get('status_callback', lambda msg, iteration=None: None)
        
        if not prompt:
            raise GenerationError("No prompt provided for generation")
        
        # Step 1: Planning
        status_callback("Planning peripheral implementation...", 1)
        plan = await self.planning_agent.execute({'prompt': prompt})
        
        # Step 2: Initial code generation
        status_callback("Generating initial code...", 2)
        initial_code = await self.coding_agent.execute({
            'prompt': prompt,
            'plan': plan
        })
        
        current_code = initial_code
        
        # Step 3: Iterative refinement
        for iteration in range(self.max_iterations):
            self.current_iteration = iteration + 1
            
            # Review phase
            status_callback(f"Reviewing code (iteration {iteration + 1})...", 3 + iteration * 2)
            reviewed_code = await self.reviewing_agent.execute({
                'code': current_code,
                'prompt': prompt
            })
            
            # Accuracy verification phase
            status_callback(f"Verifying accuracy (iteration {iteration + 1})...", 4 + iteration * 2)
            verified_code = await self.accuracy_agent.execute({
                'code': reviewed_code,
                'prompt': prompt
            })
            
            # Check if significant changes were made
            if self._is_code_stable(current_code, verified_code):
                current_code = verified_code
                break
                
            current_code = verified_code
        
        status_callback("Generation complete!", self.max_iterations * 2 + 2)
        return current_code

    def _is_code_stable(self, old_code: str, new_code: str) -> bool:
        """Check if the code has stabilized between iterations."""
        # Simple comparison - in practice, could use more sophisticated diff analysis
        similarity = len(set(old_code.split()) & set(new_code.split())) / max(len(old_code.split()), len(new_code.split()))
        return similarity > 0.95  # 95% similarity threshold

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for all agents in the routing workflow."""
        return {
            "routing": super().get_metrics(),
            "planning": self.planning_agent.get_metrics(),
            "coding": self.coding_agent.get_metrics(),
            "reviewing": self.reviewing_agent.get_metrics(),
            "accuracy": self.accuracy_agent.get_metrics(),
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations
        } 