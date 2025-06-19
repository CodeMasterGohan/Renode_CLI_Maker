"""
Main application class for the Renode Peripheral Generator CLI.
"""

import json
import time
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from .config import AppConfig
from .exceptions import RenodeGeneratorError, LLMClientError, MilvusClientError
from .clients import LanguageModelClient, MilvusClient
from .agents import RoutingAgent
from .cache import InMemoryCache
from utils.status import StatusReporter


class RenodeGeneratorCLI:
    """Main application class that orchestrates the peripheral generation process."""
    
    def __init__(self, config: AppConfig, status_reporter: StatusReporter):
        self.config = config
        self.status_reporter = status_reporter
        self.metrics = {
            'start_time': time.time(),
            'end_time': None,
            'iterations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': []
        }
        
        # Initialize cache
        self.cache = None
        if config.cache.enabled:
            cache_dir = Path(config.cache.directory).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache = InMemoryCache(
                default_ttl=config.cache.ttl,
                max_size=config.cache.max_size
            )
            self.status_reporter.verbose(f"Cache initialized at: {cache_dir}")
        
        # Initialize clients
        self._initialize_clients()
        
        # Initialize routing agent
        self.routing_agent = RoutingAgent(
            llm_client=self.llm_client,
            milvus_client=self.milvus_client,
            max_iterations=3  # Default, can be overridden
        )
        
    def _initialize_clients(self):
        """Initialize LLM and Milvus clients."""
        try:
            # Initialize LLM client
            self.status_reporter.verbose("Initializing LLM client...")
            self.llm_client = LanguageModelClient(
                llm_provider=self.config.llm.provider,
                model_name=self.config.llm.model,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.host,
                max_retries=self.config.llm.max_retries,
                cache=self.cache
            )
            self.status_reporter.verbose(f"LLM client initialized: {self.config.llm.provider}/{self.config.llm.model}")
            
            # Initialize Milvus client
            self.status_reporter.verbose("Initializing Milvus client...")
            self.milvus_client = MilvusClient(
                uri=self.config.milvus.uri,
                collection_names=self.config.milvus.collections,
                embedding_provider=self.config.embedding.provider,
                embedding_model_name=self.config.embedding.model,
                embedding_api_key=self.config.embedding.api_key,
                embedding_base_url=self.config.embedding.host,
                cache=self.cache
            )
            self.status_reporter.verbose(f"Milvus client initialized: {self.config.milvus.uri}")
            
        except Exception as e:
            error_msg = f"Failed to initialize clients: {e}"
            self.metrics['errors'].append(error_msg)
            raise RenodeGeneratorError(error_msg) from e
    
    def run(self, prompt: str, max_iterations: int = 3, use_cache: bool = True, 
            save_plan: Optional[str] = None) -> str:
        """Run the peripheral generation process."""
        
        self.status_reporter.verbose(f"Starting generation with prompt: {prompt}")
        self.status_reporter.verbose(f"Max iterations: {max_iterations}")
        self.status_reporter.verbose(f"Cache enabled: {use_cache and self.config.cache.enabled}")
        
        try:
            # Update routing agent configuration
            self.routing_agent.max_iterations = max_iterations
            
            # Create task for the routing agent
            task = {
                'prompt': prompt,
                'use_cache': use_cache and self.config.cache.enabled
            }
            
            # Save execution plan if requested
            if save_plan:
                self._save_plan(task, save_plan)
            
            # Run the generation process
            self.status_reporter.status_update("Executing generation pipeline...")
            
            # Use asyncio to run the async agent
            result = asyncio.run(self._run_generation(task))
            
            # Update metrics
            self.metrics['end_time'] = time.time()
            self.metrics['iterations'] = self.routing_agent.current_iteration
            
            # Update cache metrics if available
            if self.cache:
                cache_stats = self.cache.get_stats()
                # Note: These would need to be tracked in the cache implementation
                # self.metrics['cache_hits'] = cache_stats.get('hits', 0)
                # self.metrics['cache_misses'] = cache_stats.get('misses', 0)
            
            self.status_reporter.newline()
            self.status_reporter.success("Generation completed successfully!")
            
            return result
            
        except Exception as e:
            self.metrics['end_time'] = time.time()
            error_msg = f"Generation failed: {e}"
            self.metrics['errors'].append(error_msg)
            self.status_reporter.error(error_msg)
            raise RenodeGeneratorError(error_msg) from e
    
    async def _run_generation(self, task: Dict[str, Any]) -> str:
        """Run the actual generation process asynchronously."""
        
        # Create a status callback for the routing agent
        def status_callback(message: str, iteration: int = None):
            if iteration is not None:
                self.status_reporter.progress(message, iteration, self.routing_agent.max_iterations)
            else:
                self.status_reporter.status_update(message)
        
        # Add status callback to task
        task['status_callback'] = status_callback
        
        # Execute the routing agent
        result = await self.routing_agent.execute(task)
        
        return result
    
    def run_from_plan(self, plan_file: str) -> str:
        """Run generation from a saved execution plan."""
        
        self.status_reporter.verbose(f"Loading execution plan from: {plan_file}")
        
        try:
            with open(plan_file, 'r') as f:
                plan = json.load(f)
            
            # Extract parameters from plan
            prompt = plan.get('prompt', '')
            max_iterations = plan.get('max_iterations', 3)
            use_cache = plan.get('use_cache', True)
            
            if not prompt:
                raise ValueError("Plan file does not contain a valid prompt")
            
            self.status_reporter.info(f"Executing plan: {prompt}")
            
            return self.run(
                prompt=prompt,
                max_iterations=max_iterations,
                use_cache=use_cache
            )
            
        except Exception as e:
            error_msg = f"Failed to load execution plan: {e}"
            self.metrics['errors'].append(error_msg)
            raise RenodeGeneratorError(error_msg) from e
    
    def _save_plan(self, task: Dict[str, Any], plan_file: str):
        """Save execution plan to file."""
        
        plan = {
            'prompt': task['prompt'],
            'max_iterations': self.routing_agent.max_iterations,
            'use_cache': task['use_cache'],
            'timestamp': time.time(),
            'config': {
                'llm_provider': self.config.llm.provider,
                'llm_model': self.config.llm.model,
                'milvus_uri': self.config.milvus.uri
            }
        }
        
        try:
            # Ensure directory exists
            plan_path = Path(plan_file)
            plan_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(plan_path, 'w') as f:
                json.dump(plan, f, indent=2)
            
            self.status_reporter.verbose(f"Execution plan saved to: {plan_file}")
            
        except Exception as e:
            self.status_reporter.warning(f"Failed to save execution plan: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the generation process."""
        
        metrics = self.metrics.copy()
        
        # Calculate duration
        if metrics['end_time']:
            metrics['duration'] = metrics['end_time'] - metrics['start_time']
        else:
            metrics['duration'] = time.time() - metrics['start_time']
        
        # Add cache statistics if available
        if self.cache:
            cache_stats = self.cache.get_stats()
            metrics['cache_stats'] = cache_stats
        
        # Add agent metrics if available
        if hasattr(self.routing_agent, 'get_metrics'):
            metrics['agent_metrics'] = self.routing_agent.get_metrics()
        
        return metrics 