"""
RobotFramework Application Module

This module provides the main application logic for generating RobotFramework test suites
for Renode peripherals.
"""

import json
import time
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path

from core.config import AppConfig
from core.exceptions import RenodeGeneratorError
from core.clients import LanguageModelClient, MilvusClient
from core.rf_agents import RFRoutingAgent
from core.cache import InMemoryCache
from utils.status import StatusReporter
from .rf_templates import RFTemplateManager
from .rf_validators import RFValidator


class RFGeneratorCLI:
    """Main application class for RobotFramework test generation."""
    
    def __init__(self, config: AppConfig, status_reporter: StatusReporter):
        self.config = config
        self.status_reporter = status_reporter
        self.template_manager = RFTemplateManager()
        self.validator = RFValidator()
        
        self.metrics = {
            'start_time': time.time(),
            'end_time': None,
            'iterations': 0,
            'test_level': None,
            'validation_score': 0
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
        
        # Initialize clients
        self._initialize_clients()
        
        # Initialize RF routing agent
        self.rf_routing_agent = RFRoutingAgent(
            llm_client=self.llm_client,
            milvus_client=self.milvus_client,
            max_iterations=3
        )
        
    def _initialize_clients(self):
        """Initialize LLM and Milvus clients."""
        try:
            self.llm_client = LanguageModelClient(
                llm_provider=self.config.llm.provider,
                model_name=self.config.llm.model,
                api_key=self.config.llm.api_key,
                base_url=self.config.llm.host,
                max_retries=self.config.llm.max_retries,
                cache=self.cache
            )
            
            self.milvus_client = MilvusClient(
                uri=self.config.milvus.uri,
                collection_names=self.config.milvus.collections,
                embedding_provider=self.config.embedding.provider,
                embedding_model_name=self.config.embedding.model,
                embedding_api_key=self.config.embedding.api_key,
                embedding_base_url=self.config.embedding.host,
                cache=self.cache
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize RF clients: {e}"
            raise RenodeGeneratorError(error_msg) from e
    
    def run(self, prompt: str, test_level: str = None, max_iterations: int = 3, 
            use_cache: bool = True, output_dir: str = None) -> Dict[str, Any]:
        """Run the RobotFramework test generation process."""
        
        if test_level is None:
            test_level = self.config.robotframework.test_levels[0] if self.config.robotframework.test_levels else 'integration'
        
        self.metrics['test_level'] = test_level
        
        try:
            self.rf_routing_agent.max_iterations = max_iterations
            
            task = {
                'prompt': prompt,
                'test_level': test_level,
                'use_cache': use_cache and self.config.cache.enabled
            }
            
            self.status_reporter.status_update("Executing RF test generation pipeline...")
            
            rf_code = asyncio.run(self._run_rf_generation(task))
            
            validation_result = self.validator.validate_rf_content(rf_code)
            self.metrics['validation_score'] = validation_result.get('score', 0)
            
            result = {
                'rf_code': rf_code,
                'test_level': test_level,
                'validation': validation_result,
                'peripheral_name': self._extract_peripheral_name(prompt)
            }
            
            if output_dir:
                self._save_rf_files(result, output_dir)
            
            self.metrics['end_time'] = time.time()
            self.metrics['iterations'] = self.rf_routing_agent.current_iteration
            
            self.status_reporter.success("RF test generation completed successfully!")
            
            return result
            
        except Exception as e:
            self.metrics['end_time'] = time.time()
            error_msg = f"RF test generation failed: {e}"
            self.status_reporter.error(error_msg)
            raise RenodeGeneratorError(error_msg) from e
    
    async def _run_rf_generation(self, task: Dict[str, Any]) -> str:
        """Run the actual RF generation process asynchronously."""
        
        def status_callback(message: str, iteration: int = None):
            if iteration is not None:
                self.status_reporter.progress(message, iteration, self.rf_routing_agent.max_iterations + 4)
            else:
                self.status_reporter.status_update(message)
        
        task['status_callback'] = status_callback
        result = await self.rf_routing_agent.execute(task)
        return result
    
    def _extract_peripheral_name(self, prompt: str) -> str:
        """Extract peripheral name from prompt."""
        prompt_lower = prompt.lower()
        peripherals = ['uart', 'spi', 'i2c', 'gpio', 'timer', 'adc', 'dac', 'pwm', 'dma', 'usb']
        
        for peripheral in peripherals:
            if peripheral in prompt_lower:
                return peripheral.upper()
        
        words = prompt.split()
        for word in words:
            if len(word) > 2 and word.isalpha():
                return word.capitalize()
        
        return "GenericPeripheral"
    
    def _save_rf_files(self, result: Dict[str, Any], output_dir: str):
        """Save generated RF files to the specified directory."""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        peripheral_name = result['peripheral_name'].lower()
        test_level = result['test_level']
        
        test_file = output_path / f"{peripheral_name}_{test_level}_tests.robot"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(result['rf_code'])
        
        self.status_reporter.verbose(f"RF test file saved: {test_file}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get generation metrics."""
        metrics = self.metrics.copy()
        
        if metrics['end_time']:
            metrics['duration'] = metrics['end_time'] - metrics['start_time']
        else:
            metrics['duration'] = time.time() - metrics['start_time']
        
        return metrics