"""
title: Renode Peripheral Generator Pipeline
author: James Drummond (Enhanced by AI Assistant)
date: 2025-01-27
version: 2.0
license: MIT
description: An enhanced pipeline for generating Renode peripheral code using multi-agent systems with caching, error recovery, and structured logging.
requirements: ollama, openai, pymilvus, tenacity, structlog
"""

import os
import json
import time
import hashlib
import asyncio
from typing import List, Dict, Any, Optional, Union, Generator, Iterator
from schemas import OpenAIChatMessage
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Enhanced Imports
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    print("WARNING: 'tenacity' library not installed. Retry functionality will be limited.")
    def retry(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    stop_after_attempt = wait_exponential = retry_if_exception_type = lambda *args, **kwargs: None

try:
    import structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
except ImportError:
    print("WARNING: 'structlog' library not installed. Using standard logging.")
    import logging as structlog
    structlog.get_logger = logging.getLogger

# LLM and Database Client Imports
try:
    import ollama
except ImportError:
    print("WARNING: The 'ollama' library is not installed. Ollama features will be unavailable.")
    ollama = None

try:
    import openai
except ImportError:
    print("WARNING: The 'openai' library is not installed. OpenAI features will be unavailable.")
    openai = None

try:
    from pymilvus import connections, Collection, utility
except ImportError:
    print("FATAL ERROR: The 'pymilvus' library is not installed.")
    raise ImportError("Please install the 'pymilvus' library using: pip install pymilvus")

# Enhanced Types and Data Classes
@dataclass
class CacheEntry:
    data: Any
    timestamp: float
    ttl: float
    
    def is_expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl

@dataclass
class GenerationMetrics:
    start_time: float
    end_time: Optional[float] = None
    iterations: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

# Enhanced Exceptions
class RenodeGeneratorError(Exception):
    pass

class LLMClientError(RenodeGeneratorError):
    pass

class MilvusClientError(RenodeGeneratorError):
    pass

# Enhanced Caching System
class InMemoryCache:
    def __init__(self, default_ttl: int = 3600, max_size: int = 1000):
        self.cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl
        self.max_size = max_size
        self.logger = structlog.get_logger("cache")
        
    def _cleanup_expired(self):
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items() 
            if entry.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.logger.info("cache_cleanup", expired_count=len(expired_keys))
    
    def _enforce_size_limit(self):
        if len(self.cache) > self.max_size:
            sorted_items = sorted(
                self.cache.items(), 
                key=lambda x: x[1].timestamp
            )
            remove_count = len(self.cache) - int(self.max_size * 0.8)
            for key, _ in sorted_items[:remove_count]:
                del self.cache[key]
            
            self.logger.info("cache_size_limit", removed_count=remove_count)
    
    def get(self, key: str) -> Optional[Any]:
        self._cleanup_expired()
        
        if key in self.cache:
            entry = self.cache[key]
            if not entry.is_expired():
                self.logger.debug("cache_hit", key=key)
                return entry.data
            else:
                del self.cache[key]
        
        self.logger.debug("cache_miss", key=key)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if ttl is None:
            ttl = self.default_ttl
            
        self.cache[key] = CacheEntry(
            data=value,
            timestamp=time.time(),
            ttl=ttl
        )
        
        self._enforce_size_limit()
        self.logger.debug("cache_set", key=key, ttl=ttl)
    
    def invalidate(self, pattern: str = None) -> None:
        if pattern is None:
            self.cache.clear()
            self.logger.info("cache_invalidated_all")
        else:
            removed_keys = [key for key in self.cache.keys() if pattern in key]
            for key in removed_keys:
                del self.cache[key]
            self.logger.info("cache_invalidated_pattern", pattern=pattern, count=len(removed_keys))
    
    def get_stats(self) -> Dict[str, Any]:
        self._cleanup_expired()
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "oldest_entry": min(
                (entry.timestamp for entry in self.cache.values()), 
                default=time.time()
            ),
            "newest_entry": max(
                (entry.timestamp for entry in self.cache.values()), 
                default=time.time()
            )
        }

# Enhanced Utility Functions
def generate_cache_key(*args, **kwargs) -> str:
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())
    }
    key_string = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_string.encode()).hexdigest()[:16]

# Enhanced Client Implementations
class LanguageModelClient:
    def __init__(
        self, 
        llm_provider: str, 
        model_name: str, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None,
        max_retries: int = 3,
        cache: Optional[InMemoryCache] = None
    ):
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name
        self.max_retries = max_retries
        self.cache = cache or InMemoryCache()
        self.logger = structlog.get_logger("llm_client")
        
        # Track metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.cache_hits = 0

        if self.llm_provider == "ollama":
            if not ollama:
                raise LLMClientError("Ollama library is not installed.")
            self.client = ollama.Client(host=base_url)
            host_display = base_url or "http://ollama:11434"
            self.logger.info(
                "llm_client_initialized",
                provider="ollama",
                model=self.model_name,
                host=host_display
            )
            try:
                self.client.show(self.model_name)
                self.logger.info("llm_model_verified", model=self.model_name)
            except ollama.ResponseError as e:
                error_msg = f"Model '{self.model_name}' not found in Ollama at {host_display}"
                self.logger.error("llm_model_verification_failed", error=error_msg)
                raise LLMClientError(error_msg) from e
                
        elif self.llm_provider == "openai":
            if not openai:
                raise LLMClientError("OpenAI library is not installed.")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            self.logger.info(
                "llm_client_initialized",
                provider="openai",
                model=self.model_name
            )
        else:
            raise ValueError("Unsupported LLM provider. Choose 'ollama' or 'openai'.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError, ConnectionError)) if openai else retry_if_exception_type(ConnectionError)
    )
    async def _make_llm_request(self, messages: List[Dict[str, str]]) -> str:
        try:
            if self.llm_provider == "ollama":
                response = self.client.chat(model=self.model_name, messages=messages)
                return response["message"]["content"]
            elif self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name, 
                    messages=messages
                )
                return response.choices[0].message.content
        except Exception as e:
            self.logger.error(
                "llm_request_failed",
                provider=self.llm_provider,
                model=self.model_name,
                error=str(e)
            )
            raise LLMClientError(f"Failed to get response from {self.llm_provider}: {e}") from e

    async def generate(self, prompt: str, context: Optional[str] = None, use_cache: bool = True) -> str:
        self.total_requests += 1
        
        # Build messages
        messages = []
        if context:
            messages.append({"role": "system", "content": context})
        messages.append({"role": "user", "content": prompt})
        
        # Check cache first
        if use_cache:
            cache_key = generate_cache_key(
                provider=self.llm_provider,
                model=self.model_name,
                messages=messages
            )
            
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                self.cache_hits += 1
                self.logger.debug("llm_cache_hit", cache_key=cache_key)
                return cached_response
        
        # Make request
        start_time = time.time()
        self.logger.info(
            "llm_request_started",
            provider=self.llm_provider,
            model=self.model_name,
            prompt_length=len(prompt)
        )
        
        try:
            content = await self._make_llm_request(messages)
            duration = time.time() - start_time
            
            self.successful_requests += 1
            self.logger.info(
                "llm_request_completed",
                provider=self.llm_provider,
                model=self.model_name,
                duration=duration,
                response_length=len(content)
            )
            
            # Cache the response
            if use_cache:
                self.cache.set(cache_key, content)
            
            return content
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "llm_request_error",
                provider=self.llm_provider,
                model=self.model_name,
                duration=duration,
                error=str(e)
            )
            raise

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "cache_hits": self.cache_hits,
            "success_rate": self.successful_requests / max(self.total_requests, 1),
            "cache_hit_rate": self.cache_hits / max(self.total_requests, 1)
        }

class MilvusClient:
    def __init__(
        self,
        uri: str,
        collection_names: Dict[str, str],
        embedding_provider: str,
        embedding_model_name: str,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        cache: Optional[InMemoryCache] = None
    ):
        self.uri = uri
        self.embedding_provider = embedding_provider.lower()
        self.embedding_model_name = embedding_model_name
        self.cache = cache or InMemoryCache(default_ttl=1800)
        self.logger = structlog.get_logger("milvus_client")

        # Initialize embedding client
        if self.embedding_provider == "ollama":
            if not ollama:
                raise MilvusClientError("Ollama library is not installed for embeddings.")
            self.embedding_client = ollama.Client(host=embedding_base_url)
            host_display = embedding_base_url or "http://ollama:11434"
            try:
                self.embedding_client.show(self.embedding_model_name)
                self.logger.info(
                    "embedding_model_verified", 
                    model=self.embedding_model_name,
                    host=host_display
                )
            except ollama.ResponseError as e:
                error_msg = f"Embedding model '{self.embedding_model_name}' not found"
                self.logger.error("embedding_model_verification_failed", error=error_msg)
                raise MilvusClientError(error_msg) from e
                
        elif self.embedding_provider == "openai":
            if not openai:
                raise MilvusClientError("OpenAI library is not installed for embeddings.")
            self.embedding_client = openai.OpenAI(
                api_key=embedding_api_key, 
                base_url=embedding_base_url
            )
            self.logger.info("embedding_client_initialized", provider="openai")
        else:
            raise ValueError("Unsupported embedding provider. Choose 'ollama' or 'openai'.")

        # Connect to Milvus
        try:
            connections.connect("default", uri=self.uri)
            self.logger.info("milvus_connected", uri=self.uri)

            self.collections = {}
            for key, name in collection_names.items():
                if not utility.has_collection(name):
                    raise MilvusClientError(
                        f"Milvus collection '{name}' not found on the server at {self.uri}."
                    )
                self.collections[key] = Collection(name)
                self.collections[key].load()
                
            self.logger.info("milvus_collections_loaded", collections=list(collection_names.keys()))

        except Exception as e:
            self.logger.error("milvus_connection_failed", error=str(e))
            raise MilvusClientError(f"Failed to connect to Milvus: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((openai.RateLimitError, ConnectionError)) if openai else retry_if_exception_type(ConnectionError)
    )
    async def _generate_embedding(self, text: str) -> List[float]:
        try:
            if self.embedding_provider == "ollama":
                response = self.embedding_client.embeddings(
                    model=self.embedding_model_name, prompt=text
                )
                return response["embedding"]
            elif self.embedding_provider == "openai":
                response = self.embedding_client.embeddings.create(
                    input=[text], model=self.embedding_model_name
                )
                return response.data[0].embedding
        except Exception as e:
            self.logger.error(
                "embedding_generation_failed",
                provider=self.embedding_provider,
                model=self.embedding_model_name,
                error=str(e)
            )
            raise MilvusClientError(f"Failed to generate embedding: {e}") from e

    async def search(self, collection_key: str, query_text: str, limit: int = 5) -> List[str]:
        # Check cache first
        cache_key = generate_cache_key(
            collection=collection_key,
            query=query_text,
            limit=limit,
            model=self.embedding_model_name
        )
        
        cached_results = self.cache.get(cache_key)
        if cached_results is not None:
            self.logger.debug("search_cache_hit", collection=collection_key)
            return cached_results

        start_time = time.time()
        self.logger.info(
            "search_started",
            collection=collection_key,
            query_length=len(query_text),
            limit=limit
        )

        try:
            # Generate embedding
            query_vector = await self._generate_embedding(query_text)
            
            # Perform search
            if collection_key not in self.collections:
                raise MilvusClientError(f"Collection '{collection_key}' not available")
                
            results = self.collections[collection_key].search(
                data=[query_vector],
                anns_field="vector",
                param={"metric_type": "IP", "params": {"nprobe": 10}},
                limit=limit,
                output_fields=["text"],
            )
            
            search_results = [hit.entity.get("text") for hit in results[0]]
            duration = time.time() - start_time
            
            self.logger.info(
                "search_completed",
                collection=collection_key,
                results_count=len(search_results),
                duration=duration
            )
            
            # Cache results
            self.cache.set(cache_key, search_results)
            
            return search_results
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(
                "search_failed",
                collection=collection_key,
                duration=duration,
                error=str(e)
            )
            raise MilvusClientError(f"Search failed: {e}") from e

# Enhanced Agent Definitions
class BaseAgent:
    def __init__(
        self,
        llm_client: LanguageModelClient,
        milvus_client: Optional[MilvusClient] = None,
    ):
        self.llm_client = llm_client
        self.milvus_client = milvus_client
        self.logger = structlog.get_logger(self.__class__.__name__)
        self.execution_count = 0
        self.total_duration = 0.0

    async def execute(self, task: Dict[str, Any]) -> Any:
        self.execution_count += 1
        start_time = time.time()
        
        try:
            result = await self._execute_impl(task)
            duration = time.time() - start_time
            self.total_duration += duration
            
            self.logger.info(
                "agent_execution_completed",
                task_type=task.get("type", "unknown"),
                duration=duration,
                success=True
            )
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self.total_duration += duration
            
            self.logger.error(
                "agent_execution_failed",
                task_type=task.get("type", "unknown"),
                duration=duration,
                error=str(e)
            )
            raise

    async def _execute_impl(self, task: Dict[str, Any]) -> Any:
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "execution_count": self.execution_count,
            "total_duration": self.total_duration,
            "average_duration": self.total_duration / max(self.execution_count, 1)
        }

class PlanningAgent(BaseAgent):
    async def _execute_impl(self, task: Dict[str, Any]) -> List[str]:
        self.logger.info("planning_started", prompt_length=len(task.get('prompt', '')))
            
        prompt = (
            f"Create a detailed, step-by-step plan to generate a Renode peripheral "
            f"from the following prompt. Respond ONLY with a numbered list of specific, "
            f"actionable steps. Each step should be clear and focused.\n\n"
            f"Prompt: '{task['prompt']}'"
        )
        
        plan_str = await self.llm_client.generate(prompt)
        plan = [step.strip() for step in plan_str.split("\n") if step.strip() and any(c.isdigit() for c in step)]
        
        self.logger.info("planning_completed", steps_count=len(plan))
        return plan

class CodingAgent(BaseAgent):
    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        sub_task = task.get('sub_task', 'Generate peripheral code')
        self.logger.info("coding_started", sub_task=sub_task)
            
        context = ""
        if self.milvus_client:
            try:
                example_results = await self.milvus_client.search(
                    "examples", f"{task['prompt']} {sub_task}", limit=3
                )
                if example_results:
                    context += (
                        "Reference Examples:\n" + "="*50 + "\n"
                        + "\n" + "-"*30 + "\n".join(example_results)
                        + "\n" + "="*50 + "\n\n"
                    )
            except Exception as e:
                self.logger.warning("failed_to_fetch_examples", error=str(e))
        
        prompt = (
            f"You are a C# expert specializing in Renode peripheral development. "
            f"Generate high-quality, well-documented C# code for the Renode simulator.\n\n"
            f"Request: '{task['prompt']}'\n"
            f"Current Task: '{sub_task}'\n\n"
            f"Existing Code:\n```csharp\n{task.get('code', '// No code yet.')}\n```\n\n"
            f"Return ONLY the complete, updated C# code for the entire peripheral. "
            f"Include proper error handling, logging, and documentation."
        )
        
        code = await self.llm_client.generate(prompt, context=context)
        
        self.logger.info("coding_completed", code_length=len(code))
        return code

class ReviewingAgent(BaseAgent):
    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        self.logger.info("reviewing_started", code_length=len(task.get('code', '')))
            
        prompt = (
            f"Review the following C# code for a Renode peripheral. Provide detailed "
            f"feedback focusing on:\n"
            f"1. Code quality and best practices\n"
            f"2. Renode-specific implementation patterns\n"
            f"3. Error handling and robustness\n"
            f"4. Documentation and readability\n"
            f"5. Performance considerations\n\n"
            f"If the code is excellent, respond with 'No issues found.'\n"
            f"Otherwise, provide specific, actionable feedback as bullet points.\n\n"
            f"Code:\n```csharp\n{task['code']}\n```"
        )
        
        feedback = await self.llm_client.generate(prompt)
        
        self.logger.info("reviewing_completed", feedback_length=len(feedback))
        return feedback

class AccuracyAgent(BaseAgent):
    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        self.logger.info("accuracy_check_started")
            
        context = ""
        if self.milvus_client:
            try:
                manual_results = await self.milvus_client.search(
                    "manual", task["prompt"], limit=3
                )
                if manual_results:
                    context += (
                        "Reference Manual Sections:\n" + "="*50 + "\n"
                        + "\n" + "-"*30 + "\n".join(manual_results)
                        + "\n" + "="*50 + "\n\n"
                    )
            except Exception as e:
                self.logger.warning("failed_to_fetch_manual", error=str(e))
        
        prompt = (
            f"Verify the technical accuracy of this C# Renode peripheral code against "
            f"the provided manual context. Check:\n"
            f"1. Register addresses and bit definitions\n"
            f"2. Hardware behavior implementation\n"
            f"3. Protocol compliance\n"
            f"4. Renode framework usage\n\n"
            f"If everything is accurate, respond 'Accuracy check passed.'\n"
            f"Otherwise, note specific inaccuracies and provide corrections.\n\n"
            f"Code:\n```csharp\n{task['code']}\n```"
        )
        
        report = await self.llm_client.generate(prompt, context=context)
        
        self.logger.info("accuracy_check_completed", report_length=len(report))
        return report

class RoutingAgent(BaseAgent):
    def __init__(
        self,
        llm_client: LanguageModelClient,
        milvus_client: MilvusClient,
        max_iterations: int = 3,
    ):
        super().__init__(llm_client, milvus_client)
        self.max_iterations = max_iterations
        self.agents = {
            "planning": PlanningAgent(llm_client),
            "coding": CodingAgent(llm_client, milvus_client),
            "reviewing": ReviewingAgent(llm_client),
            "accuracy": AccuracyAgent(llm_client, milvus_client),
        }
        self.metrics = GenerationMetrics(start_time=time.time())

    async def _execute_impl(self, task: Dict[str, Any]) -> str:
        prompt = task["prompt"]
        
        # Planning phase
        try:
            plan = await self.agents["planning"].execute({"prompt": prompt})
        except Exception as e:
            self.logger.error("planning_failed", error=str(e))
            plan = ["Implement the peripheral based on the prompt."]
        
        code = ""
        best_code = ""
        best_iteration = 0
        
        for i in range(self.max_iterations):
            self.metrics.iterations += 1
            iteration_start = time.time()
            
            self.logger.info("iteration_started", iteration=i+1, max_iterations=self.max_iterations)
            
            try:
                current_task = plan[0] if plan else "Implement the peripheral based on the prompt."
                
                # Coding phase
                code = await self.agents["coding"].execute({
                    "sub_task": current_task, 
                    "prompt": prompt, 
                    "code": code
                })
                
                # Review phase
                review_feedback = await self.agents["reviewing"].execute({"code": code})
                
                # Accuracy phase
                accuracy_report = await self.agents["accuracy"].execute({
                    "code": code, 
                    "prompt": prompt
                })
                
                # Check if we're done
                review_passed = "no issues found" in review_feedback.lower()
                accuracy_passed = "accuracy check passed" in accuracy_report.lower()
                
                if review_passed and accuracy_passed:
                    self.logger.info(
                        "generation_completed_early",
                        iteration=i + 1,
                        duration=time.time() - iteration_start
                    )
                    best_code = code
                    break
                
                # If this iteration is better than previous ones, save it
                if not best_code or (review_passed or accuracy_passed):
                    best_code = code
                    best_iteration = i + 1
                
                # Generate refinement task
                if i < self.max_iterations - 1:
                    refinement_prompt = (
                        f"Given the request '{prompt}', code review feedback '{review_feedback}', "
                        f"and accuracy report '{accuracy_report}', create a concise, single-line "
                        f"task to improve the code. Focus on the most critical issues. "
                        f"If no improvements are needed, respond with 'DONE'."
                    )
                    
                    next_step = await self.llm_client.generate(refinement_prompt)
                    
                    if "DONE" in next_step.upper():
                        self.logger.info("refinement_completed", iteration=i + 1)
                        break
                    
                    plan = [next_step]
                
            except Exception as e:
                self.metrics.errors.append(f"Iteration {i+1}: {str(e)}")
                self.logger.error(
                    "iteration_failed",
                    iteration=i + 1,
                    error=str(e)
                )
                
                # If we have some code from previous iterations, use it
                if best_code:
                    break
                    
                # Otherwise, try to continue with simplified approach
                if i == self.max_iterations - 1:
                    raise RenodeGeneratorError(f"All iterations failed. Last error: {e}")
        
        self.metrics.end_time = time.time()
        
        # Log final metrics
        self.logger.info(
            "generation_completed",
            total_duration=self.metrics.duration,
            iterations=self.metrics.iterations,
            best_iteration=best_iteration,
            error_count=len(self.metrics.errors)
        )
        
        return best_code or code

    def get_generation_metrics(self) -> Dict[str, Any]:
        base_metrics = self.get_metrics()
        base_metrics.update(asdict(self.metrics))
        
        # Add agent metrics
        base_metrics["agent_metrics"] = {
            name: agent.get_metrics() 
            for name, agent in self.agents.items()
        }
        
        return base_metrics

# Pipeline Class
class Pipeline:
    def __init__(self):
        # Configuration from environment variables
        self.config = {
            # LLM Configuration
            "llm_provider": os.getenv("LLM_PROVIDER", "ollama"),
            "model_name": os.getenv("MODEL_NAME", "llama3"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "openai_base_url": os.getenv("OPENAI_BASE_URL"),
            "ollama_host": os.getenv("OLLAMA_HOST", "http://dso-wp-kasm2.sern.mil:11434"),
            
            # Embedding Configuration
            "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "ollama"),
            "embedding_model_name": os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text"),
            "embedding_api_key": os.getenv("EMBEDDING_API_KEY"),
            "embedding_base_url": os.getenv("EMBEDDING_BASE_URL", "http://dso-wp-kasm2.sern.mil:11434"),
            
            # Database Configuration
            "milvus_uri": os.getenv("MILVUS_URI", "milvus:19530"),
            
            # Enhanced Configuration
            "max_iterations": int(os.getenv("MAX_ITERATIONS", "3")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "cache_ttl": int(os.getenv("CACHE_TTL", "3600")),
            "cache_size": int(os.getenv("CACHE_SIZE", "1000")),
            "enable_metrics": os.getenv("ENABLE_METRICS", "true").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
        }
        
        # Initialize components
        self.cache = None
        self.llm_client = None
        self.milvus_client = None
        self.router = None
        self.logger = structlog.get_logger("renode_pipeline")
        
        # Set logging level
        logging.getLogger().setLevel(getattr(logging, self.config["log_level"].upper(), logging.INFO))

    async def on_startup(self):
        """Initialize all components on startup."""
        try:
            self.logger.info("pipeline_startup_started")
            
            # Initialize cache
            self.cache = InMemoryCache(
                default_ttl=self.config["cache_ttl"],
                max_size=self.config["cache_size"]
            )
            
            # Initialize LLM client
            self.llm_client = LanguageModelClient(
                llm_provider=self.config["llm_provider"],
                model_name=self.config["model_name"],
                api_key=self.config["openai_api_key"],
                base_url=(
                    self.config["openai_base_url"] 
                    if self.config["llm_provider"] == 'openai' 
                    else self.config["ollama_host"]
                ),
                max_retries=self.config["max_retries"],
                cache=self.cache
            )
            
            # Initialize Milvus client
            self.milvus_client = MilvusClient(
                uri=self.config["milvus_uri"],
                collection_names={
                    "manual": "pacer_documents",
                    "examples": "pacer_renode_peripheral_examples",
                },
                embedding_provider=self.config["embedding_provider"],
                embedding_model_name=self.config["embedding_model_name"],
                embedding_api_key=self.config["embedding_api_key"],
                embedding_base_url=(
                    self.config["embedding_base_url"] 
                    if self.config["embedding_provider"] == 'openai' 
                    else self.config["ollama_host"]
                ),
                cache=self.cache
            )
            
            # Initialize routing agent
            self.router = RoutingAgent(
                llm_client=self.llm_client,
                milvus_client=self.milvus_client,
                max_iterations=self.config["max_iterations"],
            )
            
            self.logger.info("pipeline_startup_completed")
            
        except Exception as e:
            self.logger.error("pipeline_startup_failed", error=str(e))
            raise

    async def on_shutdown(self):
        """Cleanup on shutdown."""
        self.logger.info("pipeline_shutdown_started")
        
        # Clear cache
        if self.cache:
            self.cache.invalidate()
        
        # Reset clients
        self.llm_client = None
        self.milvus_client = None
        self.router = None
        self.cache = None
        
        self.logger.info("pipeline_shutdown_completed")

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        """Main pipeline method for generating Renode peripheral code."""
        start_time = time.time()
        
        try:
            # Log request start
            self.logger.info(
                "pipeline_request_started",
                user_message_length=len(user_message),
                model_id=model_id,
                messages_count=len(messages)
            )
            
            # Validate input
            if not isinstance(user_message, str) or not user_message.strip():
                error_msg = "Prompt is empty or invalid."
                self.logger.error("invalid_input", error=error_msg)
                return f"Error: {error_msg}"
            
            # Check if components are initialized
            if not all([self.llm_client, self.milvus_client, self.router]):
                error_msg = "Pipeline components not properly initialized. Please restart the service."
                self.logger.error("components_not_initialized", error=error_msg)
                return f"Error: {error_msg}"
            
            # Execute generation using asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                final_code = loop.run_until_complete(
                    self.router.execute({"prompt": user_message})
                )
            finally:
                loop.close()
            
            # Calculate metrics
            total_duration = time.time() - start_time
            
            # Log completion with metrics
            if self.config["enable_metrics"]:
                generation_metrics = self.router.get_generation_metrics()
                llm_metrics = self.llm_client.get_metrics()
                cache_stats = self.cache.get_stats()
                
                self.logger.info(
                    "pipeline_request_completed",
                    total_duration=total_duration,
                    generation_metrics=generation_metrics,
                    llm_metrics=llm_metrics,
                    cache_stats=cache_stats
                )
                
                # Include metrics in response
                metrics_summary = (
                    f"\n\n<!-- Generation Metrics:\n"
                    f"Duration: {total_duration:.1f}s\n"
                    f"Iterations: {generation_metrics['iterations']}\n"
                    f"Cache hit rate: {llm_metrics['cache_hit_rate']:.1%}\n"
                    f"Success rate: {llm_metrics['success_rate']:.1%}\n"
                    f"-->"
                )
            else:
                metrics_summary = ""
                self.logger.info("pipeline_request_completed", total_duration=total_duration)
            
            # Clean and format the code
            cleaned_code = final_code.strip()
            if cleaned_code.startswith('```'):
                # Remove existing code blocks
                lines = cleaned_code.split('\n')
                if lines[0].startswith('```'):
                    lines = lines[1:]
                if lines and lines[-1].strip() == '```':
                    lines = lines[:-1]
                cleaned_code = '\n'.join(lines)
            
            # Remove any remaining language identifiers
            cleaned_code = cleaned_code.replace('csharp', '').strip()
            
            return f"```csharp\n{cleaned_code}\n```{metrics_summary}"
            
        except RenodeGeneratorError as e:
            # Handle known application errors
            error_msg = f"Generation failed: {str(e)}"
            self.logger.error(
                "pipeline_request_failed",
                error_type="RenodeGeneratorError",
                error=error_msg,
                duration=time.time() - start_time
            )
            return f"Error: {error_msg}"
            
        except Exception as e:
            # Handle unexpected errors
            error_msg = f"An unexpected error occurred: {str(e)}"
            self.logger.error(
                "pipeline_request_error",
                error_type=type(e).__name__,
                error=error_msg,
                duration=time.time() - start_time
            )
            return f"Error: {error_msg}"
