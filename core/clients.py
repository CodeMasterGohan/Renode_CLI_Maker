"""
Client classes for LLM and Milvus database interactions.
"""

import time
import hashlib
from typing import List, Dict, Any, Optional

from .exceptions import LLMClientError, MilvusClientError
from .cache import InMemoryCache

# Import dependencies with fallbacks
try:
    import ollama
except ImportError:
    ollama = None

try:
    import openai
except ImportError:
    openai = None

try:
    from pymilvus import connections, Collection, utility
except ImportError:
    raise ImportError("pymilvus library is required. Install with: pip install pymilvus")


class LanguageModelClient:
    """A client for a large language model, supporting Ollama and OpenAI."""

    def __init__(self, llm_provider: str, model_name: str, api_key: Optional[str] = None, 
                 base_url: Optional[str] = None, max_retries: int = 3, 
                 cache: Optional[InMemoryCache] = None):
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name
        self.max_retries = max_retries
        self.cache = cache

        if self.llm_provider == "ollama":
            if not ollama:
                raise LLMClientError("Ollama library is not installed. Install with: pip install ollama")
            self.client = ollama.Client(host=base_url)
            host_display = base_url or "http://localhost:11434"
            try:
                self.client.show(self.model_name)
            except ollama.ResponseError as e:
                raise LLMClientError(f"Model '{self.model_name}' not found in Ollama at {host_display}: {e}")
                
        elif self.llm_provider == "openai":
            if not openai:
                raise LLMClientError("OpenAI library is not installed. Install with: pip install openai")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            
        else:
            raise LLMClientError("Unsupported LLM provider. Choose 'ollama' or 'openai'.")

    def _generate_cache_key(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate a cache key for the request."""
        content = f"{self.llm_provider}:{self.model_name}:{prompt}"
        if context:
            content += f":{context}"
        return hashlib.md5(content.encode()).hexdigest()

    async def generate(self, prompt: str, context: Optional[str] = None, use_cache: bool = True) -> str:
        """Generate response from the LLM."""
        
        # Check cache first
        cache_key = None
        if use_cache and self.cache:
            cache_key = self._generate_cache_key(prompt, context)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

        messages = [{"role": "system", "content": context}] if context else []
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                if self.llm_provider == "ollama":
                    response = self.client.chat(model=self.model_name, messages=messages)
                    content = response["message"]["content"]
                elif self.llm_provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name, 
                        messages=messages
                    )
                    content = response.choices[0].message.content

                # Cache the result
                if use_cache and self.cache and cache_key:
                    self.cache.set(cache_key, content)

                return content
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise LLMClientError(f"Failed to get response from {self.llm_provider.capitalize()} after {self.max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff


class MilvusClient:
    """A client to interact with the Milvus database."""

    def __init__(self, uri: str, collection_names: Dict[str, str],
                 embedding_provider: str, embedding_model_name: str,
                 embedding_api_key: Optional[str] = None,
                 embedding_base_url: Optional[str] = None,
                 cache: Optional[InMemoryCache] = None):
        self.uri = uri
        self.embedding_provider = embedding_provider.lower()
        self.embedding_model_name = embedding_model_name
        self.cache = cache

        if self.embedding_provider == "ollama":
            if not ollama:
                raise MilvusClientError("Ollama library is not installed for embeddings.")
            self.embedding_client = ollama.Client(host=embedding_base_url)
            try:
                self.embedding_client.show(self.embedding_model_name)
            except ollama.ResponseError as e:
                host_display = embedding_base_url or "http://localhost:11434"
                raise MilvusClientError(f"Embedding model '{self.embedding_model_name}' not found in Ollama at {host_display}: {e}")
                
        elif self.embedding_provider == "openai":
            if not openai:
                raise MilvusClientError("OpenAI library is not installed for embeddings.")
            self.embedding_client = openai.OpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
            
        else:
            raise MilvusClientError("Unsupported embedding provider. Choose 'ollama' or 'openai'.")

        try:
            connections.connect("default", uri=self.uri)
            self.collections = {}
            for key, name in collection_names.items():
                if not utility.has_collection(name):
                    print(f"Warning: Milvus collection '{name}' not found on the server at {self.uri}. Skipping this collection.")
                    continue
                self.collections[key] = Collection(name)
                self.collections[key].load()
                
        except Exception as e:
            raise MilvusClientError(f"Failed to connect to Milvus or load collections: {e}")

    def _generate_embedding_cache_key(self, text: str) -> str:
        """Generate a cache key for embedding requests."""
        content = f"{self.embedding_provider}:{self.embedding_model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text using the configured model."""
        
        # Check cache first
        cache_key = None
        if self.cache:
            cache_key = self._generate_embedding_cache_key(text)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result

        try:
            if self.embedding_provider == "ollama":
                response = self.embedding_client.embeddings(
                    model=self.embedding_model_name, prompt=text
                )
                embedding = response["embedding"]
            elif self.embedding_provider == "openai":
                response = self.embedding_client.embeddings.create(
                    input=[text], model=self.embedding_model_name
                )
                embedding = response.data[0].embedding

            # Cache the result
            if self.cache and cache_key:
                self.cache.set(cache_key, embedding)

            return embedding
            
        except Exception as e:
            raise MilvusClientError(f"Error generating embedding with '{self.embedding_model_name}': {e}")

    async def search(self, collection_key: str, query_text: str, limit: int = 5) -> List[str]:
        """
        Generates an embedding for the query text and performs a search
        in the specified Milvus collection.
        """
        
        if collection_key not in self.collections:
            raise MilvusClientError(f"Collection key '{collection_key}' not found")

        try:
            # Generate embedding for the query
            query_embedding = await self._generate_embedding(query_text)
            
            # Perform the search
            collection = self.collections[collection_key]
            search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
            
            results = collection.search(
                data=[query_embedding],
                anns_field="vector",
                param=search_params,
                limit=limit,
                output_fields=["text"]
            )
            
            # Extract text results
            documents = []
            for hit in results[0]:
                documents.append(hit.entity.get("text", ""))
            
            return documents
            
        except Exception as e:
            raise MilvusClientError(f"Search failed in collection '{collection_key}': {e}")
