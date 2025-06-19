"""
title: Renode Peripheral Generator
author: James Drummond
required_open_webui_version: 0.5.0
requirements: ollama, openai, pymilvus
version: 1.7
licence: MIT
"""

import os
from typing import List, Dict, Any, Optional, Callable, Awaitable, AsyncGenerator
from pydantic import BaseModel, Field
from fastapi import Request

# --- LLM and Database Client Imports ---

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
    raise ImportError(
        "Please install the 'pymilvus' library using: pip install pymilvus"
    )


# --- OpenWebUI Helper Functions ---

EmitterType = Optional[Callable[[dict], Awaitable[None]]]


def get_send_status(
    event_emitter: EmitterType,
) -> Callable[[str, bool], Awaitable[None]]:
    """Returns an async function to send status updates to the frontend."""

    async def send_status(status_message: str, done: bool):
        if event_emitter is None:
            return
        await event_emitter(
            {
                "type": "status",
                "data": {"description": status_message, "done": done},
            }
        )

    return send_status


# --- Client Implementations ---


class LanguageModelClient:
    """A client for a large language model, supporting Ollama and OpenAI."""

    def __init__(self, llm_provider: str, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.llm_provider = llm_provider.lower()
        self.model_name = model_name

        if self.llm_provider == "ollama":
            if not ollama:
                raise ImportError("Ollama library is not installed.")
            self.client = ollama.Client(host=base_url)
            host_display = base_url or "http://localhost:11434"
            print(
                f"Initialized Ollama LanguageModelClient with model: {self.model_name} on host: {host_display}"
            )
            try:
                self.client.show(self.model_name)
                print(
                    f"Successfully verified that model '{self.model_name}' is available on Ollama host."
                )
            except ollama.ResponseError as e:
                print(
                    f"Error: Model '{self.model_name}' not found in Ollama at {host_display}."
                )
                raise e
        elif self.llm_provider == "openai":
            if not openai:
                raise ImportError("OpenAI library is not installed.")
            self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            print(
                f"Initialized OpenAI LanguageModelClient with model: {self.model_name}"
            )
        else:
            raise ValueError("Unsupported LLM provider. Choose 'ollama' or 'openai'.")

    def generate(self, prompt: str, context: Optional[str] = None) -> str:
        messages = [{"role": "system", "content": context}] if context else []
        messages.append({"role": "user", "content": prompt})
        
        print(f"--- {self.llm_provider.capitalize()} Generating (Model: {self.model_name}) ---")
        
        try:
            if self.llm_provider == "ollama":
                response = self.client.chat(model=self.model_name, messages=messages)
                content = response["message"]["content"]
            elif self.llm_provider == "openai":
                response = self.client.chat.completions.create(model=self.model_name, messages=messages)
                content = response.choices[0].message.content

            print(f"--- {self.llm_provider.capitalize()} Response ---\n{content}\n--------------------")
            return content
        except Exception as e:
            print(f"An error occurred while communicating with {self.llm_provider.capitalize()}: {e}")
            return f"Error: Could not get a response from {self.llm_provider.capitalize()}. {e}"


class MilvusClient:
    """A client to interact with the Milvus database."""

    def __init__(
        self,
        uri: str,
        collection_names: Dict[str, str],
        embedding_provider: str,
        embedding_model_name: str,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        self.uri = uri
        self.embedding_provider = embedding_provider.lower()
        self.embedding_model_name = embedding_model_name

        if self.embedding_provider == "ollama":
            if not ollama:
                raise ImportError("Ollama library is not installed for embeddings.")
            self.embedding_client = ollama.Client(host=embedding_base_url)
            host_display = embedding_base_url or "http://localhost:11434"
            try:
                 self.embedding_client.show(self.embedding_model_name)
                 print(
                    f"Successfully verified embedding model '{self.embedding_model_name}' on Ollama host {host_display}."
                 )
            except ollama.ResponseError as e:
                 print(
                    f"Error: Embedding model '{self.embedding_model_name}' not found in Ollama at {host_display}."
                 )
                 raise e
        elif self.embedding_provider == "openai":
            if not openai:
                raise ImportError("OpenAI library is not installed for embeddings.")
            self.embedding_client = openai.OpenAI(api_key=embedding_api_key, base_url=embedding_base_url)
            print(f"Initialized OpenAI for embeddings with model: {self.embedding_model_name}")
        else:
            raise ValueError("Unsupported embedding provider. Choose 'ollama' or 'openai'.")

        try:
            connections.connect("default", uri=self.uri)
            print(f"Successfully connected to Milvus at {self.uri}")

            self.collections = {}
            for key, name in collection_names.items():
                if not utility.has_collection(name):
                    raise ConnectionError(
                        f"Milvus collection '{name}' not found on the server at {self.uri}."
                    )
                self.collections[key] = Collection(name)
                self.collections[key].load()
            print("Successfully verified and loaded all required Milvus collections.")

        except Exception as e:
            print(f"FATAL: Failed to connect to Milvus or load collections. Error: {e}")
            raise e

    def _generate_embedding(self, text: str) -> List[float]:
        """Generates an embedding for the given text using the configured model."""
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
            print(f"Error generating embedding with '{self.embedding_model_name}': {e}")
            raise

    def search(self, collection_key: str, query_text: str, limit: int = 5) -> List[str]:
        """
        Generates an embedding for the query text and performs a search
        in the specified Milvus collection.
        """
        print(
            f"Generating embedding for search in '{collection_key}' with model '{self.embedding_model_name}'..."
        )
        query_vector = self._generate_embedding(query_text)

        print(f"Searching collection '{collection_key}'...")
        results = self.collections[collection_key].search(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["text"],
        )
        print("Search complete.")
        return [hit.entity.get("text") for hit in results[0]]


# --- Agent Definitions (No changes needed here) ---

class BaseAgent:
    def __init__(
        self,
        llm_client: LanguageModelClient,
        milvus_client: Optional[MilvusClient] = None,
        send_status: Optional[Callable] = None,
    ):
        self.llm_client = llm_client
        self.milvus_client = milvus_client
        self.send_status = send_status

    def execute(self, task: Dict[str, Any]) -> Any:
        raise NotImplementedError


class PlanningAgent(BaseAgent):
    async def execute(self, task: Dict[str, Any]) -> List[str]:
        if self.send_status:
            await self.send_status("Planning...", False)
        prompt = f"Create a detailed, step-by-step plan to generate a Renode peripheral from the following prompt. Respond ONLY with the numbered list. Prompt: '{task['prompt']}'"
        plan_str = self.llm_client.generate(prompt)
        plan = [step.strip() for step in plan_str.split("\n") if step.strip()]
        if self.send_status:
            await self.send_status("Planning complete.", True)
        return plan


class CodingAgent(BaseAgent):
    async def execute(self, task: Dict[str, Any]) -> str:
        if self.send_status:
            await self.send_status(f"Coding: {task['sub_task']}...", False)
        context = ""
        if self.milvus_client:
            example_results = self.milvus_client.search(
                "examples", f"{task['prompt']} {task['sub_task']}"
            )
            if example_results:
                context += (
                    "Reference Examples:\n---\n"
                    + "\n---\n".join(example_results)
                    + "\n---\n"
                )
        prompt = f"You are a C# expert specializing in Renode. Your task: generate C# code. Request: '{task['prompt']}'. Current Task: '{task['sub_task']}'. Existing Code:\n```csharp\n{task.get('code', '// No code yet.')}\n```\nReturn ONLY the complete, updated C# code for the entire peripheral."
        code = self.llm_client.generate(prompt, context=context)
        if self.send_status:
            await self.send_status(f"Coding: {task['sub_task']} complete.", True)
        return code


class ReviewingAgent(BaseAgent):
    async def execute(self, task: Dict[str, Any]) -> str:
        if self.send_status:
            await self.send_status("Reviewing code...", False)
        prompt = f"Review the following C# code for a Renode peripheral. Provide concise feedback as bullet points. If excellent, respond with 'No issues found.'.\n\nCode:\n```csharp\n{task['code']}\n```"
        feedback = self.llm_client.generate(prompt)
        if self.send_status:
            await self.send_status("Review complete.", True)
        return feedback


class AccuracyAgent(BaseAgent):
    async def execute(self, task: Dict[str, Any]) -> str:
        if self.send_status:
            await self.send_status("Verifying accuracy...", False)
        context = ""
        if self.milvus_client:
            manual_results = self.milvus_client.search(
                "manual", task["prompt"], limit=3
            )
            if manual_results:
                context += (
                    "Reference Manual Sections:\n---\n"
                    + "\n---\n".join(manual_results)
                    + "\n---\n"
                )
        prompt = f"Verify the C# code against the manual context. Check register addresses, bit definitions, and behavior. Note inaccuracies. If correct, respond 'Accuracy check passed.'.\n\nCode:\n```csharp\n{task['code']}\n```"
        report = self.llm_client.generate(prompt, context=context)
        if self.send_status:
            await self.send_status("Verification complete.", True)
        return report


class RoutingAgent(BaseAgent):
    def __init__(
        self,
        llm_client: LanguageModelClient,
        milvus_client: MilvusClient,
        max_iterations: int = 3,
        send_status: Optional[Callable] = None,
    ):
        super().__init__(llm_client, milvus_client, send_status)
        self.max_iterations = max_iterations
        self.agents = {
            "planning": PlanningAgent(llm_client, send_status=send_status),
            "coding": CodingAgent(llm_client, milvus_client, send_status=send_status),
            "reviewing": ReviewingAgent(llm_client, send_status=send_status),
            "accuracy": AccuracyAgent(
                llm_client, milvus_client, send_status=send_status
            ),
        }

    async def execute(self, task: Dict[str, Any]) -> str:
        prompt = task["prompt"]
        plan = await self.agents["planning"].execute({"prompt": prompt})
        code = ""
        for i in range(self.max_iterations):
            if self.send_status:
                await self.send_status(
                    f"Starting iteration {i + 1}/{self.max_iterations}...", False
                )
            current_task = (
                plan[0] if plan else "Implement the peripheral based on the prompt."
            )
            code = await self.agents["coding"].execute(
                {"sub_task": current_task, "prompt": prompt, "code": code}
            )
            review_feedback = await self.agents["reviewing"].execute({"code": code})
            accuracy_report = await self.agents["accuracy"].execute(
                {"code": code, "prompt": prompt}
            )
            if (
                "no issues found" in review_feedback.lower()
                and "accuracy check passed" in accuracy_report.lower()
            ):
                break
            refinement_prompt = f"Given the request '{prompt}', code review '{review_feedback}', and accuracy report '{accuracy_report}', create a new, single-line task to improve the code. If it's done, respond with 'DONE'."
            next_step = self.llm_client.generate(refinement_prompt)
            if "DONE" in next_step.upper():
                break
            plan = [next_step]
        return code


# --- OpenWebUI Pipe Class ---


class Pipe:
    """
    This class is the main entry point for the OpenWebUI function.
    It orchestrates the multi-agent system to generate Renode peripheral code.
    """

    class Valves(BaseModel):
        """Configuration settings for the pipe, loaded from environment variables."""

        LLM_PROVIDER: str = Field(
            default="ollama",
            description="The provider for the language model. Can be 'ollama' or 'openai'.",
        )
        OLLAMA_HOST: Optional[str] = Field(
            default="http://localhost:11434",
            description="URL of the Ollama server.",
        )
        MODEL_NAME: str = Field(
            default="llama3",
            description="The model to use for the agents (e.g., 'llama3', 'gpt-4').",
        )
        OPENAI_API_KEY: Optional[str] = Field(
            default=None,
            description="API key for OpenAI.",
        )
        OPENAI_BASE_URL: Optional[str] = Field(
            default=None,
            description="Base URL for OpenAI-compatible API.",
        )
        
        EMBEDDING_PROVIDER: str = Field(
            default="ollama",
            description="The provider for the embedding model. Can be 'ollama' or 'openai'.",
        )
        EMBEDDING_MODEL_NAME: str = Field(
            default="nomic-embed-text",
            description="The model to use for generating text embeddings (e.g., 'nomic-embed-text', 'text-embedding-ada-002').",
        )
        EMBEDDING_API_KEY: Optional[str] = Field(
            default=None,
            description="API key for the embedding provider (if using OpenAI).",
        )
        EMBEDDING_BASE_URL: Optional[str] = Field(
            default="http://localhost:11434",
            description="Base URL for the embedding provider's API.",
        )

        MILVUS_URI: str = Field(
            default="localhost:19530", description="URI for the Milvus database instance."
        )
        MAX_ITERATIONS: int = Field(
            default=3,
            description="Maximum number of refinement loops for the agentic workflow.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves(
            **{k: os.getenv(k, v.default) for k, v in self.Valves.model_fields.items()}
        )
        # For local development, you might want to load from a .env file
        # from dotenv import load_dotenv
        # load_dotenv()

    async def pipe(
        self,
        body: dict,
        __request__: Request,
        __user__: Optional[dict],
        __task__: Optional[str],
        __tools__: Optional[dict],
        __event_emitter__: EmitterType,
    ) -> AsyncGenerator[str, None]:
        """
        This is the main method called by OpenWebUI.
        It receives the user's request and streams back the generated code.
        """
        send_status = get_send_status(__event_emitter__)

        try:
            prompt = body["messages"][-1]["content"]
            if not isinstance(prompt, str) or not prompt.strip():
                yield "Error: Prompt is empty or invalid."
                return

            await send_status("Initializing agents...", False)
            
            llm_client = LanguageModelClient(
                llm_provider=self.valves.LLM_PROVIDER,
                model_name=self.valves.MODEL_NAME,
                api_key=self.valves.OPENAI_API_KEY,
                base_url=self.valves.OPENAI_BASE_URL if self.valves.LLM_PROVIDER == 'openai' else self.valves.OLLAMA_HOST,
            )

            milvus_client = MilvusClient(
                uri=self.valves.MILVUS_URI,
                collection_names={
                    "manual": "pacer_documents",
                    "examples": "pacer_renode_peripheral_examples",
                },
                embedding_provider=self.valves.EMBEDDING_PROVIDER,
                embedding_model_name=self.valves.EMBEDDING_MODEL_NAME,
                embedding_api_key=self.valves.EMBEDDING_API_KEY,
                embedding_base_url=self.valves.EMBEDDING_BASE_URL if self.valves.EMBEDDING_PROVIDER == 'openai' else self.valves.OLLAMA_HOST,
            )
            
            router = RoutingAgent(
                llm_client=llm_client,
                milvus_client=milvus_client,
                max_iterations=self.valves.MAX_ITERATIONS,
                send_status=send_status,
            )
            await send_status("Initialization complete. Starting generation...", True)

            final_code = await router.execute({"prompt": prompt})

            await send_status("Generation complete.", True)
            yield f"```csharp\n{final_code.strip('`').replace('csharp', '').strip()}```"

        except Exception as e:
            print(f"An error occurred in the pipe: {e}")
            yield f"Error: An unexpected error occurred during the generation process: {e}"