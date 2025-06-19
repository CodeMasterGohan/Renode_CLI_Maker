# Renode Peripheral Generator - CLI Decoupling Plan

## Overview
Convert the current OpenWebUI-based Renode peripheral generator into a standalone Command Line Interface (CLI) application. This will remove all OpenWebUI dependencies while preserving the core functionality and agent-based architecture.

## Current Architecture Analysis

### Components to Preserve
1. **Core Agent System**
   - `BaseAgent` class with metrics tracking
   - `PlanningAgent`, `CodingAgent`, `ReviewingAgent`, `AccuracyAgent`
   - `RoutingAgent` for workflow orchestration
   - Agent execution metrics and error handling

2. **LLM Integration**
   - `LanguageModelClient` with Ollama/OpenAI support
   - Retry logic and caching mechanisms
   - Model verification and error handling

3. **Vector Database Integration**
   - `MilvusClient` with embedding generation
   - Collection search functionality
   - Caching for embeddings and search results

4. **Enhanced Features**
   - `InMemoryCache` with TTL support
   - Structured logging with `structlog`
   - Retry mechanisms with `tenacity`
   - Performance metrics tracking

### Components to Remove/Replace
1. **OpenWebUI Dependencies**
   - `FastAPI Request` handling
   - `EmitterType` and event emission system
   - `__event_emitter__` status updates
   - OpenWebUI-specific `Pipe` class structure

2. **Async Generator Interface**
   - Replace `AsyncGenerator[str, None]` with direct return values
   - Remove streaming response mechanism

## New CLI Architecture

### 1. CLI Interface Design

#### Command Structure
```bash
# Basic usage
renode-generator "Create a UART peripheral for STM32"

# Advanced usage with options
renode-generator \
  --prompt "Create a SPI controller with DMA support" \
  --output uart_peripheral.cs \
  --config config.json \
  --iterations 5 \
  --verbose \
  --no-cache \
  --format pretty
```

#### CLI Arguments and Options
- **Positional Arguments**
  - `prompt`: The peripheral generation request (required)

- **Optional Arguments**
  - `--output, -o`: Output file path (default: stdout)
  - `--config, -c`: Configuration file path (JSON/YAML)
  - `--iterations, -i`: Maximum refinement iterations (default: 3)
  - `--format, -f`: Output format (raw, pretty, json) (default: pretty)
  - `--verbose, -v`: Enable verbose logging
  - `--quiet, -q`: Suppress status messages
  - `--no-cache`: Disable caching
  - `--cache-dir`: Custom cache directory
  - `--metrics`: Display performance metrics
  - `--save-plan`: Save execution plan to file
  - `--load-plan`: Load execution plan from file

#### Configuration File Support
```json
{
  "llm": {
    "provider": "ollama",
    "model": "llama3",
    "host": "http://localhost:11434",
    "api_key": null,
    "max_retries": 3
  },
  "embedding": {
    "provider": "ollama",
    "model": "nomic-embed-text",
    "host": "http://localhost:11434",
    "api_key": null
  },
  "milvus": {
    "uri": "localhost:19530",
    "collections": {
      "manual": "pacer_documents",
      "examples": "pacer_renode_peripheral_examples"
    }
  },
  "cache": {
    "ttl": 3600,
    "max_size": 1000,
    "enabled": true,
    "directory": "~/.renode-generator/cache"
  },
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": null
  }
}
```

### 2. Core Application Structure

#### Main Application Class
```python
class RenodeGeneratorCLI:
    """Main CLI application class"""
    def __init__(self, config: dict)
    def run(self, prompt: str, **kwargs) -> str
    def setup_logging(self)
    def load_config(self, config_path: str)
    def initialize_clients(self)
    def display_metrics(self)
```

#### Status Reporting System
Replace OpenWebUI event emitters with console-based status updates:
- Progress bars using `rich` or `tqdm`
- Colored status messages using `colorama` or `rich`
- Structured logging output
- Optional quiet mode for scripting

#### Output Formatting
Multiple output formats to suit different use cases:
- **Raw**: Just the generated code
- **Pretty**: Formatted with syntax highlighting and metadata
- **JSON**: Machine-readable format with metrics and metadata

### 3. File Structure

```
renode_generator_cli/
├── __init__.py
├── main.py                 # CLI entry point
├── cli.py                  # Argument parsing and CLI logic
├── config.py               # Configuration management
├── core/
│   ├── __init__.py
│   ├── agents.py           # All agent classes
│   ├── clients.py          # LLM and Milvus clients
│   ├── cache.py            # Caching system
│   ├── exceptions.py       # Custom exceptions
│   └── metrics.py          # Metrics tracking
├── utils/
│   ├── __init__.py
│   ├── logging.py          # Logging configuration
│   ├── status.py           # Status reporting
│   └── formatter.py        # Output formatting
├── config/
│   ├── default.json        # Default configuration
│   └── example.json        # Example configuration
├── requirements.txt
├── setup.py
├── README.md
└── tests/
    ├── __init__.py
    ├── test_agents.py
    ├── test_clients.py
    └── test_cli.py
```

### 4. Implementation Plan

#### Phase 1: Core Refactoring
1. **Extract Core Components** (2-3 hours)
   - Move agent classes to `core/agents.py`
   - Move client classes to `core/clients.py`
   - Move caching system to `core/cache.py`
   - Remove OpenWebUI dependencies

2. **Create CLI Interface** (2-3 hours)
   - Implement argument parsing with `argparse` or `click`
   - Create main CLI entry point
   - Design configuration system

3. **Replace Status System** (1-2 hours)
   - Replace event emitters with console output
   - Implement progress tracking
   - Add verbose/quiet modes

#### Phase 2: Enhanced Features
1. **Configuration Management** (1-2 hours)
   - JSON/YAML configuration file support
   - Environment variable support
   - Configuration validation

2. **Output Formatting** (1-2 hours)
   - Multiple output formats
   - Syntax highlighting for code
   - Metadata inclusion

3. **Caching Improvements** (1 hour)
   - Persistent cache storage
   - Cache management commands
   - Cache statistics

#### Phase 3: Quality and Polish
1. **Error Handling** (1 hour)
   - User-friendly error messages
   - Exit codes for different error types
   - Recovery suggestions

2. **Documentation** (1-2 hours)
   - Comprehensive README
   - Usage examples
   - Configuration documentation

3. **Testing** (2-3 hours)
   - Unit tests for core components
   - Integration tests for CLI
   - Mock tests for external dependencies

### 5. Key Changes from OpenWebUI Version

#### Removed Components
- `Pipe` class and OpenWebUI integration
- `FastAPI` dependencies
- Event emitter system
- Async generator streaming
- `__request__`, `__user__`, `__task__`, `__tools__` parameters

#### New Components
- CLI argument parsing
- Configuration file management
- Console-based status reporting
- Multiple output formats
- Persistent caching
- Exit code handling

#### Modified Components
- **Agents**: Remove `send_status` parameter, use console output
- **RoutingAgent**: Return final result instead of yielding
- **Status Updates**: Print to console instead of emitting events
- **Error Handling**: Use exit codes and user-friendly messages

### 6. Dependencies

#### Core Dependencies (Preserved)
- `ollama` - Ollama client
- `openai` - OpenAI client
- `pymilvus` - Milvus vector database
- `tenacity` - Retry logic
- `structlog` - Structured logging
- `pydantic` - Data validation

#### New CLI Dependencies
- `click` or `argparse` - CLI argument parsing
- `rich` - Enhanced console output and formatting
- `colorama` - Cross-platform colored output
- `pyyaml` - YAML configuration support
- `appdirs` - Application directory management

#### Development Dependencies
- `pytest` - Testing framework
- `pytest-asyncio` - Async testing support
- `black` - Code formatting
- `flake8` - Linting
- `mypy` - Type checking

### 7. Usage Examples

#### Basic Usage
```bash
# Generate a simple peripheral
renode-generator "Create a basic GPIO controller"

# Save to file
renode-generator "Create a UART with interrupts" -o uart.cs

# Use custom configuration
renode-generator "Create an I2C master" -c my-config.json

# Verbose output with metrics
renode-generator "Create a timer peripheral" -v --metrics
```

#### Advanced Usage
```bash
# Custom iterations and output format
renode-generator \
  "Create a complex DMA controller with multiple channels" \
  --iterations 5 \
  --output dma_controller.cs \
  --format pretty \
  --metrics \
  --save-plan execution_plan.json

# Quiet mode for scripting
renode-generator "Create a PWM controller" -q -o pwm.cs

# Disable caching for fresh generation
renode-generator "Create an ADC peripheral" --no-cache
```

### 8. Success Criteria

1. **Functionality Preservation**
   - All core generation capabilities preserved
   - Agent-based workflow maintains quality
   - LLM and vector database integration working

2. **CLI Usability**
   - Intuitive command-line interface
   - Comprehensive help documentation
   - Multiple output options

3. **Performance**
   - Similar or better performance than OpenWebUI version
   - Efficient caching system
   - Reasonable startup time

4. **Reliability**
   - Robust error handling
   - Graceful degradation when services unavailable
   - Clear error messages and recovery guidance

5. **Maintainability**
   - Clean separation of concerns
   - Comprehensive test coverage
   - Clear documentation

### 9. Future Enhancements

#### Short-term (Next Release)
- Interactive mode for iterative refinement
- Template system for common peripheral types
- Plugin system for custom agents
- Shell completion support

#### Long-term
- Web dashboard for job monitoring
- Distributed generation across multiple nodes
- Integration with Renode project templates
- IDE plugins and extensions

### 10. Migration Strategy

1. **Parallel Development**
   - Develop CLI version alongside OpenWebUI version
   - Share core components between versions
   - Maintain backward compatibility

2. **Testing Strategy**
   - Compare outputs between versions
   - Performance benchmarking
   - User acceptance testing

3. **Documentation**
   - Migration guide for existing users
   - Feature comparison matrix
   - Best practices documentation

## Conclusion

This plan provides a comprehensive approach to decoupling the Renode peripheral generator from OpenWebUI while preserving all core functionality and enhancing the user experience with a robust CLI interface. The modular design ensures maintainability and extensibility for future enhancements.

The estimated development time is 8-12 hours for a fully functional CLI version with all planned features and comprehensive testing. 