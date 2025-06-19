# Renode Peripheral Generator CLI

A command-line interface for generating Renode peripheral code using advanced multi-agent AI systems. This tool creates high-quality C# peripheral implementations for the Renode framework.

## Features

- **Multi-Agent Architecture**: Uses specialized AI agents for planning, coding, reviewing, and verification
- **Vector Database Integration**: Leverages Milvus for contextual code generation using documentation and examples
- **Multiple LLM Support**: Works with Ollama (local) and OpenAI (cloud) language models
- **Intelligent Caching**: Reduces generation time with smart caching of LLM responses and embeddings
- **Comprehensive Error Handling**: Provides clear, actionable error messages
- **Multiple Output Formats**: Raw code, formatted output, or JSON for automation
- **Flexible Configuration**: File-based, environment variables, or command-line configuration

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Set up Ollama** (for local LLM):
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull llama3
ollama pull nomic-embed-text
```

3. **Set up Milvus** (vector database):
```bash
# Using Docker
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  milvusdb/milvus:latest standalone
```

## Quick Start

1. **Create Default Configuration**:
```bash
python renode_generator --create-config
```

2. **Test Your Setup**:
```bash
python renode_generator --check-config
```

3. **Generate Your First Peripheral**:
```bash
python renode_generator "Create a UART peripheral for STM32"
```

## Usage Examples

### Basic Usage
```bash
# Generate a basic peripheral
python renode_generator "Create a GPIO controller"

# Save output to file
python renode_generator "Create a UART with interrupts" -o uart.cs

# Use verbose output
python renode_generator "Create an I2C master" --verbose
```

### Advanced Usage
```bash
# Custom configuration and iterations
python renode_generator "Create a DMA controller" \
  --config my-config.json \
  --iterations 5 \
  --metrics

# Use OpenAI instead of Ollama
python renode_generator "Create a USB controller" \
  --llm-provider openai \
  --llm-model gpt-4

# Disable caching for fresh generation
python renode_generator "Create a timer peripheral" --no-cache

# JSON output for automation
python renode_generator "Create an ADC" --format json
```

### Workflow Examples
```bash
# Save execution plan for reuse
python renode_generator "Create a complex peripheral" \
  --save-plan complex_peripheral_plan.json

# Execute saved plan
python renode_generator --load-plan complex_peripheral_plan.json

# Quiet mode for scripting
python renode_generator "Create a PWM controller" --quiet -o pwm.cs
```

## Configuration

### Configuration File
Create `~/.renode-generator/config.json`:

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
    "enabled": true,
    "ttl": 3600,
    "max_size": 1000,
    "directory": "~/.renode-generator/cache"
  },
  "logging": {
    "level": "INFO",
    "format": "console",
    "file": null
  }
}
```

### Environment Variables
```bash
# LLM Configuration
export RENODE_LLM_PROVIDER=ollama
export RENODE_LLM_MODEL=llama3
export RENODE_LLM_HOST=http://localhost:11434
export OPENAI_API_KEY=your_openai_key

# Milvus Configuration
export RENODE_MILVUS_URI=localhost:19530

# Cache Configuration
export RENODE_CACHE_DIR=/custom/cache/path
```

## Command Line Options

### Output Options
- `-o, --output FILE`: Output file path (default: stdout)
- `-f, --format FORMAT`: Output format: raw, pretty, json (default: pretty)

### Configuration Options
- `-c, --config FILE`: Configuration file path
- `--create-config`: Create default configuration file

### Generation Options
- `-i, --iterations N`: Maximum refinement iterations (default: 3)
- `--no-cache`: Disable caching
- `--cache-dir DIR`: Custom cache directory

### LLM Configuration
- `--llm-provider PROVIDER`: LLM provider (ollama, openai)
- `--llm-model MODEL`: LLM model name
- `--llm-host URL`: LLM host URL (for Ollama)
- `--openai-api-key KEY`: OpenAI API key

### Database Configuration
- `--milvus-uri URI`: Milvus database URI

### Execution Control
- `--save-plan FILE`: Save execution plan to file
- `--load-plan FILE`: Load execution plan from file

### Logging and Output
- `-v, --verbose`: Enable verbose output
- `-q, --quiet`: Suppress status messages
- `--metrics`: Display performance metrics
- `--debug`: Enable debug logging

### Utility Commands
- `--examples`: Show usage examples
- `--check-config`: Validate configuration
- `--version`: Show version information

## Error Handling

The CLI provides detailed error messages with specific guidance:

### Configuration Errors
```
❌ Configuration error: Cannot connect to Ollama or verify model: Connection refused
💡 Check that Ollama is running: ollama serve
💡 Verify model is available: ollama list
```

### Generation Errors
```
❌ Generation error: Milvus collection 'pacer_documents' not found
💡 Ensure Milvus collections are properly set up
💡 Check collection names in configuration file
```

### Common Issues
1. **Ollama not running**: Start with `ollama serve`
2. **Model not found**: Pull with `ollama pull llama3`
3. **Milvus connection failed**: Check Docker container is running
4. **Permission denied**: Check file/directory permissions

## Output Formats

### Pretty Format (Default)
```
================================================================================
RENODE PERIPHERAL GENERATOR - GENERATED CODE
================================================================================
Generated at: 2024-01-27 14:30:25

Generated Code:
----------------------------------------
using System;
using Antmicro.Renode.Core;
...
[C# code here]
...

Generation Metrics:
----------------------------------------
Duration: 45.2s
Iterations: 3
Cache hit rate: 67.5%
================================================================================
```

### Raw Format
```
using System;
using Antmicro.Renode.Core;
...
[C# code only]
```

### JSON Format
```json
{
  "timestamp": "2024-01-27T14:30:25.123456",
  "result": "using System;\nusing Antmicro.Renode.Core;\n...",
  "success": true,
  "metrics": {
    "duration": 45.2,
    "iterations": 3,
    "cache_hits": 27,
    "cache_misses": 13
  }
}
```

## Performance Tips

1. **Use Caching**: Keep cache enabled for faster subsequent generations
2. **Local Models**: Use Ollama for faster response times and privacy
3. **Adjust Iterations**: Reduce iterations for faster results, increase for higher quality
4. **Batch Processing**: Use JSON output with scripts for multiple generations

## Troubleshooting

### Debug Mode
```bash
python renode_generator "test prompt" --debug
```

### Configuration Validation
```bash
python renode_generator --check-config
```

### Verbose Output
```bash
python renode_generator "test prompt" --verbose
```

## Architecture

The CLI uses a multi-agent architecture:

1. **Planning Agent**: Analyzes requirements and creates implementation plan
2. **Coding Agent**: Generates initial C# peripheral code
3. **Reviewing Agent**: Reviews and improves code quality
4. **Accuracy Agent**: Verifies correctness and compliance
5. **Routing Agent**: Orchestrates the workflow

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

MIT License - see LICENSE file for details. 