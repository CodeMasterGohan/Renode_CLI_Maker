"""
Configuration management for the Renode Peripheral Generator CLI.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict

try:
    import yaml
except ImportError:
    yaml = None

from .exceptions import ConfigError


@dataclass
class LLMConfig:
    """Configuration for LLM provider."""
    provider: str = "ollama"
    model: str = "llama3"
    host: str = "http://localhost:11434"
    api_key: Optional[str] = None
    max_retries: int = 3


@dataclass
class EmbeddingConfig:
    """Configuration for embedding provider."""
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    host: str = "http://localhost:11434"
    api_key: Optional[str] = None


@dataclass
class MilvusConfig:
    """Configuration for Milvus database."""
    uri: str = "localhost:19530"
    collections: Dict[str, str] = None
    
    def __post_init__(self):
        if self.collections is None:
            self.collections = {
                "manual": "pacer_documents",
                "examples": "pacer_renode_peripheral_examples",
                "rf_examples": "robotframework_test_examples",
                "rf_docs": "robotframework_documentation"
            }


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    enabled: bool = True
    ttl: int = 3600
    max_size: int = 1000
    directory: str = "~/.renode-generator/cache"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "console"
    file: Optional[str] = None


@dataclass
class RobotFrameworkConfig:
    """Configuration for RobotFramework test generation."""
    enabled: bool = True
    test_levels: List[str] = None
    output_format: str = "robot"
    include_setup: bool = True
    include_teardown: bool = True
    keyword_library: str = "ReNodeKeywords"
    output_dir: str = "tests"
    suite_name: str = "ReNodePeripheralTests"
    
    def __post_init__(self):
        if self.test_levels is None:
            self.test_levels = ["integration"]


@dataclass
class AppConfig:
    """Main application configuration."""
    llm: LLMConfig
    embedding: EmbeddingConfig
    milvus: MilvusConfig
    cache: CacheConfig
    logging: LoggingConfig
    robotframework: RobotFrameworkConfig
    mode: str = "peripheral"  # "peripheral", "robotframework", or "both"
    
    def __init__(self, **kwargs):
        self.mode = kwargs.get('mode', 'peripheral')
        self.llm = LLMConfig(**kwargs.get('llm', {}))
        self.embedding = EmbeddingConfig(**kwargs.get('embedding', {}))
        self.milvus = MilvusConfig(**kwargs.get('milvus', {}))
        self.cache = CacheConfig(**kwargs.get('cache', {}))
        self.logging = LoggingConfig(**kwargs.get('logging', {}))
        self.robotframework = RobotFrameworkConfig(**kwargs.get('robotframework', {}))


class ConfigManager:
    """Manages configuration loading, validation, and creation."""
    
    def __init__(self):
        self.default_config_dir = Path.home() / ".renode-generator"
        self.default_config_file = self.default_config_dir / "config.json"
        
    def get_default_config(self) -> Dict[str, Any]:
        """Return the default configuration."""
        config = AppConfig()
        return asdict(config)
        
    def create_default_config(self) -> int:
        """Create a default configuration file."""
        try:
            # Create config directory if it doesn't exist
            self.default_config_dir.mkdir(parents=True, exist_ok=True)
            
            # Create default configuration
            default_config = self.get_default_config()
            
            if self.default_config_file.exists():
                response = input(f"Configuration file already exists at {self.default_config_file}. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Configuration creation cancelled.")
                    return 0
            
            with open(self.default_config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                
            print(f"✓ Default configuration created at: {self.default_config_file}")
            print("\nYou can now edit this file to customize your settings.")
            print("Run 'renode-generator --check-config' to validate your configuration.")
            
            return 0
            
        except Exception as e:
            print(f"ERROR: Failed to create configuration file: {e}")
            return 1
    
    def load_config_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from a file."""
        path = Path(config_path)
        
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")
            
        try:
            with open(path, 'r') as f:
                if path.suffix.lower() in ['.yaml', '.yml']:
                    if yaml is None:
                        raise ConfigError("PyYAML is required for YAML configuration files. Install with: pip install PyYAML")
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            if yaml and hasattr(e, '__class__') and 'yaml' in str(e.__class__).lower():
                raise ConfigError(f"Failed to parse YAML configuration file {config_path}: {e}")
            raise ConfigError(f"Failed to parse configuration file {config_path}: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to read configuration file {config_path}: {e}")
    
    def load_config(self, config_path: Optional[str] = None, cli_overrides: Optional[Dict[str, Any]] = None) -> AppConfig:
        """Load configuration from file, environment, and CLI overrides."""
        
        # Start with default configuration
        config_dict = self.get_default_config()
        
        # Load from file if specified or if default exists
        if config_path:
            file_config = self.load_config_file(config_path)
            config_dict = self._merge_configs(config_dict, file_config)
        elif self.default_config_file.exists():
            file_config = self.load_config_file(str(self.default_config_file))
            config_dict = self._merge_configs(config_dict, file_config)
        
        # Apply environment variable overrides
        env_config = self._load_from_environment()
        config_dict = self._merge_configs(config_dict, env_config)
        
        # Apply CLI overrides
        if cli_overrides:
            cli_config = self._extract_cli_config(cli_overrides)
            config_dict = self._merge_configs(config_dict, cli_config)
        
        # Create and validate configuration object
        try:
            return AppConfig(**config_dict)
        except Exception as e:
            raise ConfigError(f"Invalid configuration: {e}")
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # LLM configuration
        if os.getenv('RENODE_LLM_PROVIDER'):
            config.setdefault('llm', {})['provider'] = os.getenv('RENODE_LLM_PROVIDER')
        if os.getenv('RENODE_LLM_MODEL'):
            config.setdefault('llm', {})['model'] = os.getenv('RENODE_LLM_MODEL')
        if os.getenv('RENODE_LLM_HOST'):
            config.setdefault('llm', {})['host'] = os.getenv('RENODE_LLM_HOST')
        if os.getenv('OPENAI_API_KEY'):
            config.setdefault('llm', {})['api_key'] = os.getenv('OPENAI_API_KEY')
            
        # Milvus configuration
        if os.getenv('RENODE_MILVUS_URI'):
            config.setdefault('milvus', {})['uri'] = os.getenv('RENODE_MILVUS_URI')
            
        # Cache configuration
        if os.getenv('RENODE_CACHE_DIR'):
            config.setdefault('cache', {})['directory'] = os.getenv('RENODE_CACHE_DIR')
            
        # RobotFramework configuration
        if os.getenv('RENODE_RF_ENABLED'):
            config.setdefault('robotframework', {})['enabled'] = os.getenv('RENODE_RF_ENABLED').lower() == 'true'
        if os.getenv('RENODE_RF_OUTPUT_DIR'):
            config.setdefault('robotframework', {})['output_dir'] = os.getenv('RENODE_RF_OUTPUT_DIR')
        
        # Mode configuration
        if os.getenv('RENODE_MODE'):
            config['mode'] = os.getenv('RENODE_MODE')
            
        return config
    
    def _extract_cli_config(self, cli_args: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration from CLI arguments."""
        config = {}
        
        # LLM configuration
        llm_config = {}
        if cli_args.get('llm_provider'):
            llm_config['provider'] = cli_args['llm_provider']
        if cli_args.get('llm_model'):
            llm_config['model'] = cli_args['llm_model']
        if cli_args.get('llm_host'):
            llm_config['host'] = cli_args['llm_host']
        if cli_args.get('openai_api_key'):
            llm_config['api_key'] = cli_args['openai_api_key']
        if llm_config:
            config['llm'] = llm_config
            
        # Milvus configuration
        if cli_args.get('milvus_uri'):
            config['milvus'] = {'uri': cli_args['milvus_uri']}
            
        # Cache configuration
        cache_config = {}
        if cli_args.get('no_cache'):
            cache_config['enabled'] = False
        if cli_args.get('cache_dir'):
            cache_config['directory'] = cli_args['cache_dir']
        if cache_config:
            config['cache'] = cache_config
            
        # Logging configuration
        logging_config = {}
        if cli_args.get('debug'):
            logging_config['level'] = 'DEBUG'
        elif cli_args.get('verbose'):
            logging_config['level'] = 'INFO'
        elif cli_args.get('quiet'):
            logging_config['level'] = 'WARNING'
        if logging_config:
            config['logging'] = logging_config
            
        # Mode configuration
        if cli_args.get('mode'):
            config['mode'] = cli_args['mode']
            
        # RobotFramework configuration
        rf_config = {}
        if cli_args.get('rf_test_level'):
            rf_config['test_levels'] = [cli_args['rf_test_level']] if cli_args['rf_test_level'] != 'all' else ['unit', 'integration', 'system']
        if cli_args.get('rf_output_dir'):
            rf_config['output_dir'] = cli_args['rf_output_dir']
        if rf_config:
            config['robotframework'] = rf_config
            
        return config
    
    def validate_config(self, config: AppConfig) -> int:
        """Validate configuration and check connections."""
        print("🔍 Validating configuration...")
        
        errors = []
        warnings = []
        
        try:
            # Validate LLM configuration
            print(f"  ✓ LLM Provider: {config.llm.provider}")
            print(f"  ✓ LLM Model: {config.llm.model}")
            
            if config.llm.provider == 'ollama':
                print(f"  ✓ Ollama Host: {config.llm.host}")
                # Try to connect to Ollama
                try:
                    import ollama
                    client = ollama.Client(host=config.llm.host)
                    client.show(config.llm.model)
                    print(f"  ✅ Successfully connected to Ollama and verified model")
                except Exception as e:
                    errors.append(f"Cannot connect to Ollama or verify model: {e}")
                    
            elif config.llm.provider == 'openai':
                if not config.llm.api_key:
                    warnings.append("OpenAI API key not configured")
                else:
                    print(f"  ✓ OpenAI API key configured")
                    
            # Validate Milvus configuration
            print(f"  ✓ Milvus URI: {config.milvus.uri}")
            try:
                from pymilvus import connections, utility
                connections.connect("validation", uri=config.milvus.uri)
                
                for collection_name in config.milvus.collections.values():
                    if utility.has_collection(collection_name):
                        print(f"  ✅ Milvus collection '{collection_name}' found")
                    else:
                        errors.append(f"Milvus collection '{collection_name}' not found")
                        
                connections.disconnect("validation")
                
            except Exception as e:
                errors.append(f"Cannot connect to Milvus: {e}")
                
            # Validate cache configuration
            if config.cache.enabled:
                cache_dir = Path(config.cache.directory).expanduser()
                try:
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    print(f"  ✅ Cache directory accessible: {cache_dir}")
                except Exception as e:
                    warnings.append(f"Cache directory issue: {e}")
                    
            # Show results
            print("\n" + "="*50)
            if errors:
                print("❌ Configuration validation FAILED:")
                for error in errors:
                    print(f"  • {error}")
                print("\nPlease fix these errors before running the generator.")
                return 1
            else:
                print("✅ Configuration validation PASSED")
                
            if warnings:
                print("\n⚠️  Warnings:")
                for warning in warnings:
                    print(f"  • {warning}")
                    
            return 0
            
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            return 1 