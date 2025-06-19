#!/usr/bin/env python3
"""
Renode Peripheral Generator CLI

A command-line interface for generating Renode peripheral code using multi-agent systems.
This tool helps create C# peripheral implementations for the Renode framework.

Usage:
    renode-generator "Create a UART peripheral for STM32"
    renode-generator --help
    renode-generator --examples

For detailed usage information, run with --help
"""

import sys
import os
import argparse
import json
import traceback
from typing import Dict, Any, Optional
from pathlib import Path

# Import our core modules
try:
    from core.application import RenodeGeneratorCLI
    from core.config import ConfigManager, ConfigError
    from core.exceptions import RenodeGeneratorError
    from utils.formatter import OutputFormatter
    from utils.status import StatusReporter
except ImportError as e:
    print(f"ERROR: Failed to import core modules: {e}")
    print("Please ensure all required files are present and dependencies are installed.")
    sys.exit(1)


class CLIInterface:
    """Main CLI interface class that handles argument parsing and orchestration."""

    def __init__(self):
        self.config_manager = ConfigManager()
        self.status_reporter = StatusReporter()
        
    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser with comprehensive help."""
        
        parser = argparse.ArgumentParser(
            prog='renode-generator',
            description='''
Renode Peripheral Generator CLI - Generate C# peripheral code for Renode

This tool uses advanced multi-agent AI systems to generate high-quality peripheral 
implementations for the Renode framework. It leverages vector databases of documentation
and examples to produce accurate, well-documented code.

Examples:
  %(prog)s "Create a UART peripheral for STM32"
  %(prog)s "Create an SPI controller with DMA support" -o spi_controller.cs
  %(prog)s "Create a GPIO controller" --config my-config.json --verbose
  %(prog)s "Create a timer with interrupts" --iterations 5 --metrics
            ''',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Configuration:
  The tool can be configured via:
  1. Command line arguments (highest priority)
  2. Configuration file specified with --config
  3. Default configuration file: ~/.renode-generator/config.json
  4. Environment variables (lowest priority)

For more examples and documentation, visit:
https://github.com/renode/renode-peripheral-generator
            '''
        )

        # Positional argument
        parser.add_argument(
            'prompt',
            nargs='?',  # Make it optional to allow --help, --examples, etc.
            help='Description of the peripheral to generate (required unless using --examples or --help)'
        )

        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument(
            '-o', '--output',
            type=str,
            help='Output file path (default: stdout). Use - for stdout explicitly.'
        )
        output_group.add_argument(
            '-f', '--format',
            choices=['raw', 'pretty', 'json'],
            default='pretty',
            help='Output format: raw (code only), pretty (formatted with metadata), json (machine readable) (default: %(default)s)'
        )

        # Configuration options
        config_group = parser.add_argument_group('Configuration Options')
        config_group.add_argument(
            '-c', '--config',
            type=str,
            help='Path to configuration file (JSON or YAML)'
        )
        config_group.add_argument(
            '--create-config',
            action='store_true',
            help='Create a default configuration file and exit'
        )

        # Generation options
        generation_group = parser.add_argument_group('Generation Options')
        generation_group.add_argument(
            '-i', '--iterations',
            type=int,
            default=3,
            help='Maximum number of refinement iterations (default: %(default)s)'
        )
        generation_group.add_argument(
            '--no-cache',
            action='store_true',
            help='Disable caching (forces fresh generation)'
        )
        generation_group.add_argument(
            '--cache-dir',
            type=str,
            help='Custom cache directory (default: ~/.renode-generator/cache)'
        )

        # LLM Configuration
        llm_group = parser.add_argument_group('LLM Configuration')
        llm_group.add_argument(
            '--llm-provider',
            choices=['ollama', 'openai'],
            help='LLM provider to use (overrides config file)'
        )
        llm_group.add_argument(
            '--llm-model',
            type=str,
            help='LLM model name (overrides config file)'
        )
        llm_group.add_argument(
            '--llm-host',
            type=str,
            help='LLM host URL (for Ollama) (overrides config file)'
        )
        llm_group.add_argument(
            '--openai-api-key',
            type=str,
            help='OpenAI API key (overrides config file and environment)'
        )

        # Database Configuration
        db_group = parser.add_argument_group('Database Configuration')
        db_group.add_argument(
            '--milvus-uri',
            type=str,
            help='Milvus database URI (overrides config file)'
        )

        # Execution control
        execution_group = parser.add_argument_group('Execution Control')
        execution_group.add_argument(
            '--save-plan',
            type=str,
            help='Save execution plan to file (JSON format)'
        )
        execution_group.add_argument(
            '--load-plan',
            type=str,
            help='Load execution plan from file (JSON format)'
        )

        # Logging and verbosity
        logging_group = parser.add_argument_group('Logging and Output Control')
        logging_group.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Enable verbose output and detailed logging'
        )
        logging_group.add_argument(
            '-q', '--quiet',
            action='store_true',
            help='Suppress status messages (quiet mode for scripting)'
        )
        logging_group.add_argument(
            '--metrics',
            action='store_true',
            help='Display performance metrics after generation'
        )
        logging_group.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging (implies --verbose)'
        )

        # Utility commands
        utility_group = parser.add_argument_group('Utility Commands')
        utility_group.add_argument(
            '--examples',
            action='store_true',
            help='Show usage examples and exit'
        )
        utility_group.add_argument(
            '--check-config',
            action='store_true',
            help='Validate configuration and check connections'
        )
        utility_group.add_argument(
            '--version',
            action='version',
            version='%(prog)s 1.0.0'
        )

        return parser

    def show_examples(self):
        """Display comprehensive usage examples."""
        examples = [
            ("Basic Usage", [
                "renode-generator \"Create a basic GPIO controller\"",
                "renode-generator \"Create a UART with interrupt support\"",
                "renode-generator \"Create an I2C master controller\""
            ]),
            ("Output Control", [
                "renode-generator \"Create a timer\" -o timer.cs",
                "renode-generator \"Create a PWM controller\" --format json",
                "renode-generator \"Create an ADC\" --quiet -o adc.cs"
            ]),
            ("Advanced Configuration", [
                "renode-generator \"Create a DMA controller\" --config custom.json",
                "renode-generator \"Create an SPI controller\" --iterations 5 --verbose",
                "renode-generator \"Create a CAN controller\" --no-cache --metrics"
            ]),
            ("LLM Provider Configuration", [
                "renode-generator \"Create a USB controller\" --llm-provider openai",
                "renode-generator \"Create an Ethernet MAC\" --llm-model gpt-4",
                "renode-generator \"Create a display controller\" --llm-host http://server:11434"
            ]),
            ("Workflow Examples", [
                "renode-generator \"Create a complex peripheral\" --save-plan plan.json",
                "renode-generator --load-plan plan.json",
                "renode-generator --check-config",
                "renode-generator --create-config"
            ])
        ]

        print("=" * 60)
        print("RENODE PERIPHERAL GENERATOR - USAGE EXAMPLES")
        print("=" * 60)
        print()

        for category, example_list in examples:
            print(f"{category}:")
            print("-" * len(category))
            for example in example_list:
                print(f"  {example}")
            print()

        print("Configuration File Example:")
        print("-" * 25)
        config_example = {
            "llm": {
                "provider": "ollama",
                "model": "llama3",
                "host": "http://localhost:11434"
            },
            "milvus": {
                "uri": "localhost:19530",
                "collections": {
                    "manual": "pacer_documents",
                    "examples": "pacer_renode_peripheral_examples"
                }
            },
            "cache": {
                "enabled": True,
                "ttl": 3600
            }
        }
        print(json.dumps(config_example, indent=2))
        print()

    def validate_arguments(self, args: argparse.Namespace) -> None:
        """Validate command line arguments and provide helpful error messages."""
        
        # Check for conflicting flags
        if args.verbose and args.quiet:
            raise ValueError("Cannot use both --verbose and --quiet flags simultaneously")
        
        if args.debug:
            args.verbose = True  # Debug implies verbose
            
        # Check if prompt is required
        utility_commands = [args.examples, args.check_config, args.create_config, args.load_plan]
        if not any(utility_commands) and not args.prompt:
            raise ValueError(
                "Prompt is required unless using utility commands.\n"
                "Use 'renode-generator --help' for usage information or "
                "'renode-generator --examples' for examples."
            )
            
        # Validate file paths
        if args.config and not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")
            
        if args.load_plan and not os.path.exists(args.load_plan):
            raise FileNotFoundError(f"Plan file not found: {args.load_plan}")
            
        # Validate output directory
        if args.output and args.output != '-':
            output_dir = os.path.dirname(os.path.abspath(args.output))
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except OSError as e:
                    raise ValueError(f"Cannot create output directory {output_dir}: {e}")
                    
        # Validate iterations
        if args.iterations < 1 or args.iterations > 10:
            raise ValueError("Iterations must be between 1 and 10")

    def run(self, args: argparse.Namespace) -> int:
        """Main execution method that orchestrates the generation process."""
        
        try:
            # Handle utility commands first
            if args.examples:
                self.show_examples()
                return 0
                
            if args.create_config:
                return self.config_manager.create_default_config()
                
            # Load configuration
            config = self.config_manager.load_config(
                config_path=args.config,
                cli_overrides=vars(args)
            )
            
            if args.check_config:
                return self.config_manager.validate_config(config)
            
            # Initialize status reporter
            self.status_reporter.setup(
                verbose=args.verbose,
                quiet=args.quiet,
                debug=args.debug
            )
            
            # Initialize the main application
            if not args.quiet:
                self.status_reporter.info("Initializing Renode Peripheral Generator...")
                
            app = RenodeGeneratorCLI(config, self.status_reporter)
            
            # Execute generation
            if args.load_plan:
                result = app.run_from_plan(args.load_plan)
            else:
                result = app.run(
                    prompt=args.prompt,
                    max_iterations=args.iterations,
                    use_cache=not args.no_cache,
                    save_plan=args.save_plan
                )
            
            # Format and output result
            formatter = OutputFormatter(args.format)
            formatted_output = formatter.format(result, app.get_metrics() if args.metrics else None)
            
            if args.output and args.output != '-':
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(formatted_output)
                if not args.quiet:
                    self.status_reporter.success(f"Output written to: {args.output}")
            else:
                print(formatted_output)
                
            if args.metrics and not args.quiet:
                metrics = app.get_metrics()
                self.status_reporter.show_metrics(metrics)
                
            return 0
            
        except ConfigError as e:
            self.status_reporter.error(f"Configuration error: {e}")
            if args.verbose:
                self.status_reporter.error("Use --check-config to validate your configuration")
            return 2
            
        except RenodeGeneratorError as e:
            self.status_reporter.error(f"Generation error: {e}")
            if args.debug:
                self.status_reporter.error(traceback.format_exc())
            return 3
            
        except KeyboardInterrupt:
            self.status_reporter.warning("Generation interrupted by user")
            return 130
            
        except Exception as e:
            self.status_reporter.error(f"Unexpected error: {e}")
            if args.debug:
                self.status_reporter.error(traceback.format_exc())
            else:
                self.status_reporter.error("Use --debug for detailed error information")
            return 1


def main():
    """Main entry point for the CLI application."""
    
    cli = CLIInterface()
    parser = cli.create_parser()
    
    # Parse arguments
    try:
        args = parser.parse_args()
    except SystemExit as e:
        # argparse calls sys.exit(), we want to handle it gracefully
        return e.code if e.code is not None else 0
    
    # Validate arguments
    try:
        cli.validate_arguments(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    # Run the application
    return cli.run(args)


if __name__ == '__main__':
    sys.exit(main()) 