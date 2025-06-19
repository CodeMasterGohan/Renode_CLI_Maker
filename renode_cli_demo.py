#!/usr/bin/env python3
"""
Renode Peripheral Generator CLI - Demo Version

This is a demo version that shows the CLI interface and help functionality
without requiring external dependencies like Ollama, OpenAI, or Milvus.

Usage:
    python renode_cli_demo.py --help
    python renode_cli_demo.py --examples
    python renode_cli_demo.py "Create a UART peripheral" --demo
"""

import sys
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path


class DemoStatusReporter:
    """Demo status reporter for the CLI."""
    
    def __init__(self):
        self.verbose = False
        self.quiet = False
        self.debug = False
        
    def setup(self, verbose: bool = False, quiet: bool = False, debug: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.debug = debug
        
    def info(self, message: str):
        if not self.quiet:
            print(f"ℹ {message}")
    
    def success(self, message: str):
        if not self.quiet:
            print(f"✅ {message}")
    
    def warning(self, message: str):
        if not self.quiet:
            print(f"⚠️ {message}", file=sys.stderr)
    
    def error(self, message: str):
        print(f"❌ {message}", file=sys.stderr)
    
    def verbose_msg(self, message: str):
        if self.verbose and not self.quiet:
            print(f"📝 {message}")
    
    def status_update(self, message: str):
        if not self.quiet:
            print(f"⏳ {message}")
    
    def newline(self):
        if not self.quiet:
            print()


class CLIInterface:
    """Demo CLI interface class."""

    def __init__(self):
        self.status_reporter = DemoStatusReporter()
        
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

NOTE: This is a demo version. To use the full functionality, install:
  pip install ollama openai pymilvus colorama PyYAML
            '''
        )

        # Positional argument
        parser.add_argument(
            'prompt',
            nargs='?',
            help='Description of the peripheral to generate (required unless using --examples or --help)'
        )

        # Demo flag
        parser.add_argument(
            '--demo',
            action='store_true',
            help='Run in demo mode (generates sample output without LLM)'
        )

        # Output options
        output_group = parser.add_argument_group('Output Options')
        output_group.add_argument(
            '-o', '--output',
            type=str,
            help='Output file path (default: stdout)'
        )
        output_group.add_argument(
            '-f', '--format',
            choices=['raw', 'pretty', 'json'],
            default='pretty',
            help='Output format (default: %(default)s)'
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

        # LLM Configuration
        llm_group = parser.add_argument_group('LLM Configuration')
        llm_group.add_argument(
            '--llm-provider',
            choices=['ollama', 'openai'],
            help='LLM provider to use'
        )
        llm_group.add_argument(
            '--llm-model',
            type=str,
            help='LLM model name'
        )

        # Logging and verbosity
        logging_group = parser.add_argument_group('Logging and Output Control')
        logging_group.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
        logging_group.add_argument(
            '-q', '--quiet',
            action='store_true',
            help='Suppress status messages'
        )
        logging_group.add_argument(
            '--metrics',
            action='store_true',
            help='Display performance metrics'
        )
        logging_group.add_argument(
            '--debug',
            action='store_true',
            help='Enable debug logging'
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
            version='%(prog)s 1.0.0 (Demo)'
        )

        return parser

    def show_examples(self):
        """Display comprehensive usage examples."""
        examples = [
            ("Basic Usage", [
                "python renode_cli_demo.py \"Create a basic GPIO controller\" --demo",
                "python renode_cli_demo.py \"Create a UART with interrupt support\" --demo",
                "python renode_cli_demo.py \"Create an I2C master controller\" --demo"
            ]),
            ("Output Control", [
                "python renode_cli_demo.py \"Create a timer\" -o timer.cs --demo",
                "python renode_cli_demo.py \"Create a PWM controller\" --format json --demo",
                "python renode_cli_demo.py \"Create an ADC\" --quiet -o adc.cs --demo"
            ]),
            ("Advanced Configuration", [
                "python renode_cli_demo.py \"Create a DMA controller\" --iterations 5 --verbose --demo",
                "python renode_cli_demo.py \"Create a CAN controller\" --no-cache --metrics --demo"
            ]),
            ("Real Usage (requires dependencies)", [
                "renode-generator \"Create a USB controller\" --llm-provider openai",
                "renode-generator \"Create an Ethernet MAC\" --llm-model gpt-4",
                "renode-generator \"Create a display controller\" --config custom.json"
            ])
        ]

        print("=" * 60)
        print("RENODE PERIPHERAL GENERATOR - USAGE EXAMPLES (DEMO)")
        print("=" * 60)
        print()

        for category, example_list in examples:
            print(f"{category}:")
            print("-" * len(category))
            for example in example_list:
                print(f"  {example}")
            print()

        print("Installation for Full Functionality:")
        print("-" * 35)
        print("  pip install ollama openai pymilvus colorama PyYAML")
        print("  # Set up Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  # Set up Milvus: docker run -d milvusdb/milvus:latest")
        print()

    def create_demo_config(self) -> int:
        """Create a demo configuration file."""
        config_dir = Path.home() / ".renode-generator"
        config_file = config_dir / "config.json"
        
        try:
            config_dir.mkdir(parents=True, exist_ok=True)
            
            demo_config = {
                "llm": {
                    "provider": "ollama",
                    "model": "llama3",
                    "host": "http://localhost:11434"
                },
                "embedding": {
                    "provider": "ollama", 
                    "model": "nomic-embed-text",
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
            
            if config_file.exists():
                response = input(f"Configuration file already exists at {config_file}. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    print("Configuration creation cancelled.")
                    return 0
            
            with open(config_file, 'w') as f:
                json.dump(demo_config, f, indent=2)
                
            print(f"✅ Demo configuration created at: {config_file}")
            print("\nTo use the full version, install dependencies:")
            print("  pip install ollama openai pymilvus colorama PyYAML")
            
            return 0
            
        except Exception as e:
            print(f"❌ Failed to create configuration file: {e}")
            return 1

    def generate_demo_peripheral(self, prompt: str, format_type: str = "pretty") -> str:
        """Generate a demo peripheral based on the prompt."""
        
        # Simple demo code generation based on prompt keywords
        if "uart" in prompt.lower():
            peripheral_type = "UART"
            class_name = "UartPeripheral"
            description = "Universal Asynchronous Receiver-Transmitter"
        elif "gpio" in prompt.lower():
            peripheral_type = "GPIO"
            class_name = "GpioController"
            description = "General Purpose Input/Output Controller"
        elif "timer" in prompt.lower():
            peripheral_type = "Timer"
            class_name = "TimerPeripheral"
            description = "Hardware Timer"
        elif "spi" in prompt.lower():
            peripheral_type = "SPI"
            class_name = "SpiController"
            description = "Serial Peripheral Interface Controller"
        else:
            peripheral_type = "Generic"
            class_name = "GenericPeripheral"
            description = "Generic Peripheral Implementation"

        demo_code = f'''using System;
using Antmicro.Renode.Core;
using Antmicro.Renode.Logging;
using Antmicro.Renode.Peripherals;
using Antmicro.Renode.Peripherals.Bus;

namespace Antmicro.Renode.Peripherals.Demo
{{
    /// <summary>
    /// {description} - Generated by Renode Peripheral Generator CLI
    /// This is a demo implementation based on: {prompt}
    /// </summary>
    public class {class_name} : IDoubleWordPeripheral, IProvidesRegisterCollection<DoubleWordRegisterCollection>
    {{
        public {class_name}()
        {{
            RegistersCollection = new DoubleWordRegisterCollection(this);
            DefineRegisters();
            Reset();
        }}

        public void Reset()
        {{
            RegistersCollection.Reset();
            this.Log(LogLevel.Info, "{peripheral_type} peripheral reset");
        }}

        public uint ReadDoubleWord(long offset)
        {{
            return RegistersCollection.Read(offset);
        }}

        public void WriteDoubleWord(long offset, uint value)
        {{
            RegistersCollection.Write(offset, value);
        }}

        private void DefineRegisters()
        {{
            Registers.Control.Define(this)
                .WithFlag(0, out controlEnable, name: "ENABLE")
                .WithReservedBits(1, 31);

            Registers.Status.Define(this)
                .WithFlag(0, FieldMode.Read, name: "READY")
                .WithReservedBits(1, 31);

            Registers.Data.Define(this)
                .WithValueField(0, 32, out dataField, name: "DATA");
        }}

        public DoubleWordRegisterCollection RegistersCollection {{ get; }}

        private IFlagRegisterField controlEnable;
        private IValueRegisterField dataField;

        private enum Registers : long
        {{
            Control = 0x00,
            Status = 0x04, 
            Data = 0x08
        }}
    }}
}}'''

        if format_type == "raw":
            return demo_code
        elif format_type == "json":
            return json.dumps({
                "timestamp": "2024-01-27T14:30:25.123456",
                "result": demo_code,
                "success": True,
                "demo": True,
                "metrics": {
                    "duration": 0.1,
                    "iterations": 1,
                    "demo_mode": True
                }
            }, indent=2)
        else:  # pretty
            return f'''================================================================================
RENODE PERIPHERAL GENERATOR - GENERATED CODE (DEMO)
================================================================================
Generated at: 2024-01-27 14:30:25
Prompt: {prompt}

Generated Code:
----------------------------------------
{demo_code}

Demo Mode Info:
----------------------------------------
⚠️  This is demo output generated without AI
🔧 For real generation, install: pip install ollama openai pymilvus
📖 Use --help for full configuration options
================================================================================'''

    def validate_arguments(self, args: argparse.Namespace) -> None:
        """Validate command line arguments."""
        
        if args.verbose and args.quiet:
            raise ValueError("Cannot use both --verbose and --quiet flags simultaneously")
        
        if args.debug:
            args.verbose = True
            
        utility_commands = [args.examples, args.check_config, args.create_config]
        if not any(utility_commands) and not args.prompt:
            raise ValueError(
                "Prompt is required unless using utility commands.\n"
                "Use --help for usage information or --examples for examples."
            )
            
        if args.iterations < 1 or args.iterations > 10:
            raise ValueError("Iterations must be between 1 and 10")

    def run(self, args: argparse.Namespace) -> int:
        """Main execution method."""
        
        try:
            if args.examples:
                self.show_examples()
                return 0
                
            if args.create_config:
                return self.create_demo_config()
                
            if args.check_config:
                print("🔍 Configuration Check (Demo Mode)")
                print("⚠️  This is demo mode - actual connectivity not tested")
                print("✅ Demo configuration validation passed")
                print("💡 For real validation, install dependencies and use full version")
                return 0
            
            self.status_reporter.setup(
                verbose=args.verbose,
                quiet=args.quiet,
                debug=args.debug
            )
            
            if not args.demo and not args.quiet:
                self.status_reporter.warning("Demo mode - use --demo flag or install dependencies for full functionality")
            
            if args.demo:
                self.status_reporter.info("Running in demo mode...")
                result = self.generate_demo_peripheral(args.prompt, args.format)
            else:
                self.status_reporter.error("Full functionality requires dependencies. Use --demo for demo mode.")
                self.status_reporter.info("Install with: pip install ollama openai pymilvus colorama PyYAML")
                return 1
            
            if args.output and args.output != '-':
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(result)
                if not args.quiet:
                    self.status_reporter.success(f"Output written to: {args.output}")
            else:
                print(result)
                
            if args.metrics and not args.quiet:
                print("\nDemo Metrics:")
                print("  Duration: 0.1s (demo)")
                print("  Mode: Demo")
                print("  Status: Success")
                
            return 0
            
        except Exception as e:
            self.status_reporter.error(f"Error: {e}")
            return 1


def main():
    """Main entry point for the demo CLI."""
    
    cli = CLIInterface()
    parser = cli.create_parser()
    
    try:
        args = parser.parse_args()
    except SystemExit as e:
        return e.code if e.code is not None else 0
    
    try:
        cli.validate_arguments(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    
    return cli.run(args)


if __name__ == '__main__':
    sys.exit(main()) 