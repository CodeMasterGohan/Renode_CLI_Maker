#!/usr/bin/env python3
"""
Renode Peripheral Generator CLI - Demo Version

This is a demo version that shows the CLI interface and help functionality
without requiring external dependencies like Ollama, OpenAI, or Milvus.

Usage:
    python demo_cli.py --help
    python demo_cli.py --examples
    python demo_cli.py "Create a UART peripheral" --demo
"""

import sys
import argparse
import json
from typing import Dict, Any, Optional
from pathlib import Path


def show_examples():
    """Display comprehensive usage examples."""
    examples = [
        ("Basic Usage", [
            "python demo_cli.py \"Create a basic GPIO controller\" --demo",
            "python demo_cli.py \"Create a UART with interrupt support\" --demo",
            "python demo_cli.py \"Create an I2C master controller\" --demo"
        ]),
        ("Output Control", [
            "python demo_cli.py \"Create a timer\" -o timer.cs --demo",
            "python demo_cli.py \"Create a PWM controller\" --format json --demo",
            "python demo_cli.py \"Create an ADC\" --quiet -o adc.cs --demo"
        ]),
        ("Advanced Configuration", [
            "python demo_cli.py \"Create a DMA controller\" --iterations 5 --verbose --demo",
            "python demo_cli.py \"Create a CAN controller\" --no-cache --metrics --demo"
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


def generate_demo_peripheral(prompt: str, format_type: str = "pretty") -> str:
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


def main():
    """Main entry point for the demo CLI."""
    
    parser = argparse.ArgumentParser(
        prog='renode-generator-demo',
        description='''
Renode Peripheral Generator CLI - Demo Version

This tool demonstrates the CLI interface for generating Renode peripheral code.
The full version uses advanced multi-agent AI systems to generate high-quality 
peripheral implementations for the Renode framework.

Examples:
  %(prog)s "Create a UART peripheral for STM32" --demo
  %(prog)s "Create an SPI controller with DMA support" -o spi_controller.cs --demo
  %(prog)s "Create a GPIO controller" --verbose --demo
  %(prog)s "Create a timer with interrupts" --iterations 5 --metrics --demo
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
This is a DEMO version. For full functionality, install dependencies:
  pip install ollama openai pymilvus colorama PyYAML

For more examples and documentation, visit:
https://github.com/renode/renode-peripheral-generator
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

    # Utility commands
    utility_group = parser.add_argument_group('Utility Commands')
    utility_group.add_argument(
        '--examples',
        action='store_true',
        help='Show usage examples and exit'
    )
    utility_group.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0 (Demo)'
    )

    try:
        args = parser.parse_args()
    except SystemExit as e:
        return e.code if e.code is not None else 0

    # Handle utility commands
    if args.examples:
        show_examples()
        return 0

    # Validate arguments
    if args.verbose and args.quiet:
        print("ERROR: Cannot use both --verbose and --quiet flags simultaneously", file=sys.stderr)
        return 1

    if not args.prompt:
        print("ERROR: Prompt is required unless using utility commands.", file=sys.stderr)
        print("Use --help for usage information or --examples for examples.", file=sys.stderr)
        return 1

    if args.iterations < 1 or args.iterations > 10:
        print("ERROR: Iterations must be between 1 and 10", file=sys.stderr)
        return 1

    # Show status messages
    if not args.quiet:
        if not args.demo:
            print("⚠️  Demo mode - use --demo flag to generate sample output")
            print("💡 For full functionality, install: pip install ollama openai pymilvus")
            return 1
        
        print("ℹ Running in demo mode...")

    # Generate demo output
    try:
        if args.verbose and not args.quiet:
            print("📝 Generating demo peripheral code...")
            print(f"📝 Prompt: {args.prompt}")
            print(f"📝 Format: {args.format}")
            print(f"📝 Iterations: {args.iterations}")

        result = generate_demo_peripheral(args.prompt, args.format)

        if args.output and args.output != '-':
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result)
            if not args.quiet:
                print(f"✅ Output written to: {args.output}")
        else:
            print(result)

        if args.metrics and not args.quiet:
            print("\nDemo Metrics:")
            print("  Duration: 0.1s (demo)")
            print("  Mode: Demo")
            print("  Status: Success")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main()) 