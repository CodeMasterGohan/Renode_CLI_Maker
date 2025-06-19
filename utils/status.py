"""
Status reporting and console output utilities for the CLI.
"""

import sys
from typing import Dict, Any, Optional
from datetime import datetime


class StatusReporter:
    """Handles console output with colors and formatting."""
    
    def __init__(self):
        self.verbose = False
        self.quiet = False
        self.debug = False
        
        # Color codes for different output types
        self.colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'bold': '\033[1m',
            'reset': '\033[0m'
        }
        
        # Check if we should use colors (disable on Windows without colorama)
        self.use_colors = self._should_use_colors()
        
    def _should_use_colors(self) -> bool:
        """Determine if we should use colored output."""
        # Disable colors if output is redirected
        if not sys.stdout.isatty():
            return False
            
        # Try to use colorama on Windows
        try:
            import colorama
            colorama.init()
            return True
        except ImportError:
            # On non-Windows systems, assume color support
            import platform
            return platform.system() != 'Windows'
    
    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if not self.use_colors:
            return text
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
    
    def setup(self, verbose: bool = False, quiet: bool = False, debug: bool = False):
        """Configure the status reporter."""
        self.verbose = verbose
        self.quiet = quiet
        self.debug = debug
        
    def _timestamp(self) -> str:
        """Get formatted timestamp for debug output."""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    def info(self, message: str):
        """Display an info message."""
        if not self.quiet:
            prefix = self._colorize("ℹ", "blue")
            print(f"{prefix} {message}")
    
    def success(self, message: str):
        """Display a success message."""
        if not self.quiet:
            prefix = self._colorize("✅", "green")
            print(f"{prefix} {message}")
    
    def warning(self, message: str):
        """Display a warning message."""
        if not self.quiet:
            prefix = self._colorize("⚠️", "yellow")
            print(f"{prefix} {message}", file=sys.stderr)
    
    def error(self, message: str):
        """Display an error message."""
        prefix = self._colorize("❌", "red")
        print(f"{prefix} {message}", file=sys.stderr)
    
    def debug(self, message: str):
        """Display a debug message."""
        if self.debug:
            timestamp = self._colorize(f"[{self._timestamp()}]", "cyan")
            prefix = self._colorize("🐛", "magenta")
            print(f"{timestamp} {prefix} {message}")
    
    def verbose(self, message: str):
        """Display a verbose message."""
        if self.verbose and not self.quiet:
            prefix = self._colorize("📝", "cyan")
            print(f"{prefix} {message}")
    
    def progress(self, message: str, step: int, total: int):
        """Display progress information."""
        if not self.quiet:
            percentage = (step / total) * 100 if total > 0 else 0
            bar_length = 20
            filled_length = int(bar_length * step / total) if total > 0 else 0
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            
            progress_text = f"[{bar}] {percentage:.1f}% ({step}/{total})"
            if self.use_colors:
                progress_text = self._colorize(progress_text, "blue")
            
            print(f"\r{progress_text} {message}", end='', flush=True)
            
            if step >= total:
                print()  # New line when complete
    
    def status_update(self, message: str):
        """Display a status update (can be overwritten)."""
        if not self.quiet:
            if self.verbose:
                print(f"⏳ {message}")
            else:
                print(f"\r⏳ {message}", end='', flush=True)
    
    def newline(self):
        """Print a new line (useful after status updates)."""
        if not self.quiet:
            print()
    
    def section(self, title: str):
        """Display a section header."""
        if not self.quiet:
            separator = "─" * min(50, len(title) + 4)
            title_colored = self._colorize(title, "bold")
            print(f"\n{separator}")
            print(f"{title_colored}")
            print(f"{separator}")
    
    def show_metrics(self, metrics: Dict[str, Any]):
        """Display performance metrics in a formatted way."""
        if self.quiet:
            return
            
        self.section("Performance Metrics")
        
        # Duration
        if 'duration' in metrics:
            duration = metrics['duration']
            duration_str = f"{duration:.2f}s"
            if duration > 60:
                minutes = int(duration // 60)
                seconds = duration % 60
                duration_str = f"{minutes}m {seconds:.1f}s"
            print(f"  Duration: {self._colorize(duration_str, 'green')}")
        
        # Iterations
        if 'iterations' in metrics:
            print(f"  Iterations: {self._colorize(str(metrics['iterations']), 'blue')}")
        
        # Cache statistics
        if 'cache_hits' in metrics and 'cache_misses' in metrics:
            hits = metrics['cache_hits']
            misses = metrics['cache_misses']
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            
            print(f"  Cache hits: {self._colorize(str(hits), 'green')}")
            print(f"  Cache misses: {self._colorize(str(misses), 'yellow')}")
            print(f"  Cache hit rate: {self._colorize(f'{hit_rate:.1f}%', 'green' if hit_rate > 50 else 'yellow')}")
        
        # Agent metrics
        if 'agents' in metrics:
            print(f"  Agent executions:")
            for agent_name, agent_metrics in metrics['agents'].items():
                if 'execution_count' in agent_metrics:
                    count = agent_metrics['execution_count']
                    avg_time = agent_metrics.get('avg_execution_time', 0)
                    print(f"    {agent_name}: {self._colorize(str(count), 'blue')} " +
                          f"(avg: {avg_time:.2f}s)")
        
        # Errors
        if 'errors' in metrics and metrics['errors']:
            print(f"  Errors encountered: {self._colorize(str(len(metrics['errors'])), 'red')}")
            if self.verbose:
                for error in metrics['errors'][:3]:  # Show first 3 errors
                    print(f"    • {error}")
                if len(metrics['errors']) > 3:
                    print(f"    • ... and {len(metrics['errors']) - 3} more") 