"""
Output formatting utilities for different output formats.
"""

import json
from typing import Any, Dict, Optional
from datetime import datetime


class OutputFormatter:
    """Formats output in different styles based on user preference."""
    
    def __init__(self, format_type: str = "pretty"):
        self.format_type = format_type.lower()
        
    def format(self, result: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Format the result according to the specified format type."""
        
        if self.format_type == "raw":
            return self._format_raw(result)
        elif self.format_type == "json":
            return self._format_json(result, metrics)
        else:  # pretty (default)
            return self._format_pretty(result, metrics)
    
    def _format_raw(self, result: str) -> str:
        """Return just the raw code without any formatting."""
        return result.strip()
    
    def _format_json(self, result: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Format as JSON for machine consumption."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "success": True
        }
        
        if metrics:
            output["metrics"] = metrics
            
        return json.dumps(output, indent=2)
    
    def _format_pretty(self, result: str, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Format with nice headers and optional metrics."""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("RENODE PERIPHERAL GENERATOR - GENERATED CODE")
        lines.append("=" * 80)
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Code section
        lines.append("Generated Code:")
        lines.append("-" * 40)
        lines.append(result.strip())
        lines.append("")
        
        # Metrics section (if provided)
        if metrics:
            lines.append("Generation Metrics:")
            lines.append("-" * 40)
            
            if 'duration' in metrics:
                duration = metrics['duration']
                if duration > 60:
                    minutes = int(duration // 60)
                    seconds = duration % 60
                    duration_str = f"{minutes}m {seconds:.1f}s"
                else:
                    duration_str = f"{duration:.2f}s"
                lines.append(f"Duration: {duration_str}")
            
            if 'iterations' in metrics:
                lines.append(f"Iterations: {metrics['iterations']}")
            
            if 'cache_hits' in metrics and 'cache_misses' in metrics:
                total = metrics['cache_hits'] + metrics['cache_misses']
                hit_rate = (metrics['cache_hits'] / total * 100) if total > 0 else 0
                lines.append(f"Cache hit rate: {hit_rate:.1f}%")
            
            lines.append("")
        
        # Footer
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def format_error(self, error: str, error_type: str = "Error") -> str:
        """Format error messages consistently."""
        if self.format_type == "json":
            return json.dumps({
                "timestamp": datetime.now().isoformat(),
                "success": False,
                "error": {
                    "type": error_type,
                    "message": error
                }
            }, indent=2)
        else:
            return f"{error_type}: {error}" 