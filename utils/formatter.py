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
        
    def format(self, result, metrics: Optional[Dict[str, Any]] = None) -> str:
        """Format the result according to the specified format type."""
        
        # Handle both string results (legacy) and dict results (new multi-mode)
        if isinstance(result, str):
            result_data = {'peripheral_code': result, 'mode': 'peripheral'}
        else:
            result_data = result
        
        if self.format_type == "raw":
            return self._format_raw(result_data)
        elif self.format_type == "json":
            return self._format_json(result_data, metrics)
        else:  # pretty (default)
            return self._format_pretty(result_data, metrics)
    
    def _format_raw(self, result_data: Dict[str, Any]) -> str:
        """Return just the raw code without any formatting."""
        mode = result_data.get('mode', 'peripheral')
        
        if mode == 'peripheral':
            return result_data.get('peripheral_code', '').strip()
        elif mode == 'robotframework':
            rf_result = result_data.get('rf_tests', {})
            return rf_result.get('rf_code', '').strip()
        elif mode == 'both':
            # For both mode, return both sections
            output = []
            if 'peripheral_code' in result_data:
                output.append("# C# Peripheral Code")
                output.append(result_data['peripheral_code'].strip())
            if 'rf_tests' in result_data:
                output.append("\n# RobotFramework Tests")
                output.append(result_data['rf_tests'].get('rf_code', '').strip())
            return '\n\n'.join(output)
        
        return str(result_data).strip()
    
    def _format_json(self, result_data: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> str:
        """Format as JSON for machine consumption."""
        output = {
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "mode": result_data.get('mode', 'peripheral')
        }
        
        # Add the actual results
        if 'peripheral_code' in result_data:
            output['peripheral_code'] = result_data['peripheral_code']
        if 'rf_tests' in result_data:
            output['rf_tests'] = result_data['rf_tests']
        
        if metrics:
            output["metrics"] = metrics
            
        return json.dumps(output, indent=2)
    
    def _format_pretty(self, result_data: Dict[str, Any], metrics: Optional[Dict[str, Any]] = None) -> str:
        """Format with nice headers and optional metrics."""
        lines = []
        mode = result_data.get('mode', 'peripheral')
        
        # Header
        lines.append("=" * 80)
        lines.append(f"RENODE GENERATOR - {mode.upper()} MODE")
        lines.append("=" * 80)
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Mode: {mode}")
        lines.append("")
        
        # Add peripheral code section if present
        if 'peripheral_code' in result_data:
            lines.append("🔧 C# PERIPHERAL CODE")
            lines.append("-" * 40)
            lines.append(result_data['peripheral_code'].strip())
            lines.append("")
        
        # Add RF tests section if present
        if 'rf_tests' in result_data:
            rf_data = result_data['rf_tests']
            lines.append("🤖 ROBOTFRAMEWORK TESTS")
            lines.append("-" * 40)
            
            # Add RF metadata
            if 'peripheral_name' in rf_data:
                lines.append(f"Peripheral: {rf_data['peripheral_name']}")
            if 'test_level' in rf_data:
                lines.append(f"Test Level: {rf_data['test_level'].title()}")
            if 'validation' in rf_data:
                score = rf_data['validation'].get('score', 0)
                lines.append(f"Validation Score: {score}/100")
            lines.append("")
            
            # Add the actual RF code
            lines.append(rf_data.get('rf_code', '').strip())
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