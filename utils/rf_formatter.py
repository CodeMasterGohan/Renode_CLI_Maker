"""
RobotFramework Output Formatter

This module provides specialized formatting for RobotFramework test generation output.
"""

import json
from typing import Dict, Any, Optional
from .formatter import OutputFormatter


class RFOutputFormatter(OutputFormatter):
    """Enhanced output formatter for RobotFramework test generation."""
    
    def format_rf_output(self, result: Dict[str, Any], format_type: str = 'pretty') -> str:
        """Format RobotFramework generation result."""
        
        if format_type == 'json':
            return self._format_json_output(result)
        elif format_type == 'raw':
            return result.get('rf_code', '')
        else:  # pretty
            return self._format_pretty_output(result)
    
    def _format_pretty_output(self, result: Dict[str, Any]) -> str:
        """Format pretty output for RF generation."""
        
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append("🤖 ROBOTFRAMEWORK TEST GENERATION RESULT")
        lines.append("=" * 80)
        
        # Basic info
        peripheral_name = result.get('peripheral_name', 'Unknown')
        test_level = result.get('test_level', 'Unknown')
        
        lines.append("")
        lines.append(f"Peripheral: {peripheral_name}")
        lines.append(f"Test Level: {test_level.title()}")
        
        # Validation info
        validation = result.get('validation', {})
        score = validation.get('score', 0)
        lines.append(f"Validation Score: {score}/100")
        
        # RF Code section
        rf_code = result.get('rf_code', '')
        if rf_code:
            lines.append("")
            lines.append("🤖 ROBOTFRAMEWORK TEST CODE")
            lines.append("=" * 80)
            lines.append("")
            lines.append(rf_code)
        
        # Footer
        lines.append("")
        lines.append("=" * 80)
        lines.append("💡 TIP: Run 'robot <test_file>' to execute the generated tests")
        lines.append("=" * 80)
        
        return '\n'.join(lines)
    
    def _format_json_output(self, result: Dict[str, Any]) -> str:
        """Format JSON output for RF generation."""
        
        json_result = {
            'type': 'robotframework_test_generation',
            'peripheral_name': result.get('peripheral_name'),
            'test_level': result.get('test_level'),
            'validation': result.get('validation', {}),
            'rf_code': result.get('rf_code', ''),
            'generated_at': self._get_timestamp(),
            'success': result.get('validation', {}).get('valid', True)
        }
        
        return json.dumps(json_result, indent=2, ensure_ascii=False)