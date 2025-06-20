"""
RobotFramework Validators Module

This module provides validation functionality for RobotFramework test files
to ensure syntax correctness and best practices.
"""

import re
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path


class RFValidator:
    """Validates RobotFramework test files for syntax and best practices."""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.suggestions = []
        
    def validate_rf_content(self, content: str) -> Dict[str, Any]:
        """Validate RobotFramework test content."""
        self.errors = []
        self.warnings = []
        self.suggestions = []
        
        lines = content.split('\n')
        
        # Basic structure validation
        self._validate_basic_structure(lines)
        
        # Section validation
        self._validate_sections(lines)
        
        # Syntax validation
        self._validate_syntax(lines)
        
        # Best practices validation
        self._validate_best_practices(lines)
        
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': self.suggestions,
            'score': self._calculate_quality_score()
        }
    
    def validate_rf_file(self, file_path: str) -> Dict[str, Any]:
        """Validate a RobotFramework test file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.validate_rf_content(content)
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Failed to read file: {e}"],
                'warnings': [],
                'suggestions': [],
                'score': 0
            }
    
    def _validate_basic_structure(self, lines: List[str]) -> None:
        """Validate basic RF file structure."""
        sections_found = set()
        section_pattern = re.compile(r'^\*\*\*\s+(\w.*?)\s+\*\*\*\s*$')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            match = section_pattern.match(line)
            if match:
                section_name = match.group(1).lower()
                sections_found.add(section_name)
        
        # Check for required sections
        if not sections_found:
            self.errors.append("No sections found in RobotFramework file")
        
        # Check for common section names
        valid_sections = {
            'settings', 'variables', 'test cases', 'keywords', 'tasks'
        }
        
        for section in sections_found:
            if section not in valid_sections:
                self.warnings.append(f"Non-standard section name: '{section}'")
    
    def _validate_sections(self, lines: List[str]) -> None:
        """Validate individual sections."""
        current_section = None
        section_pattern = re.compile(r'^\*\*\*\s+(\w.*?)\s+\*\*\*\s*$')
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # Check for section headers
            match = section_pattern.match(line_stripped)
            if match:
                current_section = match.group(1).lower()
                continue
            
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            # Validate content based on current section
            if current_section == 'settings':
                self._validate_settings_line(line, line_num)
            elif current_section == 'variables':
                self._validate_variables_line(line, line_num)
            elif current_section == 'test cases':
                self._validate_test_cases_line(line, line_num)
            elif current_section == 'keywords':
                self._validate_keywords_line(line, line_num)
    
    def _validate_settings_line(self, line: str, line_num: int) -> None:
        """Validate a line in the Settings section."""
        line = line.strip()
        if not line:
            return
            
        # Common settings patterns
        settings_patterns = [
            r'^Documentation\s+',
            r'^Library\s+',
            r'^Resource\s+',
            r'^Variables\s+',
            r'^Suite Setup\s+',
            r'^Suite Teardown\s+',
            r'^Test Setup\s+',
            r'^Test Teardown\s+',
            r'^Test Tags\s+',
            r'^Default Tags\s+',
            r'^Force Tags\s+',
            r'^Test Timeout\s+',
        ]
        
        if not any(re.match(pattern, line) for pattern in settings_patterns):
            if not line.startswith('...'):  # Continuation line
                self.warnings.append(f"Line {line_num}: Unrecognized setting: '{line}'")
    
    def _validate_variables_line(self, line: str, line_num: int) -> None:
        """Validate a line in the Variables section."""
        line = line.strip()
        if not line:
            return
            
        # Variable patterns
        if not (line.startswith('${') or line.startswith('@{') or line.startswith('&{')):
            if not line.startswith('...'):  # Continuation line
                self.warnings.append(f"Line {line_num}: Invalid variable format: '{line}'")
    
    def _validate_test_cases_line(self, line: str, line_num: int) -> None:
        """Validate a line in the Test Cases section."""
        line_stripped = line.strip()
        if not line_stripped:
            return
        
        # Check indentation for test case steps
        if line.startswith('    '):  # Test case step
            # Validate test case step syntax
            step = line_stripped
            if step.startswith('[') and step.endswith(']'):
                # Test case setting like [Documentation], [Tags], etc.
                pass
            else:
                # Regular test step - check for basic keyword syntax
                if not step and not step.startswith('...'):
                    self.warnings.append(f"Line {line_num}: Empty test step")
        else:
            # Test case name - should not be indented
            if line_stripped.startswith('['):
                self.errors.append(f"Line {line_num}: Test case setting outside of test case")
    
    def _validate_keywords_line(self, line: str, line_num: int) -> None:
        """Validate a line in the Keywords section."""
        line_stripped = line.strip()
        if not line_stripped:
            return
        
        # Similar to test cases validation
        if line.startswith('    '):  # Keyword step
            step = line_stripped
            if step.startswith('[') and step.endswith(']'):
                # Keyword setting like [Documentation], [Arguments], etc.
                pass
            else:
                # Regular keyword step
                if not step and not step.startswith('...'):
                    self.warnings.append(f"Line {line_num}: Empty keyword step")
    
    def _validate_syntax(self, lines: List[str]) -> None:
        """Validate RobotFramework syntax."""
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            # Check for common syntax errors
            if '${' in line and not self._validate_variable_syntax(line):
                self.errors.append(f"Line {line_num}: Invalid variable syntax")
            
            # Check for proper spacing in sections
            if line_stripped.startswith('***') and line_stripped.endswith('***'):
                if not re.match(r'^\*\*\*\s+\w.*?\s+\*\*\*\s*$', line_stripped):
                    self.errors.append(f"Line {line_num}: Invalid section header format")
    
    def _validate_variable_syntax(self, line: str) -> bool:
        """Validate variable syntax in a line."""
        variables = re.findall(r'\$\{[^}]*\}', line)
        for var in variables:
            if not re.match(r'^\$\{[A-Za-z_][A-Za-z0-9_]*\}$', var):
                if not re.match(r'^\$\{[^}]+\}$', var):
                    return False
        return True
    
    def _validate_best_practices(self, lines: List[str]) -> None:
        """Validate RobotFramework best practices."""
        content = '\n'.join(lines)
        
        # Check for documentation
        if 'Documentation' not in content:
            self.suggestions.append("Consider adding Documentation to describe the test suite")
        
        # Check for proper test structure
        if '*** Test Cases ***' in content:
            test_cases = self._extract_test_cases(lines)
            for test_case in test_cases:
                if not test_case.get('documentation'):
                    self.suggestions.append(f"Test case '{test_case['name']}' should have [Documentation]")
                if not test_case.get('tags'):
                    self.suggestions.append(f"Test case '{test_case['name']}' should have [Tags]")
        
        # Check for setup/teardown
        if '*** Test Cases ***' in content:
            if 'Suite Setup' not in content and 'Test Setup' not in content:
                self.suggestions.append("Consider adding Suite Setup or Test Setup for initialization")
            if 'Suite Teardown' not in content and 'Test Teardown' not in content:
                self.suggestions.append("Consider adding Suite Teardown or Test Teardown for cleanup")
        
        # Check for consistent spacing
        self._check_spacing_consistency(lines)
    
    def _extract_test_cases(self, lines: List[str]) -> List[Dict[str, Any]]:
        """Extract test cases from the content."""
        test_cases = []
        current_test = None
        in_test_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            if line_stripped == '*** Test Cases ***':
                in_test_section = True
                continue
            elif line_stripped.startswith('***') and line_stripped.endswith('***'):
                in_test_section = False
                if current_test:
                    test_cases.append(current_test)
                    current_test = None
                continue
            
            if not in_test_section:
                continue
            
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            if not line.startswith('    '):  # Test case name
                if current_test:
                    test_cases.append(current_test)
                current_test = {
                    'name': line_stripped,
                    'documentation': False,
                    'tags': False,
                    'steps': []
                }
            else:  # Test case content
                if current_test:
                    if line_stripped.startswith('[Documentation]'):
                        current_test['documentation'] = True
                    elif line_stripped.startswith('[Tags]'):
                        current_test['tags'] = True
                    else:
                        current_test['steps'].append(line_stripped)
        
        if current_test:
            test_cases.append(current_test)
        
        return test_cases
    
    def _check_spacing_consistency(self, lines: List[str]) -> None:
        """Check for consistent spacing in the file."""
        indentation_levels = set()
        
        for line in lines:
            if line.strip() and not line.strip().startswith('#'):
                leading_spaces = len(line) - len(line.lstrip())
                if leading_spaces > 0:
                    indentation_levels.add(leading_spaces)
        
        # Check if indentation is consistent (should be multiples of 4)
        for level in indentation_levels:
            if level % 4 != 0:
                self.suggestions.append(f"Consider using 4-space indentation consistently (found {level} spaces)")
                break
    
    def _calculate_quality_score(self) -> int:
        """Calculate a quality score (0-100) based on validation results."""
        base_score = 100
        
        # Deduct points for errors (major issues)
        base_score -= len(self.errors) * 20
        
        # Deduct points for warnings (minor issues)
        base_score -= len(self.warnings) * 5
        
        # Minor deductions for suggestions
        base_score -= len(self.suggestions) * 2
        
        return max(0, min(100, base_score))
    
    def get_validation_summary(self) -> str:
        """Get a formatted validation summary."""
        total_issues = len(self.errors) + len(self.warnings) + len(self.suggestions)
        score = self._calculate_quality_score()
        
        summary = [
            f"Validation Summary:",
            f"  Quality Score: {score}/100",
            f"  Total Issues: {total_issues}",
            f"  Errors: {len(self.errors)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Suggestions: {len(self.suggestions)}"
        ]
        
        if self.errors:
            summary.extend(["", "Errors:"])
            for error in self.errors:
                summary.append(f"  ❌ {error}")
        
        if self.warnings:
            summary.extend(["", "Warnings:"])
            for warning in self.warnings:
                summary.append(f"  ⚠️  {warning}")
        
        if self.suggestions:
            summary.extend(["", "Suggestions:"])
            for suggestion in self.suggestions:
                summary.append(f"  💡 {suggestion}")
        
        return '\n'.join(summary) 