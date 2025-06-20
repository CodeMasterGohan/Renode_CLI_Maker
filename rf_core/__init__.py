"""
RobotFramework Core Module for Renode Peripheral Generator

This module provides RobotFramework-specific functionality for generating
test suites for Renode peripherals.
"""

from .rf_application import RFGeneratorCLI
from .rf_templates import RFTemplateManager
from .rf_validators import RFValidator

__all__ = ['RFGeneratorCLI', 'RFTemplateManager', 'RFValidator'] 