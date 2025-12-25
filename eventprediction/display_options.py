"""
Display options for event prediction results.
"""

from dataclasses import dataclass


@dataclass
class DisplayOptions:
    """
    Options for displaying analysis results.
    
    Attributes
    ----------
    include_control : bool
        Include control arm information
    include_experimental : bool
        Include experimental arm information
    text_width : int
        Width of text output
    dp : int
        Decimal places for numeric output
    """
    include_control: bool = True
    include_experimental: bool = True
    text_width: int = 60
    dp: int = 2

