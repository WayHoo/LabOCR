import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)

from _fuzzy_chinese_match import FuzzyChineseMatch
from _character_to_stroke import Stroke

import logging
log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.WARNING)
default_logger.addHandler(log_console)

__all__ = ['FuzzyChineseMatch', 'Stroke']
