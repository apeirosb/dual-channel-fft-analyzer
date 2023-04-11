#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
context.py

To give the individual tests import context
Reference: https://docs.python-guide.org/writing/structure/
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dual_channel_fft_analyzer.dual_channel_fft_analyzer as dual_channel_fft_analyzer
