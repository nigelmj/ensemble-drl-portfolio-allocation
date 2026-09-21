"""Filesystem helpers."""

from __future__ import annotations

import os


def check_and_make_directories(directories: list[str]):
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
