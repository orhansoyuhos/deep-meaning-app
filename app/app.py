#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Desktop entry point for the DeepMeaning app."""

from __future__ import annotations

from multiprocessing import freeze_support

from deepmeaning.gui import launch_app


if __name__ == "__main__":
    freeze_support()
    launch_app()
