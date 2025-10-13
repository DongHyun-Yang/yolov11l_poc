#!/usr/bin/env python3
"""
Custom ultralytics import utility module
This module should be imported before any ultralytics imports to ensure
the custom ultralytics library from lib/ultralytics/ultralytics is used
instead of the site-packages version.

Usage:
    import custom_ultralytics_setup  # Import this first
    from ultralytics import YOLO      # Now this will use custom ultralytics
"""
import os
import sys

def setup_custom_ultralytics_path():
    """
    Setup custom ultralytics path in sys.path
    This function ensures that the local ultralytics library is used
    instead of the site-packages version.
    """
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, os.path.pardir)
    
    # Build path to custom ultralytics library
    ultralytics_lib_path = os.path.join(project_root, 'lib', 'ultralytics')
    
    # Check if the path exists
    if os.path.exists(ultralytics_lib_path):
        # Insert at the beginning of sys.path to prioritize over site-packages
        if ultralytics_lib_path not in sys.path:
            sys.path.insert(0, ultralytics_lib_path)
            print(f"Added custom ultralytics path: {ultralytics_lib_path}")
    else:
        print(f"Warning: Custom ultralytics path not found: {ultralytics_lib_path}")

# Automatically setup when this module is imported
setup_custom_ultralytics_path()