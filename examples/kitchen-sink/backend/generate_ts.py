#!/usr/bin/env python3
"""
Script to generate the TypeScript client for the kitchen sink example.
"""

import os
import sys

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the command modules to register their commands
import tasks
import users
import weather

# Generate the TypeScript
from zynk import generate_typescript

output_path = "../frontend/src/generated/api.ts"
generate_typescript(output_path)
print(f"Generated TypeScript client: {output_path}")
