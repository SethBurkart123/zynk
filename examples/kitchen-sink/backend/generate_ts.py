#!/usr/bin/env python3
"""Script to generate the TypeScript client for the kitchen sink example via the zynk CLI."""

import os
import sys

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import subprocess

output_path = "../frontend/src/generated/api.ts"
subprocess.run(
    [
        "zynk",
        "gen",
        "typescript",
        "--target",
        "python",
        "--out",
        output_path,
        "--app",
        "main:app",
    ],
    cwd=os.path.dirname(__file__),
    check=True,
)
print(f"Generated TypeScript client: {output_path}")
