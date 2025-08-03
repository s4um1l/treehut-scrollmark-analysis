#!/usr/bin/env python3
"""
Instagram Scrollmark Analysis POC
Main entry point for the application.
"""

import sys
from typing import Optional


def main(args: Optional[list] = None) -> int:
    """Main function to run the Instagram scrollmark analysis.

    Args:
        args: Command line arguments (defaults to sys.argv[1:])

    Returns:
        Exit code (0 for success)
    """
    if args is None:
        args = sys.argv[1:]

    print("🚀 Instagram Scrollmark Analysis POC")  
    print(f"📍 Python version: {sys.version.split()[0]}")
    print("✅ Analysis pipeline ready!")

    if args:
        print(f"📝 Command line arguments: {args}")

    print("\n🔍 Available Analysis Components:")
    print("  - Exploratory Data Analysis (EDA)")
    print("  - Parallel Data Enrichment Pipeline") 
    print("  - Advanced Trend Analysis")
    print("  - Interactive Dashboard Server")
    print("  - Automated Alerting System")
    print("\n💡 Use ./scripts/run_pipeline.sh to execute full analysis")

    return 0


if __name__ == "__main__":
    sys.exit(main()) 