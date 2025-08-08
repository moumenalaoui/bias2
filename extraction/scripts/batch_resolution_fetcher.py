#!/usr/bin/env python3
"""
Batch Resolution Fetcher - Simplified version
Since we have the key resolutions (1701, 1559, etc.) already cached,
this script now just validates the cache exists.

Usage:
    python batch_resolution_fetcher.py
"""

import os
import json
from pathlib import Path

def main():
    """Main function - validate resolution cache exists"""
    # Get the script directory and find resolution_cache relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    cache_dir = project_root / "resolution_cache"
    
    print("🔍 Checking resolution cache...")
    
    if not cache_dir.exists():
        print("❌ Resolution cache directory not found")
        return 1
    
    # Count cached resolutions
    cached_files = list(cache_dir.glob("*.json"))
    
    if len(cached_files) == 0:
        print("⚠️  No cached resolutions found")
        return 1
    
    print(f"✅ Found {len(cached_files)} cached resolutions")
    
    # List the cached resolutions
    for cache_file in sorted(cached_files):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                resolution_num = data.get('resolution_number', 'Unknown')
                print(f"   📄 Resolution {resolution_num}")
        except Exception as e:
            print(f"   ⚠️  Error reading {cache_file.name}: {e}")
    
    print("✅ Resolution cache validation completed")
    return 0

if __name__ == "__main__":
    exit(main())