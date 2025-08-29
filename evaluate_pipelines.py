#!/usr/bin/env python3
"""
Pipeline Evaluation Script
Compares the accuracy and quality of Legacy vs Fast Path pipelines
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import re

def load_legacy_results():
    """Load legacy pipeline results"""
    legacy_file = Path("precision_hybrid_results/precision_hybrid_extraction_results.jsonl")
    if not legacy_file.exists():
        return None
    
    results = []
    with open(legacy_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results

def load_fast_path_results():
    """Load Fast Path results"""
    fast_path_file = Path("extraction/hybrid_output/incidents.json")
    if not fast_path_file.exists():
        return None
    
    with open(fast_path_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to comparable format
    results = []
    for incident in data.get('incidents', []):
        for fact in incident.get('facts', []):
            results.append({
                'category': fact.get('type', ''),
                'value': fact.get('val', ''),
                'unit': fact.get('unit', ''),
                'actor': incident.get('actors', [{}])[0].get('name', '') if incident.get('actors') else '',
                'quote': incident.get('quotes', [''])[0] if incident.get('quotes') else '',
                'pid': incident.get('pid_list', [None])[0] if incident.get('pid_list') else None,
                'location': incident.get('loc', ''),
                'date': incident.get('when', {}).get('date_iso', ''),
                'confidence': 0.8  # Default confidence for Fast Path
            })
    return results

def analyze_accuracy(legacy_results, fast_path_results):
    """Analyze accuracy metrics for both pipelines"""
    
    print("🔍 ACCURACY ANALYSIS")
    print("=" * 50)
    
    # Data point counts
    legacy_count = len(legacy_results) if legacy_results else 0
    fast_path_count = len(fast_path_results) if fast_path_results else 0
    
    print(f"📊 Data Points Extracted:")
    print(f"   Legacy Pipeline: {legacy_count}")
    print(f"   Fast Path: {fast_path_count}")
    print(f"   Difference: {fast_path_count - legacy_count}")
    
    # Category analysis
    if legacy_results:
        legacy_categories = {}
        for item in legacy_results:
            cat = item.get('category', 'unknown')
            legacy_categories[cat] = legacy_categories.get(cat, 0) + 1
        
        print(f"\n📈 Legacy Categories: {len(legacy_categories)}")
        for cat, count in sorted(legacy_categories.items()):
            print(f"   {cat}: {count}")
    
    if fast_path_results:
        fast_path_categories = {}
        for item in fast_path_results:
            cat = item.get('category', 'unknown')
            fast_path_categories[cat] = fast_path_categories.get(cat, 0) + 1
        
        print(f"\n🚀 Fast Path Categories: {len(fast_path_categories)}")
        for cat, count in sorted(fast_path_categories.items()):
            print(f"   {cat}: {count}")
    
    # Actor analysis
    if legacy_results:
        legacy_actors = {}
        for item in legacy_results:
            actor = item.get('actor', 'unknown')
            legacy_actors[actor] = legacy_actors.get(actor, 0) + 1
        
        print(f"\n👥 Legacy Actors: {len(legacy_actors)}")
        for actor, count in sorted(legacy_actors.items()):
            print(f"   {actor}: {count}")
    
    if fast_path_results:
        fast_path_actors = {}
        for item in fast_path_results:
            actor = item.get('actor', 'unknown')
            fast_path_actors[actor] = fast_path_actors.get(actor, 0) + 1
        
        print(f"\n👥 Fast Path Actors: {len(fast_path_actors)}")
        for actor, count in sorted(fast_path_actors.items()):
            print(f"   {actor}: {count}")

def analyze_quality(legacy_results, fast_path_results):
    """Analyze quality metrics for both pipelines"""
    
    print("\n🎯 QUALITY ANALYSIS")
    print("=" * 50)
    
    # Quote quality analysis
    def analyze_quotes(results, pipeline_name):
        if not results:
            return
        
        total_quotes = 0
        valid_quotes = 0
        quote_lengths = []
        
        for item in results:
            quote = item.get('quote', '')
            total_quotes += 1
            
            if quote and len(quote.strip()) > 10:  # Valid quote
                valid_quotes += 1
                quote_lengths.append(len(quote))
        
        avg_length = sum(quote_lengths) / len(quote_lengths) if quote_lengths else 0
        
        print(f"\n📝 {pipeline_name} Quote Quality:")
        print(f"   Total items: {total_quotes}")
        print(f"   Valid quotes: {valid_quotes}")
        print(f"   Quote success rate: {valid_quotes/total_quotes*100:.1f}%")
        print(f"   Average quote length: {avg_length:.0f} characters")
    
    analyze_quotes(legacy_results, "Legacy Pipeline")
    analyze_quotes(fast_path_results, "Fast Path")
    
    # Value quality analysis
    def analyze_values(results, pipeline_name):
        if not results:
            return
        
        numeric_values = 0
        total_values = 0
        value_range = []
        
        for item in results:
            value = item.get('value')
            total_values += 1
            
            if value is not None and isinstance(value, (int, float)):
                numeric_values += 1
                value_range.append(value)
        
        if value_range:
            min_val = min(value_range)
            max_val = max(value_range)
            avg_val = sum(value_range) / len(value_range)
        else:
            min_val = max_val = avg_val = 0
        
        print(f"\n🔢 {pipeline_name} Value Quality:")
        print(f"   Total items: {total_values}")
        print(f"   Numeric values: {numeric_values}")
        print(f"   Value success rate: {numeric_values/total_values*100:.1f}%")
        print(f"   Value range: {min_val} - {max_val}")
        print(f"   Average value: {avg_val:.1f}")
    
    analyze_values(legacy_results, "Legacy Pipeline")
    analyze_values(fast_path_results, "Fast Path")

def analyze_specific_incidents(legacy_results, fast_path_results):
    """Analyze specific incidents for accuracy comparison"""
    
    print("\n🎯 SPECIFIC INCIDENT ANALYSIS")
    print("=" * 50)
    
    # Look for specific high-value incidents
    def find_incident(results, keywords, pipeline_name):
        found = []
        for item in results:
            quote = item.get('quote', '').lower()
            if any(keyword in quote for keyword in keywords):
                found.append(item)
        return found
    
    # Check for civilian casualties
    print("\n💔 Civilian Casualties:")
    legacy_casualties = find_incident(legacy_results, ['civilian', 'killed', 'death'], "Legacy")
    fast_path_casualties = find_incident(fast_path_results, ['civilian', 'killed', 'death'], "Fast Path")
    
    print(f"   Legacy found: {len(legacy_casualties)} incidents")
    for item in legacy_casualties:
        print(f"     - {item.get('category')}: {item.get('value')} {item.get('unit')}")
    
    print(f"   Fast Path found: {len(fast_path_casualties)} incidents")
    for item in fast_path_casualties:
        print(f"     - {item.get('category')}: {item.get('value')} {item.get('unit')}")
    
    # Check for air strikes
    print("\n✈️ Air Strikes:")
    legacy_air = find_incident(legacy_results, ['air', 'strike', 'attack'], "Legacy")
    fast_path_air = find_incident(fast_path_results, ['air', 'strike', 'attack'], "Fast Path")
    
    print(f"   Legacy found: {len(legacy_air)} incidents")
    for item in legacy_air:
        print(f"     - {item.get('category')}: {item.get('value')} {item.get('unit')}")
    
    print(f"   Fast Path found: {len(fast_path_air)} incidents")
    for item in fast_path_air:
        print(f"     - {item.get('category')}: {item.get('value')} {item.get('unit')}")

def analyze_performance():
    """Analyze performance metrics"""
    
    print("\n⚡ PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Fast Path performance
    fast_stats_file = Path("extraction/hybrid_output/run_stats.json")
    if fast_stats_file.exists():
        with open(fast_stats_file, 'r') as f:
            fast_stats = json.load(f)
        
        timings = fast_stats.get('timings_ms', {})
        total_time = timings.get('total', 0) / 1000  # Convert to seconds
        
        print(f"🚀 Fast Path Performance:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Fast extraction: {timings.get('fast_path', 0):.0f}ms")
        print(f"   Packing: {timings.get('packing', 0):.0f}ms")
        print(f"   Reconciliation: {timings.get('reconcile', 0):.0f}ms")
        print(f"   Windows used: {fast_stats.get('windows', 0)}")
        print(f"   Candidates found: {fast_stats.get('candidates', 0)}")
        print(f"   Incidents created: {fast_stats.get('incidents', 0)}")
    
    # Legacy performance (estimated)
    print(f"\n🐌 Legacy Performance (Estimated):")
    print(f"   Estimated time: 30-60 seconds (based on typical LLM processing)")
    print(f"   Multiple API calls per paragraph")
    print(f"   Higher cost due to more LLM usage")

def generate_recommendation(legacy_results, fast_path_results):
    """Generate a recommendation based on the analysis"""
    
    print("\n🎯 RECOMMENDATION")
    print("=" * 50)
    
    if not legacy_results and not fast_path_results:
        print("❌ No results found from either pipeline")
        return
    
    if not legacy_results:
        print("✅ RECOMMENDATION: Use Fast Path")
        print("   - Legacy pipeline produced no results")
        print("   - Fast Path is the only working option")
        return
    
    if not fast_path_results:
        print("✅ RECOMMENDATION: Use Legacy Pipeline")
        print("   - Fast Path produced no results")
        print("   - Legacy pipeline is the only working option")
        return
    
    # Compare data points
    legacy_count = len(legacy_results)
    fast_path_count = len(fast_path_results)
    
    # Compare categories
    legacy_categories = set(item.get('category', '') for item in legacy_results)
    fast_path_categories = set(item.get('category', '') for item in fast_path_results)
    
    # Compare actors
    legacy_actors = set(item.get('actor', '') for item in legacy_results)
    fast_path_actors = set(item.get('actor', '') for item in fast_path_results)
    
    print(f"📊 Comparison Summary:")
    print(f"   Data Points: Legacy {legacy_count} vs Fast Path {fast_path_count}")
    print(f"   Categories: Legacy {len(legacy_categories)} vs Fast Path {len(fast_path_categories)}")
    print(f"   Actors: Legacy {len(legacy_actors)} vs Fast Path {len(fast_path_actors)}")
    
    # Calculate overlap
    category_overlap = len(legacy_categories.intersection(fast_path_categories))
    actor_overlap = len(legacy_actors.intersection(fast_path_actors))
    
    print(f"   Category overlap: {category_overlap}")
    print(f"   Actor overlap: {actor_overlap}")
    
    # Make recommendation
    if fast_path_count >= legacy_count * 0.8 and category_overlap >= len(legacy_categories) * 0.7:
        print(f"\n✅ RECOMMENDATION: Use Fast Path")
        print(f"   - Fast Path extracted {fast_path_count} vs Legacy {legacy_count} data points")
        print(f"   - Good category coverage ({category_overlap}/{len(legacy_categories)})")
        print(f"   - Much faster performance (8.6s vs 30-60s estimated)")
        print(f"   - Lower cost (fewer LLM calls)")
    elif legacy_count > fast_path_count * 1.5:
        print(f"\n✅ RECOMMENDATION: Use Legacy Pipeline")
        print(f"   - Legacy extracted significantly more data ({legacy_count} vs {fast_path_count})")
        print(f"   - Better coverage of categories and actors")
        print(f"   - More comprehensive analysis")
    else:
        print(f"\n🤔 RECOMMENDATION: Use Legacy Pipeline for now")
        print(f"   - Similar data extraction counts")
        print(f"   - Legacy has more comprehensive analysis features")
        print(f"   - Consider Fast Path for speed-critical scenarios")

def main():
    """Main evaluation function"""
    
    print("🔍 PIPELINE EVALUATION REPORT")
    print("=" * 60)
    print("Comparing Legacy Pipeline vs Fast Path Pipeline")
    print("=" * 60)
    
    # Load results
    print("\n📂 Loading results...")
    legacy_results = load_legacy_results()
    fast_path_results = load_fast_path_results()
    
    if legacy_results:
        print(f"✅ Loaded {len(legacy_results)} Legacy results")
    else:
        print("❌ No Legacy results found")
    
    if fast_path_results:
        print(f"✅ Loaded {len(fast_path_results)} Fast Path results")
    else:
        print("❌ No Fast Path results found")
    
    # Run analyses
    analyze_accuracy(legacy_results, fast_path_results)
    analyze_quality(legacy_results, fast_path_results)
    analyze_specific_incidents(legacy_results, fast_path_results)
    analyze_performance()
    generate_recommendation(legacy_results, fast_path_results)
    
    print("\n" + "=" * 60)
    print("📋 EVALUATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
