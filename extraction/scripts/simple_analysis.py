#!/usr/bin/env python3
"""
Simple Analysis Script for GPT Extraction Results
Provides text-based analysis without visualization dependencies
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

def analyze_extraction_results(input_file: str) -> Dict:
    """Analyze extraction results and return comprehensive statistics"""
    
    # Load data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        return {"error": "No data to analyze"}
    
    # Basic statistics
    total_data_points = len(data)
    
    # Category analysis
    categories = [item.get('category', 'unknown') for item in data]
    category_counts = Counter(categories)
    
    # Actor analysis
    actors = [item.get('actor', 'unknown') for item in data]
    actor_counts = Counter(actors)
    
    # Value analysis
    numeric_values = []
    for item in data:
        value = item.get('value')
        if value is not None and isinstance(value, (int, float)):
            numeric_values.append(value)
    
    value_stats = {
        'total': sum(numeric_values) if numeric_values else 0,
        'mean': sum(numeric_values) / len(numeric_values) if numeric_values else 0,
        'min': min(numeric_values) if numeric_values else 0,
        'max': max(numeric_values) if numeric_values else 0,
        'count': len(numeric_values)
    }
    
    # Confidence analysis
    confidences = [item.get('confidence_score', 0) for item in data]
    confidence_stats = {
        'mean': sum(confidences) / len(confidences) if confidences else 0,
        'min': min(confidences) if confidences else 0,
        'max': max(confidences) if confidences else 0,
        'count': len(confidences)
    }
    
    # UNSCR 1701 legal mapping analysis
    legal_mappings = []
    for item in data:
        legal_article = item.get('legal_article_violated', 'unknown')
        # Ensure UNSCR 1701 format
        if legal_article and legal_article != 'unknown':
            if 'UNSCR 1701' not in legal_article:
                legal_article = f"UNSCR 1701 (2006), {legal_article}"
        legal_mappings.append(legal_article)
    legal_counts = Counter(legal_mappings)
    
    # Top incidents by value
    def get_numeric_value(item):
        value = item.get('value')
        if value is not None and isinstance(value, (int, float)):
            return value
        return 0
    
    sorted_by_value = sorted(data, key=get_numeric_value, reverse=True)
    top_incidents = sorted_by_value[:10]
    
    # Category insights
    category_insights = {}
    for category in set(categories):
        category_data = [item for item in data if item.get('category') == category]
        category_values = []
        for item in category_data:
            value = item.get('value')
            if value is not None and isinstance(value, (int, float)):
                category_values.append(value)
        category_actors = [item.get('actor', 'unknown') for item in category_data]
        
        category_insights[category] = {
            'count': len(category_data),
            'total_value': sum(category_values),
            'avg_value': sum(category_values) / len(category_values) if category_values else 0,
            'actors': Counter(category_actors),
            'top_incident': max(category_data, key=get_numeric_value) if category_data else None
        }
    
    return {
        'summary': {
            'total_data_points': total_data_points,
            'unique_categories': len(category_counts),
            'unique_actors': len(actor_counts),
            'average_confidence': confidence_stats['mean']
        },
        'category_distribution': dict(category_counts),
        'actor_distribution': dict(actor_counts),
        'value_statistics': value_stats,
        'confidence_statistics': confidence_stats,
        'legal_mapping_distribution': dict(legal_counts),
        'top_incidents': top_incidents,
        'category_insights': category_insights
    }

def print_analysis_report(analysis: Dict):
    """Print a comprehensive analysis report"""
    
    print("🎉 GPT-4o Quantitative Extraction Analysis Report")
    print("=" * 60)
    
    # Summary
    summary = analysis['summary']
    print(f"\n📊 SUMMARY STATISTICS")
    print(f"   Total data points extracted: {summary['total_data_points']}")
    print(f"   Unique categories found: {summary['unique_categories']}")
    print(f"   Unique actors identified: {summary['unique_actors']}")
    print(f"   Average confidence score: {summary['average_confidence']:.3f}")
    
    # Category distribution
    print(f"\n📈 CATEGORY DISTRIBUTION")
    for category, count in analysis['category_distribution'].items():
        print(f"   {category}: {count} data points")
    
    # Actor distribution
    print(f"\n👥 ACTOR DISTRIBUTION")
    for actor, count in analysis['actor_distribution'].items():
        print(f"   {actor}: {count} mentions")
    
    # Value statistics
    value_stats = analysis['value_statistics']
    print(f"\n💰 VALUE STATISTICS")
    print(f"   Total value across all data: {value_stats['total']:,}")
    print(f"   Average value: {value_stats['mean']:.1f}")
    print(f"   Value range: {value_stats['min']:,} - {value_stats['max']:,}")
    
    # Top incidents
    print(f"\n🔥 TOP 10 INCIDENTS BY VALUE")
    for i, incident in enumerate(analysis['top_incidents'], 1):
        print(f"   {i:2d}. {incident['category']}: {incident['value']:,} {incident.get('unit', '')} by {incident['actor']}")
    
    # UNSCR 1701 Legal mapping distribution
    print(f"\n⚖️ UNSCR 1701 LEGAL MAPPING DISTRIBUTION")
    for legal_article, count in analysis['legal_mapping_distribution'].items():
        if legal_article != 'unknown':
            print(f"   {legal_article}: {count} violations")
        else:
            print(f"   {legal_article}: {count} data points")
    
    # Category insights
    print(f"\n🔍 CATEGORY INSIGHTS")
    for category, insights in analysis['category_insights'].items():
        print(f"\n   📊 {category.upper()}")
        print(f"      Count: {insights['count']}")
        print(f"      Total value: {insights['total_value']:,}")
        print(f"      Average value: {insights['avg_value']:.1f}")
        print(f"      Top actor: {insights['actors'].most_common(1)[0][0] if insights['actors'] else 'None'}")
        if insights['top_incident']:
            top = insights['top_incident']
            value = top.get('value', 0) or 0
            # Ensure value is numeric for formatting
            try:
                numeric_value = float(value) if value is not None else 0
                formatted_value = f"{numeric_value:,.0f}" if numeric_value.is_integer() else f"{numeric_value:,.1f}"
            except (ValueError, TypeError):
                formatted_value = str(value) if value is not None else "0"
            unit = top.get('unit', '')
            print(f"      Largest incident: {formatted_value} {unit} by {top['actor']}")
    
    # Data quality assessment
    print(f"\n✅ DATA QUALITY ASSESSMENT")
    unknown_actors = analysis['actor_distribution'].get('unknown', 0)
    total_actors = sum(analysis['actor_distribution'].values())
    actor_completeness = (total_actors - unknown_actors) / total_actors if total_actors > 0 else 0
    
    print(f"   Actor attribution completeness: {actor_completeness:.1%}")
    print(f"   Average confidence score: {summary['average_confidence']:.1%}")
    
    if actor_completeness < 0.8:
        print(f"   ⚠️  Recommendation: Consider actor disambiguation for {unknown_actors} unknown actors")
    
    if summary['average_confidence'] < 0.8:
        print(f"   ⚠️  Recommendation: Review low-confidence extractions")

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple analysis of GPT extraction results")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, help="Output JSON file path for analysis")
    
    args = parser.parse_args()
    
    # Analyze results
    print("🔍 Analyzing extraction results...")
    analysis = analyze_extraction_results(args.input)
    
    # Print report
    print_analysis_report(analysis)
    
    # Save analysis if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Analysis saved to: {args.output}")

if __name__ == "__main__":
    main() 