#!/usr/bin/env python3
"""
Actor Disambiguation Script
Resolves unknown actors based on context and patterns
"""

import json
import re
from typing import Dict, List, Any
from pathlib import Path

class ActorDisambiguator:
    """Disambiguates unknown actors in extraction results"""
    
    def __init__(self):
        # Define disambiguation rules
        self.disambiguation_rules = {
            'missile_launches': {
                'patterns': [
                    r'from.*lebanon',
                    r'from.*north.*blue.*line',
                    r'fired.*from.*area.*operations',
                    r'projectiles.*fired.*from'
                ],
                'default_actor': 'Hizbullah',
                'confidence': 0.8
            },
            'air_violations': {
                'patterns': [
                    r'israel.*defense.*forces',
                    r'idf',
                    r'israeli.*air',
                    r'israeli.*strikes'
                ],
                'default_actor': 'IDF',
                'confidence': 0.7
            },
            'fatalities_total': {
                'patterns': [
                    r'israeli.*strikes',
                    r'israeli.*attack',
                    r'idf.*strike',
                    r'israel.*defense.*forces'
                ],
                'default_actor': 'IDF',
                'confidence': 0.6
            },
            'fatalities_children': {
                'patterns': [
                    r'israeli.*strikes',
                    r'israeli.*attack',
                    r'civilian.*casualties'
                ],
                'default_actor': 'IDF',
                'confidence': 0.6
            },
            'displacement_total': {
                'patterns': [
                    r'israeli.*operations',
                    r'israeli.*strikes',
                    r'forced.*displacement'
                ],
                'default_actor': 'IDF',
                'confidence': 0.6
            },
            'homes_destroyed': {
                'patterns': [
                    r'israeli.*strikes',
                    r'israeli.*bombing',
                    r'residential.*buildings'
                ],
                'default_actor': 'IDF',
                'confidence': 0.7
            },
            'medical_damage': {
                'patterns': [
                    r'israeli.*strikes',
                    r'israeli.*attack',
                    r'medical.*facilities'
                ],
                'default_actor': 'IDF',
                'confidence': 0.7
            }
        }
    
    def disambiguate_actor(self, item: Dict) -> Dict:
        """Disambiguate a single data point"""
        
        if item.get('actor') != 'unknown':
            return item
        
        category = item.get('category', '')
        quote = item.get('quote', '').lower()
        
        # Check if we have rules for this category
        if category in self.disambiguation_rules:
            rules = self.disambiguation_rules[category]
            
            # Check patterns
            for pattern in rules['patterns']:
                if re.search(pattern, quote, re.IGNORECASE):
                    item['actor'] = rules['default_actor']
                    item['disambiguation_confidence'] = rules['confidence']
                    item['disambiguation_method'] = 'pattern_match'
                    return item
            
            # Apply default if no pattern matches
            item['actor'] = rules['default_actor']
            item['disambiguation_confidence'] = rules['confidence'] * 0.8  # Lower confidence for default
            item['disambiguation_method'] = 'category_default'
            return item
        
        return item
    
    def disambiguate_file(self, input_file: str, output_file: str) -> Dict:
        """Disambiguate all unknown actors in a file"""
        
        # Load data
        data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # Track statistics
        stats = {
            'total_items': len(data),
            'unknown_before': 0,
            'unknown_after': 0,
            'disambiguated': 0,
            'disambiguation_by_category': {}
        }
        
        # Process each item
        for item in data:
            if item.get('actor') == 'unknown':
                stats['unknown_before'] += 1
                
                disambiguated_item = self.disambiguate_actor(item)
                
                if disambiguated_item.get('actor') != 'unknown':
                    stats['disambiguated'] += 1
                    category = disambiguated_item.get('category', 'unknown')
                    if category not in stats['disambiguation_by_category']:
                        stats['disambiguation_by_category'][category] = 0
                    stats['disambiguation_by_category'][category] += 1
                else:
                    stats['unknown_after'] += 1
        
        # Save disambiguated data
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        return stats
    
    def generate_disambiguation_report(self, stats: Dict) -> str:
        """Generate a human-readable disambiguation report"""
        
        # Calculate disambiguation rate safely
        disambiguation_rate = 0.0
        if stats['unknown_before'] > 0:
            disambiguation_rate = (stats['disambiguated'] / stats['unknown_before'] * 100)
        
        report = f"""
# Actor Disambiguation Report

## Summary
- **Total items processed**: {stats['total_items']}
- **Unknown actors before**: {stats['unknown_before']}
- **Unknown actors after**: {stats['unknown_after']}
- **Successfully disambiguated**: {stats['disambiguated']}
- **Disambiguation rate**: {disambiguation_rate:.1f}%

## Disambiguation by Category
"""
        
        for category, count in stats['disambiguation_by_category'].items():
            report += f"- **{category}**: {count} items disambiguated\n"
        
        report += f"""
## Remaining Unknown Actors
- **Still unknown**: {stats['unknown_after']} items
"""
        
        return report

def main():
    """Main function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Disambiguate unknown actors in extraction results")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--report", type=str, help="Output report file path")
    
    args = parser.parse_args()
    
    # Initialize disambiguator
    disambiguator = ActorDisambiguator()
    
    # Process file
    print("🔍 Disambiguating unknown actors...")
    stats = disambiguator.disambiguate_file(args.input, args.output)
    
    # Generate report
    report = disambiguator.generate_disambiguation_report(stats)
    
    # Save report if requested
    if args.report:
        with open(args.report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"📋 Disambiguation report saved to: {args.report}")
    
    # Print summary
    print(f"\n🎉 Disambiguation Complete!")
    print(f"📊 Total items: {stats['total_items']}")
    print(f"📊 Unknown before: {stats['unknown_before']}")
    print(f"📊 Unknown after: {stats['unknown_after']}")
    print(f"📊 Successfully disambiguated: {stats['disambiguated']}")
    
    # Calculate disambiguation rate safely
    disambiguation_rate = 0.0
    if stats['unknown_before'] > 0:
        disambiguation_rate = (stats['disambiguated'] / stats['unknown_before'] * 100)
    print(f"📊 Disambiguation rate: {disambiguation_rate:.1f}%")
    
    if stats['disambiguation_by_category']:
        print(f"\n📈 Disambiguation by category:")
        for category, count in stats['disambiguation_by_category'].items():
            print(f"   - {category}: {count} items")

if __name__ == "__main__":
    main() 