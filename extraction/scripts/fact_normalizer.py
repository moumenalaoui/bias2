#!/usr/bin/env python3
"""
Fact Normalizer - LLM-powered data quality enhancement
Handles: multi-number explosion, LLM-based actor attribution, data cleaning
"""

import re
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import openai
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FactNormalizer:
    """LLM-powered fact normalization for maximum data quality"""
    
    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # Initialize OpenAI client
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        # Multi-number detection patterns (keep these - they're still useful)
        self.multi_number_patterns = [
            r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s+(?:and|&)\s+(\d{1,3}(?:,\d{3})*(?:\.\d+)?)',  # "1,295 and 1,331"
            r'(\d+)\s+(?:and|&)\s+(\d+)',  # "194 and 6"
        ]
        
        # Basic actor patterns (for obvious cases only)
        self.basic_actor_patterns = {
            r'\bIDF\b': 'Israel Defense Forces',
            r'\bUNIFIL\b': 'United Nations Interim Force in Lebanon',
            r'\bLAF\b': 'Lebanese Armed Forces',
        }
    
    def normalize_facts(self, facts_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Main normalization pipeline - applies all fixes to extracted facts
        """
        logger.info(f"Starting fact normalization for {len(facts_data)} facts")
        
        normalized_facts = []
        
        for fact in facts_data:
            # Apply all normalization steps
            normalized_fact = self._normalize_single_fact(fact)
            
            # Handle multi-number explosion
            if self._has_multiple_numbers(normalized_fact):
                exploded_facts = self._explode_multi_numbers(normalized_fact)
                normalized_facts.extend(exploded_facts)
            else:
                normalized_facts.append(normalized_fact)
        
        # Final data cleaning
        normalized_facts = self._clean_data(normalized_facts)
        
        logger.info(f"Normalization complete: {len(facts_data)} → {len(normalized_facts)} facts")
        return normalized_facts
    
    def _normalize_single_fact(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a single fact with all fixes"""
        normalized = fact.copy()
        
        # 1. LLM-based actor attribution (the main improvement)
        normalized = self._resolve_actor_with_llm(normalized)
        
        # 2. Clean and normalize values
        normalized = self._normalize_values(normalized)
        
        # 3. Basic actor normalization (for obvious cases)
        normalized = self._apply_basic_actor_patterns(normalized)
        
        return normalized
    
    def _resolve_actor_with_llm(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to resolve actor attribution based on context"""
        quote = fact.get('quote', '')
        context = fact.get('context', '')
        category = fact.get('category', '')
        current_actor = fact.get('actor', '')
        
        # Use LLM for ALL actor resolution (no exceptions)
        try:
            resolved_actor = self._call_llm_for_actor(quote, context, category)
            if resolved_actor and resolved_actor.lower() not in ['unknown', 'none', 'n/a']:
                fact['actor'] = resolved_actor
                fact['actor_resolution_method'] = 'llm_context_analysis'
                fact['actor_normalized'] = True
                logger.info(f"LLM resolved actor: {resolved_actor} for category: {category}")
            else:
                fact['actor_resolution_method'] = 'llm_no_clear_actor'
        except Exception as e:
            logger.warning(f"LLM actor resolution failed: {e}")
            fact['actor_resolution_method'] = 'llm_failed'
        
        return fact
    
    def _call_llm_for_actor(self, quote: str, context: str, category: str) -> str:
        """Call LLM to resolve actor attribution based on context"""
        prompt = f"""
        Based on this quote and context, identify the ACTUAL PERPETRATOR of the action, not just who reported it.

        Quote: "{quote}"
        Context: "{context}"
        Category: {category}

        IMPORTANT: 
        - UNIFIL, UN observers, and reporters are NOT perpetrators
        - Look for phrases like "from south to north" (implies Hezbollah direction)
        - "Israel" or "IDF" attacking Lebanon = Israel is perpetrator
        - "Hezbollah" attacking Israel = Hezbollah is perpetrator
        - If unclear who fired/launched, return "Unknown"

        Return ONLY the perpetrator name or "Unknown" if unclear.
        """
        
        try:
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"LLM response for actor resolution: {result}")
            return result
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Unknown"
    
    def _has_multiple_numbers(self, fact: Dict[str, Any]) -> bool:
        """Check if fact contains multiple numbers that should be split"""
        value = str(fact.get('value', ''))
        for pattern in self.multi_number_patterns:
            if re.search(pattern, value):
                return True
        return False
    
    def _explode_multi_numbers(self, fact: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split facts with multiple numbers into separate facts"""
        exploded_facts = []
        value = str(fact.get('value', ''))
        
        for pattern in self.multi_number_patterns:
            matches = re.findall(pattern, value)
            if matches:
                for match in matches:
                    if isinstance(match, tuple):
                        for num in match:
                            new_fact = fact.copy()
                            new_fact['value'] = num.strip()
                            new_fact['exploded_from'] = value
                            exploded_facts.append(new_fact)
                    else:
                        new_fact = fact.copy()
                        new_fact['value'] = match.strip()
                        new_fact['exploded_from'] = value
                        exploded_facts.append(new_fact)
                break
        
        return exploded_facts if exploded_facts else [fact]
    
    def _normalize_values(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize fact values"""
        normalized = fact.copy()
        
        # Clean value field
        if 'value' in normalized:
            value = str(normalized['value'])
            # Remove extra whitespace
            value = re.sub(r'\s+', ' ', value).strip()
            # Remove common artifacts
            value = re.sub(r'[^\d\s,\.\-]', '', value)
            normalized['value'] = value
        
        # Clean quote field
        if 'quote' in normalized:
            quote = str(normalized['quote'])
            # Preserve figure captions with numbers
            if re.search(r'figure\s+\d+', quote.lower()):
                normalized['figure_caption_preserved'] = True
            # Clean extra whitespace
            quote = re.sub(r'\s+', ' ', quote).strip()
            normalized['quote'] = quote
        
        return normalized
    
    def _apply_basic_actor_patterns(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic actor normalization patterns"""
        normalized = fact.copy()
        
        if 'actor' in normalized:
            actor = str(normalized['actor'])
            for pattern, replacement in self.basic_actor_patterns.items():
                if re.search(pattern, actor, re.IGNORECASE):
                    normalized['actor'] = replacement
                    normalized['actor_normalized'] = True
                    break
        
        return normalized
    
    def _clean_data(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Final data cleaning and validation"""
        cleaned_facts = []
        
        for fact in facts:
            # Remove empty or invalid facts
            if not fact.get('value') or fact.get('value') == '':
                continue
            
            # Ensure required fields exist
            if 'category' not in fact:
                fact['category'] = 'Unknown'
            
            # Add normalization metadata
            fact['normalized_at'] = datetime.now().isoformat()
            fact['normalization_version'] = '2.0'
            
            cleaned_facts.append(fact)
        
        return cleaned_facts
    
    def get_normalization_stats(self, original_count: int, normalized_count: int) -> Dict[str, Any]:
        """Generate statistics about the normalization process"""
        return {
            'original_facts': original_count,
            'normalized_facts': normalized_count,
            'exploded_facts': normalized_count - original_count,
            'normalization_timestamp': datetime.now().isoformat(),
            'version': '2.0'
        }

# Standalone function for CLI usage
def normalize_facts_from_file(input_file: str, output_file: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """Normalize facts from a JSONL file and save results"""
    try:
        # Load environment variables from .env file
        from dotenv import load_dotenv
        load_dotenv()
        
        # Read input data
        with open(input_file, 'r', encoding='utf-8') as f:
            facts_data = [json.loads(line) for line in f if line.strip()]
        
        # Initialize normalizer
        normalizer = FactNormalizer(api_key)
        
        # Normalize facts
        normalized_facts = normalizer.normalize_facts(facts_data)
        
        # Save results
        with open(output_file, 'w', encoding='utf-8') as f:
            for fact in normalized_facts:
                f.write(json.dumps(fact, ensure_ascii=False) + '\n')
        
        # Generate stats
        stats = normalizer.get_normalization_stats(len(facts_data), len(normalized_facts))
        
        logger.info(f"Normalization complete: {input_file} → {output_file}")
        logger.info(f"Stats: {stats}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Normalize extracted facts using LLM-powered analysis")
    parser.add_argument("input_file", help="Input JSONL file with extracted facts")
    parser.add_argument("output_file", help="Output JSONL file for normalized facts")
    parser.add_argument("--api-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    try:
        stats = normalize_facts_from_file(args.input_file, args.output_file, args.api_key)
        print(f"✅ Normalization complete!")
        print(f"📊 Stats: {stats}")
    except Exception as e:
        print(f"❌ Error: {e}")
        exit(1)