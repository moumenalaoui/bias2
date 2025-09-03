#!/usr/bin/env python3
"""
Ultra-Fast Quantitative Extractor
Extracts quantitative data from UN Security Council reports using ≤3 LLM calls
instead of the legacy 100+ calls approach.

This extractor uses a three-stage approach:
1. Fast pattern matching to identify candidate paragraphs
2. Smart chunking into 2-3 optimal windows
3. Single LLM call per window for comprehensive extraction

Usage:
    python ultra_fast_quantitative_extractor.py --input report.jsonl --output_dir ./output
"""

import json
import os
import sys
import re
import time
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import openai
from dotenv import load_dotenv
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

class UltraFastQuantitativeExtractor:
    """
    Ultra-fast quantitative data extractor using ≤3 LLM calls
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables
        load_dotenv()
        
        # Initialize API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        
        # Define extraction categories
        self.extraction_categories = [
            "fatalities_civilians", "fatalities_children", "fatalities_women", "fatalities_total",
            "injuries_total", "displacement_total", "displacement_returned", "missile_launches",
            "rocket_launches", "air_violations", "ground_violations", "weapon_caches",
            "tunnels_detected", "financial_aid", "force_size_laf", "force_size_idf",
            "force_size_unifil", "medical_damage", "homes_destroyed", "schools_affected",
            "infrastructure_damage", "food_insecurity", "detention_arrests", "blue_line_crossings"
        ]
        
        # Define quantitative patterns for fast filtering
        self.quantitative_patterns = {
            'numbers': r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
            'percentages': r'\b\d+(?:\.\d+)?%\b',
            'ranges': r'\b\d+\s*[-–]\s*\d+\b',
            'approximations': r'\b(?:about|approximately|around|some|several|few|many|hundreds?|thousands?|millions?)\b',
            'units': r'\b(?:people|persons|civilians|children|women|men|soldiers|forces|missiles|rockets|homes|buildings|schools|hospitals|tunnels|weapons|dollars|USD|km|miles|acres|hectares)\b'
        }
        
        # Enhanced extraction prompt template for maximum accuracy
        self.extraction_prompt = """You are an expert quantitative data extractor for UN Security Council reports. Your task is to extract ALL quantitative data points with EXTREME PRECISION and ACCURACY.

CRITICAL REQUIREMENTS:
1. Extract EVERY numerical fact, statistic, or measurement mentioned
2. Include ranges, estimates, and approximations
3. Identify responsible actors with EXACT names (no abbreviations)
4. Map to UNSCR 1701 (2006) articles when applicable
5. Provide EXACT quotes from the text (word-for-word)
6. Assign precise confidence scores (0.0-1.0)
7. NEVER invent or infer data - only extract what is explicitly stated
8. If a number is mentioned multiple times, extract it each time with context

OUTPUT FORMAT (JSON object with "data" array):
{{
  "data": [
    {{
      "category": "fatalities_civilians",
      "value": 79,
      "unit": "people",
      "actor": "Israel Defense Forces",
      "legal_article_violated": "UNSCR 1701 (2006), Article 1",
      "quote": "at least 79 Lebanese civilians have been killed",
      "confidence_score": 0.9,
      "paragraph_id": "1"
    }}
  ]
}}

CATEGORIES TO EXTRACT:
- fatalities_civilians, fatalities_children, fatalities_women, fatalities_total
- injuries_total, displacement_total, displacement_returned
- missile_launches, rocket_launches, air_violations, ground_violations
- weapon_caches, tunnels_detected, financial_aid
- force_size_laf, force_size_idf, force_size_unifil
- medical_damage, homes_destroyed, schools_affected
- infrastructure_damage, food_insecurity, detention_arrests, blue_line_crossings

ACTOR NAMES (use EXACT names):
- "Israel Defense Forces" (not "Israeli forces")
- "Hizbullah" (not "Hezbollah" or "militants")
- "Lebanese Armed Forces" (not "Lebanese army")
- "United Nations Interim Force in Lebanon" (not "UN forces")
- If unclear, use "unknown" and provide context

LEGAL MAPPING:
Map to UNSCR 1701 (2006) articles 1-19 based on the violation type:
- Article 1: Cessation of hostilities (attacks, strikes, fire)
- Article 2: Israeli withdrawal from southern Lebanon
- Article 3: Blue Line respect (territorial sovereignty)
- Article 4: Lebanese territorial integrity
- Article 5: Security arrangements
- Article 6: Taif Accords implementation
- Article 7: Release of abducted soldiers
- Article 8: No foreign forces without consent
- Article 9: Disarmament of armed groups
- Article 10: Arms sales restrictions
- Article 11: UNIFIL mandate
- Article 12: UNIFIL freedom of movement
- Article 13: UNIFIL force strength
- Article 14: Lebanese government control
- Article 15: Prevention of unauthorized arms
- Article 16: International assistance
- Article 17: Implementation review
- Article 18: Coordination with organizations
- Article 19: Previous resolutions validity

CONFIDENCE SCORING:
- 0.9-1.0: Exact numbers with clear context and direct quotes
- 0.7-0.9: Numbers with good context but some ambiguity
- 0.5-0.7: Numbers with limited context or indirect references
- 0.3-0.5: Estimated or approximate numbers
- 0.1-0.3: Uncertain or inferred numbers

TEXT TO ANALYZE:
{text}

CRITICAL: Return ONLY valid JSON in the exact format specified above. Do not add any text before or after the JSON. The response must start with {{ and end with }}."""

    def fast_pattern_filter(self, paragraphs: List[Dict]) -> List[Dict]:
        """
        Fast pattern matching to identify paragraphs likely to contain quantitative data.
        Returns paragraphs that match quantitative patterns.
        """
        print("🔍 Running fast pattern filtering...")
        
        candidates = []
        for para in paragraphs:
            text = para.get('text', '').lower()
            para_id = para.get('paragraph_id', para.get('id', 'unknown'))
            
            # Check for quantitative indicators
            has_numbers = bool(re.search(self.quantitative_patterns['numbers'], text))
            has_units = bool(re.search(self.quantitative_patterns['units'], text))
            has_approximations = bool(re.search(self.quantitative_patterns['approximations'], text))
            
            # Score the paragraph
            score = 0
            if has_numbers: score += 3
            if has_units: score += 2
            if has_approximations: score += 1
            
            # Include if score is high enough
            if score >= 2:
                para['quantitative_score'] = score
                candidates.append(para)
        
        print(f"✅ Fast filtering complete: {len(candidates)}/{len(paragraphs)} paragraphs selected")
        return candidates

    def smart_chunking(self, paragraphs: List[Dict], max_windows: int = 3) -> List[Dict]:
        """
        Smart chunking algorithm to create optimal windows for LLM processing.
        Aims to create 2-3 balanced windows with maximum 8000 tokens each.
        """
        print(f"📦 Smart chunking into {max_windows} windows...")
        
        if len(paragraphs) <= max_windows:
            # Simple case: one paragraph per window
            windows = []
            for i, para in enumerate(paragraphs):
                window = {
                    'window_id': i + 1,
                    'paragraphs': [para],
                    'total_chars': len(para.get('text', '')),
                    'paragraph_ids': [para.get('paragraph_id', para.get('id', 'unknown'))]
                }
                windows.append(window)
            return windows
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
        total_chars = sum(len(p.get('text', '')) for p in paragraphs)
        target_chars_per_window = total_chars // max_windows
        
        # Sort paragraphs by quantitative score (highest first)
        sorted_paragraphs = sorted(paragraphs, key=lambda x: x.get('quantitative_score', 0), reverse=True)
        
        windows = []
        current_window = {
            'window_id': 1,
            'paragraphs': [],
            'total_chars': 0,
            'paragraph_ids': []
        }
        
        for para in sorted_paragraphs:
            para_chars = len(para.get('text', ''))
            
            # Check if adding this paragraph would exceed target
            if (current_window['total_chars'] + para_chars > target_chars_per_window * 1.2 and 
                len(current_window['paragraphs']) > 0 and 
                len(windows) < max_windows - 1):
                
                # Start new window
                windows.append(current_window)
                current_window = {
                    'window_id': len(windows) + 1,
                    'paragraphs': [],
                    'total_chars': 0,
                    'paragraph_ids': []
                }
            
            # Add paragraph to current window
            current_window['paragraphs'].append(para)
            current_window['total_chars'] += para_chars
            current_window['paragraph_ids'].append(para.get('paragraph_id', para.get('id', 'unknown')))
        
        # Add the last window
        if current_window['paragraphs']:
            windows.append(current_window)
        
        # Ensure we don't exceed max_windows
        if len(windows) > max_windows:
            # Merge smallest windows
            while len(windows) > max_windows:
                # Find two smallest windows to merge
                smallest_idx = min(range(len(windows)), key=lambda i: windows[i]['total_chars'])
                second_smallest_idx = min(
                    [i for i in range(len(windows)) if i != smallest_idx],
                    key=lambda i: windows[i]['total_chars']
                )
                
                # Merge into first window
                windows[smallest_idx]['paragraphs'].extend(windows[second_smallest_idx]['paragraphs'])
                windows[smallest_idx]['total_chars'] += windows[second_smallest_idx]['total_chars']
                windows[smallest_idx]['paragraph_ids'].extend(windows[second_smallest_idx]['paragraph_ids'])
                
                # Remove second window
                windows.pop(second_smallest_idx)
                
                # Renumber windows
                for i, window in enumerate(windows):
                    window['window_id'] = i + 1
        
        print(f"✅ Chunking complete: {len(windows)} windows created")
        for i, window in enumerate(windows):
            print(f"   Window {i+1}: {len(window['paragraphs'])} paragraphs, {window['total_chars']} chars")
        
        return windows

    def extract_from_window(self, window: Dict, window_idx: int) -> Dict:
        """
        Extract quantitative data from a single window using one LLM call.
        """
        try:
            # Prepare text for this window
            window_text = "\n\n---\n\n".join([
                f"Paragraph ID: {para.get('paragraph_id', para.get('id', 'unknown'))}\nText: {para.get('text', '')}"
                for para in window['paragraphs']
            ])
            
            # Create prompt for this window
            prompt = self.extraction_prompt.format(text=window_text)
            
            # Debug: Verify prompt is properly formatted
            if "{text}" in prompt:
                print(f"⚠️ Warning: Prompt template not properly formatted!")
                print(f"🔍 Prompt preview: {prompt[:200]}...")
            
            # Debug: Show what text we're sending to the LLM
            print(f"🔍 Window text preview: {window_text[:200]}...")
            print(f"🔍 Prompt length: {len(prompt)} characters")
            print(f"🔍 Prompt ends with: {prompt[-100:]}...")
            
            print(f"🤖 Processing window {window_idx + 1}...")
            print(f"📝 Prompt length: {len(prompt)} characters")
            
            # Call OpenAI API - Optimized for accuracy and cost
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",  # Cost-effective but still very accurate
                messages=[{"role": "user", "content": prompt}],
                temperature=0,  # Deterministic for consistency
                max_tokens=6000,  # Increased for comprehensive extraction
                top_p=1.0,  # Full probability distribution for accuracy
                frequency_penalty=0.0,  # No penalty for repetition
                presence_penalty=0.0,  # No penalty for presence
                response_format={"type": "json_object"}  # Force JSON output
            )
            
            # Extract response
            gpt_response = response.choices[0].message.content
            print(f"📥 Response received: {len(gpt_response)} characters")
            print(f"📝 Response preview: {gpt_response[:200]}...")
            print(f"🔍 Full response: {gpt_response}")
            
            # Parse JSON response with standardized format
            try:
                response_data = json.loads(gpt_response)
                
                # Handle standardized format: {"data": [...]}
                if isinstance(response_data, dict) and "data" in response_data:
                    extracted_data = response_data["data"]
                elif isinstance(response_data, list):
                    # Fallback for direct array format
                    extracted_data = response_data
                else:
                    # Single object format
                    extracted_data = [response_data]
                
                if not isinstance(extracted_data, list):
                    extracted_data = [extracted_data]
                    
                print(f"✅ JSON parsed successfully: {len(extracted_data)} items")
                
            except json.JSONDecodeError as json_err:
                print(f"⚠️ JSON parse error: {json_err}")
                print(f"🔍 Raw response: {gpt_response}")
                # Try to extract JSON from response
                extracted_data = self.extract_json_from_response(gpt_response)
                print(f"🔧 Extracted {len(extracted_data)} items after cleanup")
            
            return {
                'window_id': window['window_id'],
                'paragraph_ids': window['paragraph_ids'],
                'extracted_data': extracted_data,
                'raw_response': gpt_response,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error processing window {window_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'window_id': window['window_id'],
                'paragraph_ids': window['paragraph_ids'],
                'extracted_data': [],
                'raw_response': f"ERROR: {str(e)}",
                'success': False
            }

    def extract_json_from_response(self, response_text: str) -> List[Dict]:
        """
        Extract JSON from GPT response text.
        """
        # Try to find JSON array in the response
        json_pattern = r'\[.*\]'
        matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        # If no array found, try to extract individual objects
        object_pattern = r'\{[^{}]*\}'
        matches = re.findall(object_pattern, response_text)
        
        objects = []
        for match in matches:
            try:
                obj = json.loads(match)
                objects.append(obj)
            except json.JSONDecodeError:
                continue
        
        return objects

    def process_windows_parallel(self, windows: List[Dict]) -> List[Dict]:
        """
        Process all windows in parallel for maximum speed.
        """
        print(f"🚀 Processing {len(windows)} windows in parallel...")
        
        results = []
        with ThreadPoolExecutor(max_workers=min(len(windows), 3)) as executor:
            # Submit all tasks
            future_to_window = {
                executor.submit(self.extract_from_window, window, i): (window, i)
                for i, window in enumerate(windows)
            }
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_window):
                window, idx = future_to_window[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ Window {idx + 1} completed")
                except Exception as e:
                    print(f"❌ Window {idx + 1} failed: {e}")
                    results.append({
                        'window_id': window['window_id'],
                        'paragraph_ids': window['paragraph_ids'],
                        'extracted_data': [],
                        'raw_response': f"ERROR: {str(e)}",
                        'success': False
                    })
        
        # Sort results by window_id
        results.sort(key=lambda x: x['window_id'])
        return results

    def process_windows_sequential(self, windows: List[Dict]) -> List[Dict]:
        """
        Process windows sequentially (fallback if parallel fails).
        """
        print(f"🔄 Processing {len(windows)} windows sequentially...")
        
        results = []
        for i, window in enumerate(windows):
            result = self.extract_from_window(window, i)
            results.append(result)
            
            # Small delay between calls to avoid rate limiting
            if i < len(windows) - 1:
                time.sleep(1)
        
        return results

    def consolidate_results(self, window_results: List[Dict]) -> List[Dict]:
        """
        Consolidate results from all windows into a single dataset.
        """
        print("🔗 Consolidating results from all windows...")
        
        all_extracted_data = []
        
        for result in window_results:
            if result['success'] and result['extracted_data']:
                # Add window context to each data point
                for item in result['extracted_data']:
                    if isinstance(item, dict):
                        item['window_id'] = result['window_id']
                        item['source_paragraphs'] = result['paragraph_ids']
                        all_extracted_data.append(item)
        
        print(f"✅ Consolidated {len(all_extracted_data)} data points from {len(window_results)} windows")
        return all_extracted_data

    def add_metadata(self, data_points: List[Dict]) -> List[Dict]:
        """
        Add metadata and validation to extracted data points.
        """
        print("📊 Adding metadata and validation...")
        
        for item in data_points:
            # Ensure required fields exist
            if 'timestamp' not in item:
                item['timestamp'] = datetime.now().isoformat()
            
            # Validate confidence score
            if 'confidence_score' not in item or not isinstance(item['confidence_score'], (int, float)):
                item['confidence_score'] = 0.5
            
            # Ensure confidence is in valid range
            item['confidence_score'] = max(0.0, min(1.0, float(item['confidence_score'])))
            
            # Add extraction method
            item['extraction_method'] = 'ultra_fast_llm'
            item['llm_calls_used'] = len(self.windows)
        
        return data_points

    def generate_summary(self, data_points: List[Dict]) -> Dict:
        """
        Generate summary statistics for the extraction.
        """
        if not data_points:
            return {
                'total_data_points': 0,
                'categories_found': 0,
                'average_confidence': 0.0,
                'llm_calls_used': 0,
                'extraction_method': 'ultra_fast_llm'
            }
        
        # Count by category
        category_counts = {}
        for item in data_points:
            category = item.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Calculate average confidence
        confidences = [item.get('confidence_score', 0.0) for item in data_points]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            'total_data_points': len(data_points),
            'categories_found': len(category_counts),
            'category_distribution': category_counts,
            'average_confidence': round(avg_confidence, 3),
            'llm_calls_used': len(self.windows),
            'extraction_method': 'ultra_fast_llm',
            'extraction_timestamp': datetime.now().isoformat()
        }

    def extract(self, input_file: str, output_dir: str, max_windows: int = 3) -> Dict:
        """
        Main extraction method using ≤3 LLM calls.
        """
        start_time = time.time()
        
        print(f"🚀 Starting Ultra-Fast Quantitative Extraction")
        print(f"📁 Input: {input_file}")
        print(f"📁 Output: {output_dir}")
        print(f"🎯 Target: ≤{max_windows} LLM calls")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read input paragraphs
        print("📖 Reading input paragraphs...")
        with open(input_file, 'r', encoding='utf-8') as f:
            paragraphs = []
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    paragraphs.append(data)
        
        print(f"📊 Loaded {len(paragraphs)} paragraphs")
        
        # Stage 1: Fast pattern filtering
        candidates = self.fast_pattern_filter(paragraphs)
        
        if not candidates:
            print("⚠️ No quantitative candidates found. Processing all paragraphs...")
            candidates = paragraphs
        
        # Stage 2: Smart chunking
        self.windows = self.smart_chunking(candidates, max_windows)
        
        # Stage 3: LLM extraction (≤3 calls)
        try:
            window_results = self.process_windows_parallel(self.windows)
        except Exception as e:
            print(f"⚠️ Parallel processing failed, falling back to sequential: {e}")
            window_results = self.process_windows_sequential(self.windows)
        
        # Stage 4: Consolidate results
        consolidated_data = self.consolidate_results(window_results)
        
        # Stage 5: Add metadata
        final_data = self.add_metadata(consolidated_data)
        
        # Stage 6: Generate summary
        summary = self.generate_summary(final_data)
        
        # Save results
        results_file = output_path / "ultra_fast_extraction_results.jsonl"
        summary_file = output_path / "ultra_fast_extraction_summary.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            for item in final_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Save raw window results for debugging
        debug_file = output_path / "window_results_debug.json"
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(window_results, f, ensure_ascii=False, indent=2)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 Ultra-Fast Extraction Complete!")
        print(f"⏱️  Total time: {total_time:.2f} seconds")
        print(f"🤖 LLM calls used: {summary['llm_calls_used']}")
        print(f"📊 Data points extracted: {summary['total_data_points']}")
        print(f"📁 Output files:")
        print(f"   - Results: {results_file}")
        print(f"   - Summary: {summary_file}")
        print(f"   - Debug: {debug_file}")
        
        return {
            'success': True,
            'summary': summary,
            'output_files': {
                'results': str(results_file),
                'summary': str(summary_file),
                'debug': str(debug_file)
            },
            'timing': {
                'total_seconds': total_time,
                'llm_calls': summary['llm_calls_used']
            }
        }


def main():
    """Main CLI function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ultra-Fast Quantitative Extractor (≤3 LLM calls)")
    parser.add_argument("--input", required=True, help="Input JSONL file with paragraphs")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--max_windows", type=int, default=3, help="Maximum number of LLM calls (default: 3)")
    parser.add_argument("--api_key", help="OpenAI API key (optional, uses env var if not provided)")
    
    args = parser.parse_args()
    
    try:
        extractor = UltraFastQuantitativeExtractor(api_key=args.api_key)
        result = extractor.extract(args.input, args.output_dir, args.max_windows)
        
        if result['success']:
            print(f"\n✅ Extraction successful!")
            return 0
        else:
            print(f"\n❌ Extraction failed!")
            return 1
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
