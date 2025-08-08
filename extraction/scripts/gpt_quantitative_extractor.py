#!/usr/bin/env python3
"""
GPT-4o Quantitative Extractor
Extracts structured quantitative data from UN Security Council reports using GPT-4o
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import time
import openai
from dotenv import load_dotenv
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class RateLimiter:
    """Rate limiter for OpenAI API calls to prevent 429 errors."""
    
    def __init__(self, max_requests_per_minute=60):
        self.max_requests_per_minute = max_requests_per_minute
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if we're approaching the rate limit."""
        with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests_per_minute:
                # Wait until we can make another request
                sleep_time = 60 - (now - self.requests[0]) + 1
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    return self.wait_if_needed()  # Recursive call after waiting
            
            self.requests.append(now)

class GPTQuantitativeExtractor:
    """
    GPT-4o based quantitative data extractor for UN Security Council reports
    """
    
    def __init__(self, api_key: Optional[str] = None):
        # Load environment variables from .env file
        load_dotenv()
        
        # Initialize API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable or pass it as a parameter.")
        
        # Initialize rate limiter for concurrent processing
        self.rate_limiter = RateLimiter(max_requests_per_minute=50)  # Conservative rate limit
        
        openai.api_key = self.api_key
        
        # Define comprehensive extraction prompt for batched processing
        self.extraction_prompt_template = """You are a precision extractor of quantitative data from UN Security Council reports. Your task is to identify and extract all numerical data points with their context, actors, and legal implications from multiple paragraphs.

Extract quantitative data in the following JSON format:
[
  {{
    "category": "air_violations|ground_violations|missile_launches|rocket_launches|fatalities_total|fatalities_children|fatalities_women|fatalities_civilians|injuries_total|displacement_total|displacement_returned|weapon_caches|tunnels_detected|financial_aid|force_size_laf|force_size_idf|force_size_unifil|medical_damage|homes_destroyed|schools_affected|infrastructure_damage|food_insecurity|detention_arrests|blue_line_crossings",
    "value": <numeric_value>,
    "unit": "<unit_of_measurement>",
    "actor": "<responsible_actor>",
    "legal_article_violated": "UNSCR 1701 (2006), Article X",
    "quote": "<exact_text_containing_the_data>",
    "confidence_score": <0.0-1.0>,
    "paragraph_id": "<exact_paragraph_id_from_input>"
  }}
]

Guidelines:
- Extract ALL numerical data points from ALL paragraphs provided
- **CRITICAL**: Use the exact paragraph_id from the input (e.g., "1", "2", "3") for each data point
- Identify the responsible actor with PRECISION:
  * Use EXACT names: "Israel Defense Forces" (not "Israeli forces")
  * Use EXACT names: "Hizbullah" (not "Hezbollah" or "militants")
  * Use EXACT names: "Lebanese Armed Forces" (not "Lebanese army")
  * Use EXACT names: "United Nations Interim Force in Lebanon" (not "UN forces")
  * If unclear, use "unknown" and provide context in quote
- Map to UNSCR 1701 (2006) articles with PRECISION (all 19 articles):
  * Article 1: Immediate cessation of hostilities (ALL attacks, strikes, fire)
  * Article 2: Withdrawal of Israeli forces from southern Lebanon
  * Article 3: Full respect for Blue Line (territorial sovereignty)
  * Article 4: Territorial integrity and sovereignty of Lebanon
  * Article 5: Security arrangements to prevent resumption of hostilities
  * Article 6: Full implementation of Taif Accords and Resolution 1559
  * Article 7: Unconditional release of abducted Israeli soldiers
  * Article 8: No foreign forces without Lebanese government consent
  * Article 9: Disarmament of all armed groups in Lebanon
  * Article 10: No sales or supply of arms except to Lebanese government
  * Article 11: UNIFIL mandate and operations
  * Article 12: UNIFIL freedom of movement and access
  * Article 13: UNIFIL force strength and equipment
  * Article 14: Lebanese government control over territory
  * Article 15: Prevention of unauthorized arms
  * Article 16: International assistance for Lebanese government
  * Article 17: Implementation review and reporting
  * Article 18: Coordination with UN and regional organizations
  * Article 19: Continued validity of previous resolutions
- Provide exact quotes from the text
- Assign confidence scores based on:
  * 0.9-1.0: Exact numbers with clear context and direct quotes
  * 0.7-0.9: Numbers with good context but some ambiguity
  * 0.5-0.7: Numbers with limited context or indirect references
  * 0.3-0.5: Estimated or approximate numbers
  * 0.1-0.3: Uncertain or inferred numbers
- Use appropriate categories from the predefined list
- Handle ranges, estimates, and approximations appropriately
- VALIDATION RULES:
  * Reject data points with confidence < 0.3
  * Ensure all quotes are at least 15 characters long
  * Verify numeric values are reasonable (no negative casualties, etc.)
  * Check that legal articles are properly formatted: "UNSCR 1701 (2006), Article N"

Paragraphs to analyze:
{paragraph_text}

Extract all quantitative data points from all paragraphs, ensuring each data point has the correct paragraph_id:"""

    def create_extraction_prompts(self, input_file: str, batch_size: int = 5) -> List[Dict]:
        """Create GPT-4o extraction prompts from JSONL input file with smart batching."""
        
        prompts = []
        
        with open(input_file, 'r', encoding='utf-8') as f:
            paragraphs = []
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    paragraphs.append(data)
            
            # Group paragraphs into batches
            for i in range(0, len(paragraphs), batch_size):
                batch = paragraphs[i:i + batch_size]
                
                # Create combined prompt for batch with explicit paragraph IDs
                combined_text = "\n\n---\n\n".join([
                    f"Paragraph ID: {item.get('paragraph_id', item.get('id', 'unknown'))}\nText: {item.get('text', '')}"
                    for item in batch
                ])
                
                prompt_data = {
                    "id": f"batch_{i//batch_size + 1}",
                    "text": combined_text,
                    "batch_size": len(batch),
                    "paragraph_ids": [item.get('paragraph_id', item.get('id', 'unknown')) for item in batch]
                }
                
                prompts.append(prompt_data)
        
        print(f"📝 Created {len(prompts)} batched extraction prompts (batch size: {batch_size})")
        return prompts

    def extract_with_gpt4o(self, prompts: List[Dict], output_file: str, batch_size: int = 10, max_workers: int = 8) -> List[Dict]:
        """Extract quantitative data using GPT-4o with parallel processing."""
        
        results = []
        total_prompts = len(prompts)
        
        print(f"🤖 Starting parallel GPT-4o extraction for {total_prompts} prompts with {max_workers} workers...")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_prompt = {
                executor.submit(self.process_single_prompt, prompt_data): prompt_data 
                for prompt_data in prompts
            }
            
            # Process completed tasks
            with open(output_file, 'w', encoding='utf-8') as out_f:
                completed = 0
                for future in as_completed(future_to_prompt):
                    prompt_data = future_to_prompt[future]
                    try:
                        result = future.result()
                        
                        # Write to file immediately
                        out_f.write(json.dumps(result) + "\n")
                        out_f.flush()
                        
                        results.append(result)
                        completed += 1
                        
                        # Progress update
                        if completed % 5 == 0 or completed == total_prompts:
                            print(f"✅ Progress: {completed}/{total_prompts} prompts processed ({completed/total_prompts*100:.1f}%)")
                        
                    except Exception as e:
                        print(f"❌ Exception for {prompt_data['id']}: {e}")
                        # Write error result
                        error_result = {
                            "id": prompt_data["id"],
                            "quantitative_data": f"ERROR: {str(e)}",
                            "original_text": prompt_data["original_text"],
                            "timestamp": datetime.now().isoformat()
                        }
                        out_f.write(json.dumps(error_result) + "\n")
                        out_f.flush()
                        results.append(error_result)
        
        print(f"✅ Parallel GPT-4o extraction complete. Processed {len(results)} prompts.")
        return results

    def process_single_prompt(self, prompt_data: Dict) -> Dict:
        """Process a single prompt with rate limiting and error handling."""
        try:
            # Wait for rate limiter
            self.rate_limiter.wait_if_needed()
            
            # Create the prompt for batched processing
            prompt = self.extraction_prompt_template.format(
                paragraph_text=prompt_data["text"]
            )
            
            # Call GPT-4o
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=4000,  # Comprehensive responses
                top_p=0.9,  # Focus on most likely tokens
                frequency_penalty=0.1,  # Reduce repetition
                presence_penalty=0.1  # Encourage diverse coverage
            )
            
            # Extract response content
            gpt_response = response.choices[0].message.content
            
            # Create result entry
            result = {
                "id": prompt_data["id"],
                "quantitative_data": gpt_response,
                "original_text": prompt_data["text"],
                "batch_size": prompt_data.get("batch_size", 1),
                "paragraph_ids": prompt_data.get("paragraph_ids", []),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error processing {prompt_data['id']}: {e}")
            
            # Handle rate limiting specifically
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"⏳ Rate limit hit for {prompt_data['id']}. Waiting 10 seconds...")
                time.sleep(10)
                # Try again once
                try:
                    self.rate_limiter.wait_if_needed()
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": prompt_data["prompt"]}],
                        temperature=0,
                        max_tokens=4000,  # Comprehensive responses
                        top_p=0.9,  # Focus on most likely tokens
                        frequency_penalty=0.1,  # Reduce repetition
                        presence_penalty=0.1  # Encourage diverse coverage
                    )
                    gpt_response = response.choices[0].message.content
                    
                    result = {
                        "id": prompt_data["id"],
                        "quantitative_data": gpt_response,
                        "original_text": prompt_data["original_text"],
                        "timestamp": datetime.now().isoformat()
                    }
                    print(f"✅ Retry successful for {prompt_data['id']}")
                    return result
                except Exception as retry_e:
                    print(f"❌ Retry failed for {prompt_data['id']}: {retry_e}")
            
            # Return error result
            return {
                "id": prompt_data["id"],
                "quantitative_data": f"ERROR: {str(e)}",
                "original_text": prompt_data["original_text"],
                "timestamp": datetime.now().isoformat()
            }

    def postprocess_gpt_output(self, gpt_output_file: str, final_output_file: str) -> List[Dict]:
        """Postprocess GPT-4o output into structured format."""
        
        cleaned_results = []
        
        with open(gpt_output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                
                try:
                    data = json.loads(line)
                    
                    # Skip error entries
                    if data.get("quantitative_data", "").startswith("ERROR:"):
                        continue
                    
                    # Parse GPT response as JSON
                    try:
                        quantitative_data = json.loads(data["quantitative_data"])
                        
                        # Add paragraph ID to each extracted item
                        for item in quantitative_data:
                            if isinstance(item, dict):
                                # Use the paragraph_id from GPT response if available, otherwise use batch info
                                if "paragraph_id" in item:
                                    # GPT provided specific paragraph ID
                                    cleaned_results.append(item)
                                else:
                                    # Fallback to batch ID (shouldn't happen with proper prompting)
                                    item["paragraph_id"] = data["id"]
                                    cleaned_results.append(item)
                        
                    except json.JSONDecodeError:
                        # Try to extract JSON from the response
                        cleaned_data = self.extract_json_from_response(data["quantitative_data"])
                        if cleaned_data:
                            for item in cleaned_data:
                                if isinstance(item, dict):
                                    # Use the paragraph_id from GPT response if available
                                    if "paragraph_id" not in item:
                                        item["paragraph_id"] = data["id"]
                                    cleaned_results.append(item)
                
                except json.JSONDecodeError as e:
                    print(f"⚠️ Error parsing GPT output line: {e}")
                    continue
        
        # Clean up qualitative values that shouldn't be treated as numerical
        cleaned_results = self.clean_qualitative_values(cleaned_results)
        
        # Save cleaned results
        with open(final_output_file, 'w', encoding='utf-8') as out_f:
            for item in cleaned_results:
                out_f.write(json.dumps(item) + "\n")
        
        print(f"🧹 Postprocessing complete. {len(cleaned_results)} structured data points extracted.")
        return cleaned_results

    def clean_qualitative_values(self, data_points: List[Dict]) -> List[Dict]:
        """Clean up qualitative values that shouldn't be treated as numerical."""
        
        # Define qualitative units that indicate the value should be N/A
        qualitative_units = {
            'daily', 'large-scale', 'several', 'area', 'locations', 'incursions',
            'extensive', 'extensively', 'significant', 'multiple', 'various', 'numerous',
            'frequent', 'regular', 'continuous', 'ongoing', 'repeated'
        }
        
        cleaned_data = []
        
        for item in data_points:
            # Check if value is 1 and unit is qualitative
            value = item.get('value')
            unit = item.get('unit') or ''  # Handle None values
            unit = unit.lower() if unit else ''
            
            if value == 1 and unit in qualitative_units:
                # Replace value with "N/A" for qualitative measures
                item['value'] = "N/A"
                print(f"🔄 Converted qualitative value: {unit} -> N/A")
            
            cleaned_data.append(item)
        
        return cleaned_data

    def extract_json_from_response(self, response_text: str) -> List[Dict]:
        """Extract JSON from GPT response text."""
        
        # Try to find JSON array in the response
        import re
        
        # Look for JSON array pattern
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

    def add_reliability_scoring(self, data_points: List[Dict]) -> List[Dict]:
        """Add reliability scoring to extracted data points."""
        
        for item in data_points:
            # Calculate reliability score based on various factors
            confidence = item.get('confidence_score', 0.5)
            
            # Adjust based on data quality indicators
            quote_quality = len(item.get('quote', '')) > 20
            actor_specificity = bool(item.get('actor'))
            legal_mapping = bool(item.get('legal_article_violated'))
            
            # Calculate final reliability score with enhanced factors
            reliability_factors = [confidence]
            if quote_quality:
                reliability_factors.append(0.15)  # Increased weight for good quotes
            if actor_specificity:
                reliability_factors.append(0.15)  # Increased weight for specific actors
            if legal_mapping:
                reliability_factors.append(0.15)  # Increased weight for legal mapping
            
            # Additional quality factors
            value_specificity = isinstance(item.get('value'), (int, float)) and item.get('value') > 0
            unit_specificity = bool(item.get('unit')) and item.get('unit') != 'unknown'
            
            if value_specificity:
                reliability_factors.append(0.1)
            if unit_specificity:
                reliability_factors.append(0.1)
            
            item['reliability_score'] = min(1.0, sum(reliability_factors))
        
        return data_points

    def generate_extraction_summary(self, data_points: List[Dict]) -> Dict:
        """Generate summary statistics for extracted data."""
        
        if not data_points:
            return {"error": "No data points to summarize"}
        
        # Category distribution
        category_counts = {}
        actor_counts = {}
        value_ranges = {}
        reliability_scores = []
        
        for item in data_points:
            # Category counts
            category = item.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Actor counts
            actor = item.get('actor', 'unknown')
            actor_counts[actor] = actor_counts.get(actor, 0) + 1
            
            # Value ranges
            value = item.get('value')
            if value is not None and isinstance(value, (int, float)):
                if category not in value_ranges:
                    value_ranges[category] = {'min': float('inf'), 'max': float('-inf'), 'values': []}
                value_ranges[category]['values'].append(value)
                value_ranges[category]['min'] = min(value_ranges[category]['min'], value)
                value_ranges[category]['max'] = max(value_ranges[category]['max'], value)
            
            # Reliability scores
            reliability_scores.append(item.get('reliability_score', 0.5))
        
        # Calculate averages
        avg_reliability = sum(reliability_scores) / len(reliability_scores) if reliability_scores else 0
        
        summary = {
            "total_data_points": len(data_points),
            "category_distribution": category_counts,
            "actor_distribution": actor_counts,
            "value_ranges": value_ranges,
            "average_reliability": avg_reliability,
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        return summary

def main():
    """Main function for GPT quantitative extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT-4o Quantitative Extractor")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=5, help="Number of paragraphs per API call (batching)")
    parser.add_argument("--processing-batch-size", type=int, default=10, help="Batch size for parallel processing")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--api-key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--postprocess-only", action="store_true", help="Only postprocess existing GPT output")
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize extractor
    try:
        extractor = GPTQuantitativeExtractor(api_key=args.api_key)
    except ValueError as e:
        print(f"❌ {e}")
        return 1
    
    # Generate output file paths
    gpt_output_file = output_dir / "gpt_quant_extraction_output.jsonl"
    final_output_file = output_dir / "precision_hybrid_extraction_results.jsonl"
    summary_file = output_dir / "extraction_summary.json"
    
    try:
        if args.postprocess_only:
            # Only postprocess existing GPT output
            print("🧹 Postprocessing existing GPT output...")
            cleaned_results = extractor.postprocess_gpt_output(
                args.input, 
                str(final_output_file)
            )
            
            # Step 4: Add reliability scoring
            print("📊 Step 4: Adding reliability scoring...")
            scored_results = extractor.add_reliability_scoring(cleaned_results)
            
            # Step 5: Add legal grounding and validate legal article mapping
            print("⚖️ Step 5: Adding legal grounding and validating UNSCR 1701 mapping...")
            try:
                from legal_violation_mapper import LegalViolationMapper
                mapper = LegalViolationMapper()
                
                # Add legal grounding to each data point
                legally_grounded_results = []
                for data_point in scored_results:
                    # Get legal grounding using the mapper
                    legal_grounding = mapper.get_legal_grounding(data_point)
                    
                    # Validate and enhance the legal_article_violated field
                    original_legal_article = data_point.get("legal_article_violated", "")
                    if legal_grounding["primary_violation"]:
                        # Use the mapper's result to ensure UNSCR 1701 consistency
                        validated_legal_article = legal_grounding["primary_violation"]["legal_citation"]
                    else:
                        # If no mapping found, keep original but ensure UNSCR 1701 format
                        if "UNSCR 1701" not in original_legal_article and original_legal_article.strip():
                            validated_legal_article = f"UNSCR 1701 (2006), {original_legal_article}"
                        else:
                            validated_legal_article = original_legal_article
                    
                    # Update the data point with legal grounding and validated legal article
                    data_point.update({
                        "legal_article_violated": validated_legal_article,
                        "legal_violations": legal_grounding["legal_articles"],
                        "primary_legal_violation": legal_grounding["primary_violation"],
                        "legal_grounding_summary": legal_grounding["legal_grounding_summary"]
                    })
                    legally_grounded_results.append(data_point)
                
                scored_results = legally_grounded_results
                print(f"✅ Added legal grounding and validated UNSCR 1701 mapping for {len(scored_results)} data points")
            except Exception as e:
                print(f"⚠️ Warning: Could not add legal grounding: {e}")
                print("Continuing without legal grounding...")
            
            # Step 6: Generate summary
            print("📋 Step 6: Generating summary...")
            summary = extractor.generate_extraction_summary(scored_results)
            
            # Save summary
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            # Print final results
            print(f"\n🎉 GPT Quantitative Extraction Complete!")
            print(f"📊 Total data points extracted: {len(scored_results)}")
            print(f"📊 Categories found: {len(summary['category_distribution'])}")
            print(f"📊 Average reliability score: {summary['average_reliability']:.3f}")
            print(f"📁 Output files:")
            print(f"   - Structured results: {final_output_file}")
            print(f"   - Summary: {summary_file}")
            
            return 0
        else:
            # Full pipeline
            # Step 1: Create extraction prompts
            print("📝 Step 1: Creating extraction prompts...")
            prompts = extractor.create_extraction_prompts(args.input, batch_size=args.batch_size)
            
            if not prompts:
                print("❌ No valid prompts created from input file")
                return 1
            
                        # Step 2: Run GPT-4o extraction
            print("🤖 Step 2: Running GPT-4o extraction...")
            gpt_results = extractor.extract_with_gpt4o(
                prompts, 
                str(gpt_output_file),
                batch_size=args.processing_batch_size,
                max_workers=args.max_workers
            )
            
            # Step 3: Postprocess results
            print("🧹 Step 3: Postprocessing results...")
            cleaned_results = extractor.postprocess_gpt_output(
                str(gpt_output_file), 
                str(final_output_file)
            )
            
            # Step 4: Add reliability scoring
            print("📊 Step 4: Adding reliability scoring...")
            scored_results = extractor.add_reliability_scoring(cleaned_results)
            
            # Step 5: Generate summary
            print("📋 Step 5: Generating summary...")
            summary = extractor.generate_extraction_summary(scored_results)
            
            # Save summary
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            # Print final results
            print(f"\n🎉 GPT Quantitative Extraction Complete!")
            print(f"📊 Total data points extracted: {len(scored_results)}")
            print(f"📊 Categories found: {len(summary['category_distribution'])}")
            print(f"📊 Average reliability score: {summary['average_reliability']:.3f}")
            print(f"📁 Output files:")
            print(f"   - GPT raw output: {gpt_output_file}")
            print(f"   - Structured results: {final_output_file}")
            print(f"   - Summary: {summary_file}")
            
            return 0
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 