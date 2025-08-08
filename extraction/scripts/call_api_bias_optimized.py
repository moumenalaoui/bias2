#!/usr/bin/env python3
"""
Optimized Text-based Bias Analysis with OpenAI API
Features parallel processing and improved rate limiting
"""

import json
import re
import os
import sys
import time
import asyncio
from datetime import datetime
from openai import AsyncOpenAI
from pathlib import Path
from dotenv import load_dotenv
import aiofiles
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Configuration
MODEL = "gpt-4o"
RATE_LIMIT = 50  # requests per minute
MAX_CONCURRENT_REQUESTS = 5  # Number of concurrent API calls
BATCH_SIZE = 10  # Process paragraphs in batches

# Resolution cache directory
RESOLUTION_CACHE_DIR = "resolution_cache"

class RateLimiter:
    """Improved rate limiter for concurrent processing"""
    
    def __init__(self, max_requests_per_minute=50, max_concurrent=5):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_concurrent = max_concurrent
        self.requests = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire permission to make a request"""
        await self.semaphore.acquire()
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 minute
            self.requests = [req_time for req_time in self.requests if now - req_time < 60]
            
            if len(self.requests) >= self.max_requests_per_minute:
                # Wait until we can make another request
                sleep_time = 60 - (now - self.requests[0]) + 1
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    return await self.acquire()  # Recursive call after waiting
            
            self.requests.append(now)
    
    def release(self):
        """Release the semaphore"""
        self.semaphore.release()

def load_resolution_cache():
    """Load resolution data from JSON files in resolution_cache folder."""
    resolution_data = {}
    if not os.path.exists(RESOLUTION_CACHE_DIR):
        logger.warning(f"Resolution cache directory {RESOLUTION_CACHE_DIR} not found")
        return resolution_data

    for filename in os.listdir(RESOLUTION_CACHE_DIR):
        if filename.endswith(".json"):
            try:
                with open(os.path.join(RESOLUTION_CACHE_DIR, filename), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    resolution_id = data.get("resolution_number", "")
                    if resolution_id:
                        resolution_data[resolution_id] = {
                            "articles": data.get("articles", {}),
                            "text": data.get("full_text", ""),
                            "keywords": data.get("keywords", []),
                            "article_count": data.get("article_count", 0),
                            "url": data.get("url", "")
                        }
            except Exception as e:
                logger.warning(f"Error loading {filename}: {e}")
                continue
    
    logger.info(f"Loaded {len(resolution_data)} resolutions from cache")
    return resolution_data

def extract_bias_info():
    """Extract bias information from UN report paragraphs using Entman's framing theory."""
    return {
        "name": "extract_bias_info",
        "description": "Analyze bias in UN report paragraphs using Entman's framing theory",
        "parameters": {
            "type": "object",
            "properties": {
                "core_actors": {"type": "array", "items": {"type": "string"}, "description": "Main actors mentioned in the paragraph"},
                "action": {"type": ["string", "null"], "description": "Main action or event described"},
                "location": {"type": ["string", "null"], "description": "Geographic location of events"},
                "occurrence_date": {"type": ["string", "null"], "description": "Date or time period of events"},
                "violation_type": {"type": "array", "items": {"type": "string"}, "description": "Types of violations mentioned"},
                "violation_subtypes": {"type": "array", "items": {"type": "string"}, "description": "Specific subtypes of violations"},
                "bias_flag": {"type": "string", "enum": ["none", "framing", "selection", "omission"], "description": "Type of bias detected"},
                "bias_reason": {"type": "string", "description": "Detailed explanation of bias detection"},
                "summary": {"type": "string", "description": "Neutral summary of the paragraph"},
                "entman_framing_analysis": {"type": "object", "description": "Explicit analysis using Entman's four framing functions"},
                "framing_bias_indicators": {"type": "object", "description": "Specific indicators of framing bias"},
                "language_analysis": {"type": "object", "description": "Analysis of language use"},
                "actor_coverage_analysis": {"type": "object", "description": "Analysis of actor coverage balance"},
                "incident_specific_analysis": {"type": "array", "description": "Analysis of specific incidents mentioned in the paragraph", "items": {"type": "object"}},
                "actor_comparison_analysis": {"type": "object", "description": "Comparative analysis of how different actors are portrayed"},
                "systemic_bias_indicators": {"type": "object", "description": "Indicators of potential systemic bias patterns"}
            },
            "required": [
                "core_actors", "action", "location", "occurrence_date", "violation_type",
                "violation_subtypes", "bias_flag", "bias_reason",
                "summary", "entman_framing_analysis", "framing_bias_indicators", "language_analysis", 
                "actor_coverage_analysis", "incident_specific_analysis", "actor_comparison_analysis", 
                "systemic_bias_indicators"
            ]
        }
    }

def get_system_prompt(analysis_type):
    """Get system prompt for bias analysis."""
    return """You are an expert analyst specializing in bias detection in UN Security Council reports using Entman's framing theory. Your task is to analyze text for potential bias in reporting on conflicts, particularly in the context of the Israel-Lebanon border region.

Key Analysis Areas:
1. **Framing Bias**: How events are presented and contextualized
2. **Selection Bias**: What information is included or excluded
3. **Omission Bias**: What important information might be missing
4. **Actor Coverage**: Balance in how different actors are portrayed

Guidelines:
- Be objective and analytical
- Focus on structural patterns rather than individual word choices
- Consider the broader context of UN reporting
- Identify both explicit and implicit bias indicators
- Provide specific evidence for bias claims
- Balance criticism with acknowledgment of neutral reporting

Use Entman's four framing functions:
1. Define problems (what's wrong)
2. Diagnose causes (who's responsible)
3. Make moral judgments (evaluate actions)
4. Suggest remedies (what should be done)"""

def get_user_prompt(analysis_type, content, resolution_cache=None):
    """Get user prompt for bias analysis."""
    return f"""Analyze the following paragraph from a UN Security Council report for potential bias using Entman's framing theory:

PARAGRAPH TEXT:
{content}

ANALYSIS REQUIREMENTS:
1. Identify all actors mentioned and their roles
2. Analyze the framing of events and actions
3. Assess balance in actor coverage
4. Identify potential bias indicators
5. Provide specific evidence for conclusions
6. Consider the broader context of UN reporting

Please provide a comprehensive bias analysis following the specified JSON schema."""

async def call_openai_api(content, function_schema, analysis_type, resolution_cache=None, rate_limiter=None):
    """Call OpenAI API with function calling for structured extraction."""
    if rate_limiter:
        await rate_limiter.acquire()
    
    try:
        for attempt in range(3):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": get_system_prompt(analysis_type)},
                        {"role": "user", "content": get_user_prompt(analysis_type, content, resolution_cache)}
                    ],
                    functions=[function_schema],
                    function_call={"name": function_schema["name"]},
                    temperature=0,  # Maximum precision
                    max_tokens=4000,  # Comprehensive analysis
                    top_p=0.9,  # Focus on most likely tokens
                    frequency_penalty=0.1,  # Reduce repetition
                    presence_penalty=0.1  # Encourage diverse coverage
                )
                
                if response.choices and response.choices[0].message.function_call:
                    function_call = response.choices[0].message.function_call
                    return json.loads(function_call.arguments)
                else:
                    logger.warning(f"No function call in response, attempt {attempt + 1}")
                    
            except Exception as e:
                logger.warning(f"API call failed, attempt {attempt + 1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise e
                    
    except Exception as e:
        logger.error(f"All API attempts failed: {e}")
        return None
    finally:
        if rate_limiter:
            rate_limiter.release()

async def process_paragraph_batch(paragraphs_batch, rate_limiter, resolution_cache, output_file, report_info):
    """Process a batch of paragraphs concurrently."""
    tasks = []
    
    for para in paragraphs_batch:
        content = para.get("text", "")
        paragraph_id = para.get("paragraph_id", "unknown")
        
        if not content.strip():
            continue
            
        task = asyncio.create_task(
            process_single_paragraph(content, paragraph_id, rate_limiter, resolution_cache, report_info)
        )
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Write results to file
    async with aiofiles.open(output_file, 'a', encoding='utf-8') as f:
        for result in results:
            if isinstance(result, dict) and result:
                await f.write(json.dumps(result, ensure_ascii=False) + '\n')
            elif isinstance(result, Exception):
                logger.error(f"Paragraph processing failed: {result}")

async def process_single_paragraph(content, paragraph_id, rate_limiter, resolution_cache, report_info):
    """Process a single paragraph."""
    try:
        logger.info(f"Processing paragraph {paragraph_id} ({len(content)} chars)")
        
        bias_result = await call_openai_api(
            content, extract_bias_info(), "bias", resolution_cache, rate_limiter
        )
        
        if bias_result:
            bias_result.update({
                "paragraph_id": paragraph_id,
                "text": content,
                **report_info
            })
            
            # Add legal grounding to bias analysis
            try:
                from legal_violation_mapper import LegalViolationMapper
                mapper = LegalViolationMapper()
                enhanced_bias = mapper.analyze_bias_with_legal_grounding(bias_result)
                bias_result.update(enhanced_bias)
                logger.info(f"Added legal grounding to bias analysis for paragraph {paragraph_id}")
            except Exception as e:
                logger.warning(f"Could not add legal grounding to bias analysis: {e}")
            
            logger.info(f"Completed bias analysis for paragraph {paragraph_id}")
            return bias_result
        else:
            logger.warning(f"Failed to get bias analysis for paragraph {paragraph_id}")
            return None
            
    except Exception as e:
        logger.error(f"Error processing paragraph {paragraph_id}: {e}")
        return None

async def process_paragraphs_optimized(test_mode=True, specific_file=None, output_dir=None):
    """Process all paragraphs with optimized parallel processing."""
    resolution_cache = load_resolution_cache()
    rate_limiter = RateLimiter(RATE_LIMIT, MAX_CONCURRENT_REQUESTS)
    
    # Use provided input file or default directory
    if specific_file:
        input_dir = os.path.dirname(specific_file)
        files = [os.path.basename(specific_file)]
    else:
        input_dir = "extraction/JSONL_outputs"
        files = [f for f in os.listdir(input_dir) if f.endswith('.jsonl')]
    
    # Use provided output directory or default
    if output_dir:
        bias_output = os.path.join(output_dir, "text_bias_analysis_results.jsonl")
    else:
        bias_output = "extraction/API_output/bias_analysis.jsonl"
    
    # Clear output file
    if os.path.exists(bias_output):
        os.remove(bias_output)

    total_processed = 0
    start_time = time.time()

    for file_idx, filename in enumerate(files):
        if not filename.endswith(".jsonl"):
            continue
        file_path = os.path.join(input_dir, filename)

        logger.info(f"Processing {filename} ({file_idx + 1}/{len(files)})")
        paragraphs = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    paragraphs.append(json.loads(line))
        
        if not paragraphs:
            logger.warning(f"No paragraphs found in {filename}")
            continue
        
        logger.info(f"Found {len(paragraphs)} paragraphs in {filename}")
        
        # Limit paragraphs for test mode
        if test_mode:
            paragraphs = paragraphs[:3]
            logger.info(f"Test mode: processing only first 3 paragraphs")
        
        report_info = {}
        
        # Process paragraphs in batches
        for i in range(0, len(paragraphs), BATCH_SIZE):
            batch = paragraphs[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(paragraphs) + BATCH_SIZE - 1)//BATCH_SIZE}")
            
            await process_paragraph_batch(batch, rate_limiter, resolution_cache, bias_output, report_info)
            total_processed += len(batch)
            
            # Progress update
            elapsed = time.time() - start_time
            avg_time_per_para = elapsed / total_processed if total_processed > 0 else 0
            remaining_paras = len(paragraphs) - total_processed
            eta = remaining_paras * avg_time_per_para
            
            logger.info(f"Progress: {total_processed}/{len(paragraphs)} paragraphs processed")
            logger.info(f"Average time per paragraph: {avg_time_per_para:.1f}s")
            logger.info(f"Estimated time remaining: {eta/60:.1f} minutes")
    
    total_time = time.time() - start_time
    logger.info(f"🎉 Optimized bias analysis completed!")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Total paragraphs processed: {total_processed}")
    logger.info(f"Average time per paragraph: {total_time/total_processed:.1f}s")
    logger.info(f"Results saved to: {bias_output}")
    
    return bias_output

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized text-based bias analysis with OpenAI API")
    parser.add_argument("--input", type=str, help="Input JSONL file path")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--test", action="store_true", help="Run in test mode (process only first 3 paragraphs)")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Maximum concurrent API requests")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Update configuration based on arguments
    MAX_CONCURRENT_REQUESTS = args.max_concurrent
    BATCH_SIZE = args.batch_size
    
    # Use provided arguments or defaults
    input_file = args.input if args.input else None
    output_dir = args.output if args.output else None
    test_mode = args.test
    
    asyncio.run(process_paragraphs_optimized(
        test_mode=test_mode, 
        specific_file=input_file, 
        output_dir=output_dir
    )) 