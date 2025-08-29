#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_fastpath_vs_legacy.py
Evaluator script that compares Fast Path vs Legacy extraction to identify gaps
and generate improvement suggestions for the lexicon.
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

import orjson

# Add extraction scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "extraction" / "scripts"))

try:
    from fast_path_extractor import extract as fast_path_extract
    from build_patterns import create_pattern_bank
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)

# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass
class ExtractionResult:
    """Result from an extraction method"""
    method: str
    candidates: List[Dict[str, Any]]
    stats: Dict[str, Any]
    elapsed_ms: int

@dataclass
class ComparisonResult:
    """Comparison between two extraction methods"""
    fast_path: ExtractionResult
    legacy: ExtractionResult
    precision: float
    recall: float
    f1_score: float
    missed_by_fast_path: List[Dict[str, Any]]
    false_positives: List[Dict[str, Any]]
    coverage_by_family: Dict[str, Dict[str, float]]

@dataclass
class GapAnalysis:
    """Analysis of gaps in Fast Path coverage"""
    missing_patterns: List[str]
    missing_verbs: List[str]
    missing_nouns: List[str]
    missing_extras: List[str]
    suggested_additions: Dict[str, List[str]]

# ------------------------------------------------------------
# Core comparison logic
# ------------------------------------------------------------

def run_legacy_extraction(paragraphs_path: str, temp_dir: str) -> ExtractionResult:
    """Run legacy extraction method"""
    # This would call your existing hybrid_extractor.py
    # For now, we'll simulate it by reading existing results
    legacy_candidates_path = paragraphs_path.replace('.jsonl', '_legacy_candidates.ndjson')
    
    if os.path.exists(legacy_candidates_path):
        with open(legacy_candidates_path, 'rb') as f:
            candidates = [orjson.loads(line) for line in f if line.strip()]
    else:
        # Simulate legacy extraction
        candidates = []
    
    return ExtractionResult(
        method="legacy",
        candidates=candidates,
        stats={"candidates": len(candidates)},
        elapsed_ms=0
    )

def run_fast_path_extraction(paragraphs_path: str, temp_dir: str) -> ExtractionResult:
    """Run Fast Path extraction using PatternBank"""
    t0 = time.time()
    
    # Load paragraphs
    with open(paragraphs_path, 'rb') as f:
        paragraphs = [orjson.loads(line) for line in f if line.strip()]
    
    # Create PatternBank
    bank = create_pattern_bank()
    
    # Extract candidates
    all_candidates = []
    for p in paragraphs:
        pid = int(p.get("pid") or p.get("paragraph_id") or -1)
        page = int(p.get("page") or -1)
        text = p.get("text") or ""
        
        candidates = bank.find_candidates(text, pid, page)
        for c in candidates:
            all_candidates.append(asdict(c))
    
    elapsed_ms = int((time.time() - t0) * 1000)
    
    return ExtractionResult(
        method="fast_path",
        candidates=all_candidates,
        stats={"candidates": len(all_candidates)},
        elapsed_ms=elapsed_ms
    )

def normalize_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize candidate for comparison"""
    # Extract key fields for comparison
    return {
        "pid": candidate.get("pid", -1),
        "type": candidate.get("type") or candidate.get("family", ""),
        "value": candidate.get("value"),
        "unit": candidate.get("unit"),
        "actor_norm": candidate.get("actor_norm"),
        "quote": candidate.get("quote", "")[:100],  # First 100 chars for matching
    }

def calculate_metrics(fast_path: List[Dict[str, Any]], legacy: List[Dict[str, Any]]) -> Tuple[float, float, float]:
    """Calculate precision, recall, and F1 score"""
    fast_path_norm = [normalize_candidate(c) for c in fast_path]
    legacy_norm = [normalize_candidate(c) for c in legacy]
    
    # Simple matching based on key fields
    matches = 0
    for fp in fast_path_norm:
        for lg in legacy_norm:
            if (fp["pid"] == lg["pid"] and 
                fp["type"] == lg["type"] and 
                fp["value"] == lg["value"] and
                fp["unit"] == lg["unit"]):
                matches += 1
                break
    
    precision = matches / len(fast_path_norm) if fast_path_norm else 0.0
    recall = matches / len(legacy_norm) if legacy_norm else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def identify_misses(fast_path: List[Dict[str, Any]], legacy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify candidates missed by Fast Path"""
    fast_path_norm = [normalize_candidate(c) for c in fast_path]
    legacy_norm = [normalize_candidate(c) for c in legacy]
    
    missed = []
    for lg in legacy_norm:
        found = False
        for fp in fast_path_norm:
            if (fp["pid"] == lg["pid"] and 
                fp["type"] == lg["type"] and 
                fp["value"] == lg["value"] and
                fp["unit"] == lg["unit"]):
                found = True
                break
        if not found:
            # Find original legacy candidate
            for orig_lg in legacy:
                if normalize_candidate(orig_lg) == lg:
                    missed.append(orig_lg)
                    break
    
    return missed

def analyze_gaps(missed_candidates: List[Dict[str, Any]]) -> GapAnalysis:
    """Analyze gaps and suggest lexicon improvements"""
    missing_patterns = []
    missing_verbs = []
    missing_nouns = []
    missing_extras = []
    suggested_additions = defaultdict(list)
    
    for candidate in missed_candidates:
        quote = candidate.get("quote", "")
        candidate_type = candidate.get("type", "")
        
        # Analyze the quote to identify missing patterns
        if "detected" in quote.lower() and "trajectory" in quote.lower():
            if "detected" not in missing_verbs:
                missing_verbs.append("detected")
            if "trajectory" not in missing_nouns:
                missing_nouns.append("trajectory")
        
        if "observed" in quote.lower() and "air strike" in quote.lower():
            if "observed" not in missing_verbs:
                missing_verbs.append("observed")
            if "air strike" not in missing_nouns:
                missing_nouns.append("air strike")
        
        if "killed" in quote.lower() and "civilian" in quote.lower():
            if "killed" not in missing_verbs:
                missing_verbs.append("killed")
            if "civilian" not in missing_nouns:
                missing_nouns.append("civilian")
        
        # Add to suggested additions
        if candidate_type:
            suggested_additions[candidate_type].append(quote)
    
    return GapAnalysis(
        missing_patterns=missing_patterns,
        missing_verbs=missing_verbs,
        missing_nouns=missing_nouns,
        missing_extras=missing_extras,
        suggested_additions=dict(suggested_additions)
    )

def compare_extractions(paragraphs_path: str, temp_dir: str) -> ComparisonResult:
    """Compare Fast Path vs Legacy extraction"""
    logging.info("[compare] Running Fast Path extraction...")
    fast_path_result = run_fast_path_extraction(paragraphs_path, temp_dir)
    
    logging.info("[compare] Running Legacy extraction...")
    legacy_result = run_legacy_extraction(paragraphs_path, temp_dir)
    
    logging.info("[compare] Calculating metrics...")
    precision, recall, f1 = calculate_metrics(
        fast_path_result.candidates, 
        legacy_result.candidates
    )
    
    logging.info("[compare] Identifying misses...")
    missed = identify_misses(fast_path_result.candidates, legacy_result.candidates)
    
    logging.info("[compare] Analyzing gaps...")
    gap_analysis = analyze_gaps(missed)
    
    # Calculate coverage by family
    coverage_by_family = {}
    fast_path_by_family = defaultdict(list)
    legacy_by_family = defaultdict(list)
    
    for c in fast_path_result.candidates:
        family = c.get("family", c.get("type", "unknown"))
        fast_path_by_family[family].append(c)
    
    for c in legacy_result.candidates:
        family = c.get("type", "unknown")
        legacy_by_family[family].append(c)
    
    for family in set(fast_path_by_family.keys()) | set(legacy_by_family.keys()):
        fp_count = len(fast_path_by_family[family])
        lg_count = len(legacy_by_family[family])
        recall_family = fp_count / lg_count if lg_count > 0 else 0.0
        coverage_by_family[family] = {
            "fast_path_count": fp_count,
            "legacy_count": lg_count,
            "recall": recall_family
        }
    
    return ComparisonResult(
        fast_path=fast_path_result,
        legacy=legacy_result,
        precision=precision,
        recall=recall,
        f1_score=f1,
        missed_by_fast_path=missed,
        false_positives=[],  # Would need more sophisticated analysis
        coverage_by_family=coverage_by_family
    )

# ------------------------------------------------------------
# Reporting and output
# ------------------------------------------------------------

def generate_report(comparison: ComparisonResult, output_dir: str):
    """Generate detailed comparison report"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Main comparison report
    report = {
        "summary": {
            "fast_path_candidates": len(comparison.fast_path.candidates),
            "legacy_candidates": len(comparison.legacy.candidates),
            "precision": comparison.precision,
            "recall": comparison.recall,
            "f1_score": comparison.f1_score,
            "fast_path_elapsed_ms": comparison.fast_path.elapsed_ms,
            "legacy_elapsed_ms": comparison.legacy.elapsed_ms,
        },
        "coverage_by_family": comparison.coverage_by_family,
        "missed_candidates": comparison.missed_by_fast_path,
        "gap_analysis": asdict(analyze_gaps(comparison.missed_by_fast_path))
    }
    
    # Write main report
    with open(os.path.join(output_dir, "comparison_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    
    # Write missed candidates for manual review
    with open(os.path.join(output_dir, "missed_candidates.jsonl"), "w") as f:
        for candidate in comparison.missed_by_fast_path:
            f.write(json.dumps(candidate) + "\n")
    
    # Write suggested lexicon improvements
    gap_analysis = analyze_gaps(comparison.missed_by_fast_path)
    suggestions = {
        "missing_verbs": gap_analysis.missing_verbs,
        "missing_nouns": gap_analysis.missing_nouns,
        "suggested_additions": gap_analysis.suggested_additions
    }
    
    with open(os.path.join(output_dir, "lexicon_suggestions.json"), "w") as f:
        json.dump(suggestions, f, indent=2)
    
    # Print summary
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"Fast Path candidates: {len(comparison.fast_path.candidates)}")
    print(f"Legacy candidates: {len(comparison.legacy.candidates)}")
    print(f"Precision: {comparison.precision:.3f}")
    print(f"Recall: {comparison.recall:.3f}")
    print(f"F1 Score: {comparison.f1_score:.3f}")
    print(f"Fast Path time: {comparison.fast_path.elapsed_ms}ms")
    print(f"Legacy time: {comparison.legacy.elapsed_ms}ms")
    print(f"Missed candidates: {len(comparison.missed_by_fast_path)}")
    
    print(f"\n=== COVERAGE BY FAMILY ===")
    for family, stats in comparison.coverage_by_family.items():
        print(f"{family}: {stats['fast_path_count']}/{stats['legacy_count']} ({stats['recall']:.3f})")
    
    print(f"\n=== LEXICON SUGGESTIONS ===")
    if gap_analysis.missing_verbs:
        print(f"Missing verbs: {', '.join(gap_analysis.missing_verbs)}")
    if gap_analysis.missing_nouns:
        print(f"Missing nouns: {', '.join(gap_analysis.missing_nouns)}")

# ------------------------------------------------------------
# CLI interface
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Fast Path vs Legacy extraction")
    parser.add_argument("--paragraphs", required=True, help="Path to paragraphs JSONL file")
    parser.add_argument("--output_dir", default="comparison_results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    if not os.path.exists(args.paragraphs):
        print(f"Error: Paragraphs file not found: {args.paragraphs}")
        sys.exit(1)
    
    try:
        comparison = compare_extractions(args.paragraphs, args.output_dir)
        generate_report(comparison, args.output_dir)
        print(f"\nResults written to: {args.output_dir}")
    except Exception as e:
        print(f"Error during comparison: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
