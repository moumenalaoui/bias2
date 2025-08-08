#!/usr/bin/env python3
"""
Multi-Report Batch Processor
Coordinates the 3-layer hierarchical analysis for revolutionary cross-report insights
"""

import json
import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import shutil
import tempfile

# Add paths for importing modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))
sys.path.append(str(current_dir.parent))
sys.path.append(str(current_dir.parent / "extraction" / "scripts"))

class MultiReportBatchProcessor:
    """
    Main orchestrator for multi-report analysis using hierarchical intelligence
    """
    
    def __init__(self, base_dir: str = None):
        """Initialize the batch processor"""
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.results_dir = self.base_dir / "precision_hybrid_results"
        self.individual_dir = self.results_dir / "individual_reports"
        self.patterns_dir = self.results_dir / "cross_report_patterns"
        self.meta_dir = self.results_dir / "meta_intelligence"
        
        # Ensure directories exist
        for dir_path in [self.individual_dir, self.patterns_dir, self.meta_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_multiple_reports(self, pdf_files: List[str], report_names: List[str] = None) -> Dict[str, Any]:
        """
        Main orchestrator function for multi-report analysis
        
        Args:
            pdf_files: List of PDF file paths
            report_names: Optional list of report names (defaults to filenames)
            
        Returns:
            Complete multi-report analysis results
        """
        print(f"🚀 Starting Multi-Report Hierarchical Analysis")
        print(f"📊 Processing {len(pdf_files)} reports...")
        
        # Generate report names if not provided
        if not report_names:
            report_names = [Path(pdf).stem for pdf in pdf_files]
        
        # Phase 1: Individual Report Analysis (using existing pipeline)
        print("\n📋 Phase 1: Individual Report Analysis")
        individual_results = []
        
        for i, (pdf_file, report_name) in enumerate(zip(pdf_files, report_names), 1):
            print(f"  Processing {i}/{len(pdf_files)}: {report_name}")
            
            try:
                result = self.run_individual_analysis(pdf_file, report_name)
                individual_results.append(result)
                print(f"    ✅ {result['summary']['total_data_points']} data points, "
                      f"avg confidence: {result['summary']['average_confidence']:.3f}")
            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
                individual_results.append({
                    "report_name": report_name,
                    "error": str(e),
                    "status": "failed"
                })
        
        # Phase 2: Cross-Report Pattern Synthesis
        print(f"\n🔍 Phase 2: Cross-Report Pattern Synthesis")
        successful_results = [r for r in individual_results if "error" not in r]
        
        if len(successful_results) < 2:
            print("    ⚠️ Need at least 2 successful reports for pattern analysis")
            patterns = {"error": "Insufficient data for pattern analysis"}
        else:
            from cross_report_analyzer import CrossReportAnalyzer
            analyzer = CrossReportAnalyzer()
            patterns = analyzer.synthesize_patterns(successful_results)
            print(f"    ✅ Pattern synthesis completed")
            
            # Save pattern results
            patterns_file = self.patterns_dir / f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns, f, indent=2, default=str)
        
        # Phase 3: Meta-Intelligence Generation
        print(f"\n🧠 Phase 3: Meta-Intelligence Generation")
        
        if "error" in patterns:
            print("    ⚠️ Skipping meta-intelligence due to pattern analysis failure")
            meta_insights = {"error": "Pattern analysis failed"}
        else:
            try:
                from meta_intelligence_engine import MetaIntelligenceEngine
                meta_engine = MetaIntelligenceEngine()
                meta_insights = meta_engine.generate_meta_intelligence(successful_results, patterns)
                print(f"    ✅ Meta-intelligence generated")
                
                # Save meta-intelligence results
                meta_file = self.meta_dir / f"meta_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(meta_insights, f, indent=2, default=str)
                    
            except Exception as e:
                print(f"    ❌ Meta-intelligence failed: {str(e)}")
                meta_insights = {"error": str(e)}
        
        # Compile final results
        final_results = {
            "timestamp": datetime.now().isoformat(),
            "reports_processed": len(pdf_files),
            "successful_analyses": len(successful_results),
            "individual_analyses": individual_results,
            "cross_report_patterns": patterns,
            "meta_intelligence": meta_insights,
            "summary_stats": self.generate_summary_stats(individual_results)
        }
        
        # Save complete results
        complete_file = self.results_dir / f"complete_multi_report_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"\n🎉 Multi-Report Analysis Complete!")
        print(f"📁 Results saved to: {complete_file}")
        
        return final_results
    
    def run_individual_analysis(self, pdf_file: str, report_name: str) -> Dict[str, Any]:
        """
        Run individual report analysis using existing pipeline
        """
        # Create temporary directory for this report
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_pdf = Path(temp_dir) / f"{report_name}.pdf"
            shutil.copy2(pdf_file, temp_pdf)
            
            # Run the existing pipeline
            cmd = ["bash", str(self.base_dir / "run_bias2.sh"), str(temp_pdf)]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir,
                    timeout=1800  # 30 minute timeout
                )
                
                if result.returncode != 0:
                    raise Exception(f"Pipeline failed: {result.stderr}")
                
                # Load the results from precision_hybrid_results
                analysis_results = self.load_individual_results(report_name)
                
                # Move results to individual_reports directory
                self.archive_individual_results(report_name)
                
                return analysis_results
                
            except subprocess.TimeoutExpired:
                raise Exception("Analysis timed out after 30 minutes")
            except Exception as e:
                raise Exception(f"Analysis failed: {str(e)}")
    
    def load_individual_results(self, report_name: str) -> Dict[str, Any]:
        """Load results from the standard precision_hybrid_results directory"""
        
        results = {
            "report_name": report_name,
            "timestamp": datetime.now().isoformat(),
            "quantitative_data": [],
            "bias_data": [],
            "ai_analysis": {},
            "summary": {}
        }
        
        # Load quantitative data
        quant_files = [
            self.results_dir / "precision_hybrid_extraction_results.jsonl",
            self.results_dir / "disambiguated_results.jsonl"
        ]
        
        for file_path in quant_files:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            results["quantitative_data"].append(json.loads(line))
                break
        
        # Load bias data
        bias_file = self.results_dir / "text_bias_analysis_results.jsonl"
        if bias_file.exists():
            with open(bias_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        results["bias_data"].append(json.loads(line))
        
        # Load AI analysis
        ai_file = self.results_dir / "ai_analysis_report.json"
        if ai_file.exists():
            with open(ai_file, 'r', encoding='utf-8') as f:
                results["ai_analysis"] = json.load(f)
        
        # Generate summary
        results["summary"] = {
            "total_data_points": len(results["quantitative_data"]),
            "bias_paragraphs": len(results["bias_data"]),
            "categories_found": len(set(item.get('category', '') for item in results["quantitative_data"])),
            "actors_identified": len(set(item.get('actor', '') for item in results["quantitative_data"])),
            "average_confidence": sum(item.get('confidence_score', 0) for item in results["quantitative_data"]) / max(len(results["quantitative_data"]), 1)
        }
        
        return results
    
    def archive_individual_results(self, report_name: str):
        """Move individual results to archived directory"""
        
        archive_dir = self.individual_dir / report_name
        archive_dir.mkdir(exist_ok=True)
        
        # Files to archive
        files_to_archive = [
            "precision_hybrid_extraction_results.jsonl",
            "disambiguated_results.jsonl", 
            "text_bias_analysis_results.jsonl",
            "ai_analysis_report.json",
            "gpt_quant_extraction_output.jsonl",
            "extraction_summary.json"
        ]
        
        for filename in files_to_archive:
            source = self.results_dir / filename
            if source.exists():
                dest = archive_dir / filename
                shutil.move(str(source), str(dest))
    
    def generate_summary_stats(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Generate overall summary statistics"""
        
        successful = [r for r in individual_results if "error" not in r]
        
        if not successful:
            return {"error": "No successful analyses"}
        
        total_data_points = sum(r["summary"]["total_data_points"] for r in successful)
        total_bias_paragraphs = sum(r["summary"]["bias_paragraphs"] for r in successful)
        avg_confidence = sum(r["summary"]["average_confidence"] for r in successful) / len(successful)
        
        # Collect all categories and actors
        all_categories = set()
        all_actors = set()
        
        for result in successful:
            for item in result["quantitative_data"]:
                all_categories.add(item.get('category', ''))
                all_actors.add(item.get('actor', ''))
        
        return {
            "reports_analyzed": len(successful),
            "total_data_points": total_data_points,
            "total_bias_paragraphs": total_bias_paragraphs,
            "average_confidence_across_all": avg_confidence,
            "unique_categories": len(all_categories),
            "unique_actors": len(all_actors),
            "categories_found": sorted(list(all_categories)),
            "actors_identified": sorted(list(all_actors))
        }


def main():
    """Command line interface for batch processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Report Batch Processor")
    parser.add_argument("pdf_files", nargs="+", help="PDF files to process")
    parser.add_argument("--names", nargs="*", help="Report names (optional)")
    parser.add_argument("--output-dir", help="Output directory")
    
    args = parser.parse_args()
    
    processor = MultiReportBatchProcessor(args.output_dir)
    results = processor.process_multiple_reports(args.pdf_files, args.names)
    
    print(f"\n📊 Final Summary:")
    print(f"   Reports processed: {results['reports_processed']}")
    print(f"   Successful analyses: {results['successful_analyses']}")
    print(f"   Total data points: {results['summary_stats'].get('total_data_points', 0)}")
    print(f"   Average confidence: {results['summary_stats'].get('average_confidence_across_all', 0):.3f}")


if __name__ == "__main__":
    main()