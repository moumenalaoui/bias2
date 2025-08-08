#!/usr/bin/env python3
"""
AI Analysis Agent for UN Report Bias Analysis
Provides intelligent, factually-grounded analysis based on Entman's framing theory
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import openai
from pathlib import Path
from dotenv import load_dotenv

class AIAnalysisAgent:
    """
    AI agent that provides intelligent analysis of UN report bias analysis results
    based on Entman's framing theory and quantitative data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the AI analysis agent."""
        # Load environment variables
        load_dotenv()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Analysis prompt template
        self.analysis_prompt_template = """You are an expert analyst specializing in media bias analysis using Entman's framing theory (1993). Your task is to provide a comprehensive, factually-grounded analysis of UN Security Council report bias analysis results.

## CONTEXT
You have access to:
1. **Quantitative Data**: Extracted numerical data points with actors, values, and legal violations
2. **Bias Analysis**: Entman framing theory analysis of each paragraph
3. **Legal Grounding**: UNSCR 1701 violation mappings

## ANALYSIS REQUIREMENTS
Provide a comprehensive analysis that includes:

### 1. EXECUTIVE SUMMARY (2-3 paragraphs)
- Key findings and patterns
- Overall bias assessment
- Most significant violations and actors

### 2. QUANTITATIVE INSIGHTS (2-3 paragraphs)
- Statistical patterns in the data
- Actor distribution and responsibility
- Scale and scope of violations

### 3. FRAMING BIAS ANALYSIS (3-4 paragraphs)
- How Entman's framing theory applies to this report
- Selection bias patterns (what's included/excluded)
- Framing bias patterns (how events are presented)
- Omission bias patterns (what's missing)

### 4. LEGAL VIOLATION ASSESSMENT (2-3 paragraphs)
- UNSCR 1701 compliance analysis
- Most frequent violation types
- Legal implications and recommendations

### 5. ACTOR ANALYSIS (2-3 paragraphs)
- Comparative analysis of different actors
- Responsibility distribution
- Potential bias in actor representation

### 6. METHODOLOGICAL ASSESSMENT (1-2 paragraphs)
- Strengths and limitations of the analysis
- Data quality assessment
- Recommendations for improvement

## GUIDELINES
- Base ALL analysis on the provided datasets
- Use specific numbers and quotes from the data
- Apply Entman's framing theory rigorously
- Maintain objectivity and academic rigor
- Provide actionable insights
- Use clear, professional language

## DATASETS TO ANALYZE

### QUANTITATIVE DATA:
{quantitative_data}

### BIAS ANALYSIS DATA:
{bias_data}

### LEGAL GROUNDING SUMMARY:
{legal_summary}

Provide your comprehensive analysis:"""

    def load_quantitative_data(self, file_path: str) -> List[Dict]:
        """Load quantitative extraction results."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def load_bias_data(self, file_path: str) -> List[Dict]:
        """Load bias analysis results."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

    def generate_quantitative_summary(self, data: List[Dict]) -> str:
        """Generate a summary of quantitative data for the prompt."""
        if not data:
            return "No quantitative data available."
        
        # Group by category
        categories = {}
        actors = {}
        violations = {}
        
        for item in data:
            category = item.get('category', 'unknown')
            actor = item.get('actor', 'unknown')
            violation = item.get('legal_article_violated', 'unknown')
            
            if category not in categories:
                categories[category] = []
            categories[category].append(item)
            
            if actor not in actors:
                actors[actor] = []
            actors[actor].append(item)
            
            if violation not in violations:
                violations[violation] = []
            violations[violation].append(item)
        
        summary = f"Total data points: {len(data)}\n\n"
        
        # Category breakdown
        summary += "CATEGORY BREAKDOWN:\n"
        for category, items in categories.items():
            total_value = sum(float(item.get('value', 0)) for item in items if str(item.get('value', '')).replace('.', '').isdigit())
            summary += f"- {category}: {len(items)} items, total value: {total_value:,.0f}\n"
        
        # Actor breakdown
        summary += "\nACTOR BREAKDOWN:\n"
        for actor, items in actors.items():
            summary += f"- {actor}: {len(items)} mentions\n"
        
        # Top violations
        summary += "\nTOP LEGAL VIOLATIONS:\n"
        for violation, items in violations.items():
            summary += f"- {violation}: {len(items)} instances\n"
        
        return summary

    def generate_bias_summary(self, data: List[Dict]) -> str:
        """Generate a summary of bias analysis data for the prompt."""
        if not data:
            return "No bias analysis data available."
        
        # Bias type distribution
        bias_types = {}
        actors_mentioned = set()
        violation_types = set()
        
        for item in data:
            bias_flag = item.get('bias_flag', 'unknown')
            if bias_flag not in bias_types:
                bias_types[bias_flag] = 0
            bias_types[bias_flag] += 1
            
            # Collect actors
            core_actors = item.get('core_actors', [])
            actors_mentioned.update(core_actors)
            
            # Collect violation types
            violation_subtypes = item.get('violation_subtypes', [])
            violation_types.update(violation_subtypes)
        
        summary = f"Total paragraphs analyzed: {len(data)}\n\n"
        
        # Bias type breakdown
        summary += "BIAS TYPE DISTRIBUTION:\n"
        for bias_type, count in bias_types.items():
            summary += f"- {bias_type}: {count} paragraphs\n"
        
        # Actors mentioned
        summary += f"\nACTORS MENTIONED: {', '.join(sorted(actors_mentioned))}\n"
        
        # Sample bias reasons
        summary += "\nSAMPLE BIAS REASONS:\n"
        for i, item in enumerate(data[:3]):  # First 3 examples
            summary += f"- Paragraph {item.get('paragraph_id', 'unknown')}: {item.get('bias_reason', 'No reason provided')[:200]}...\n"
        
        return summary

    def generate_legal_summary(self, quantitative_data: List[Dict], bias_data: List[Dict]) -> str:
        """Generate a summary of legal violations."""
        # Collect all legal violations
        violations = {}
        
        # From quantitative data
        for item in quantitative_data:
            violation = item.get('legal_article_violated', 'unknown')
            if violation not in violations:
                violations[violation] = 0
            violations[violation] += 1
        
        # From bias data
        for item in bias_data:
            legal_summary = item.get('legal_grounding_summary', '')
            if legal_summary and legal_summary != "No UNSCR 1701 violations identified":
                violations[legal_summary] = violations.get(legal_summary, 0) + 1
        
        summary = "LEGAL VIOLATION SUMMARY:\n"
        for violation, count in violations.items():
            summary += f"- {violation}: {count} instances\n"
        
        return summary

    def generate_analysis(self, quantitative_file: str, bias_file: str) -> Dict:
        """Generate comprehensive AI analysis."""
        try:
            # Load data
            quantitative_data = self.load_quantitative_data(quantitative_file)
            bias_data = self.load_bias_data(bias_file)
            
            # Generate summaries for the prompt
            quantitative_summary = self.generate_quantitative_summary(quantitative_data)
            bias_summary = self.generate_bias_summary(bias_data)
            legal_summary = self.generate_legal_summary(quantitative_data, bias_data)
            
            # Create the analysis prompt
            prompt = self.analysis_prompt_template.format(
                quantitative_data=quantitative_summary,
                bias_data=bias_summary,
                legal_summary=legal_summary
            )
            
            # Call GPT-4o for analysis
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more consistent analysis
                max_tokens=4000
            )
            
            analysis_text = response.choices[0].message.content
            
            # Create structured output
            analysis_result = {
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis_text,
                "data_summary": {
                    "quantitative_points": len(quantitative_data),
                    "bias_paragraphs": len(bias_data),
                    "categories_found": len(set(item.get('category', '') for item in quantitative_data)),
                    "actors_identified": len(set(item.get('actor', '') for item in quantitative_data)),
                    "bias_types": list(set(item.get('bias_flag', '') for item in bias_data))
                }
            }
            
            return analysis_result
            
        except Exception as e:
            return {
                "error": f"Analysis generation failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def save_analysis(self, analysis_result: Dict, output_file: str):
        """Save analysis results to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)

def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AI Analysis Agent for UN Report Bias Analysis")
    parser.add_argument("--quantitative", type=str, required=True, help="Path to quantitative results JSONL")
    parser.add_argument("--bias", type=str, required=True, help="Path to bias analysis results JSONL")
    parser.add_argument("--output", type=str, default="ai_analysis_report.json", help="Output file path")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = AIAnalysisAgent(api_key=args.api_key)
    
    # Generate analysis
    print("🤖 Generating AI analysis...")
    analysis_result = agent.generate_analysis(args.quantitative, args.bias)
    
    # Save results
    agent.save_analysis(analysis_result, args.output)
    
    if "error" in analysis_result:
        print(f"❌ Analysis failed: {analysis_result['error']}")
        sys.exit(1)
    else:
        print(f"✅ Analysis complete! Saved to {args.output}")
        print(f"📊 Data summary: {analysis_result['data_summary']}")

if __name__ == "__main__":
    main() 