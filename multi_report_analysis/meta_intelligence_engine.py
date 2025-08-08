#!/usr/bin/env python3
"""
Meta-Intelligence Engine
Revolutionary AI system that generates meta-insights from cross-report patterns
Uses single GPT-4o call for maximum intelligence with minimum cost
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv

class MetaIntelligenceEngine:
    """
    Revolutionary system that uses GPT-4o to analyze SUMMARIES instead of raw text
    Generates insights impossible from single reports
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the meta-intelligence engine"""
        # Load environment variables
        load_dotenv()
        
        # Initialize API key
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Meta-analysis prompt template
        self.meta_analysis_prompt = """You are a revolutionary AI analyst specializing in institutional bias detection and predictive intelligence. You have access to comprehensive analysis results from multiple UN Security Council reports and can detect meta-patterns invisible to single-report analysis.

## YOUR REVOLUTIONARY CAPABILITIES
- **Institutional Bias Detection**: Identify systematic bias patterns across multiple reports
- **Predictive Intelligence**: Forecast future bias patterns based on historical evolution
- **Meta-Pattern Recognition**: Detect correlations and patterns across different analysis dimensions
- **Hidden Correlation Discovery**: Find non-obvious relationships between variables
- **Bias Fingerprinting**: Create unique bias signatures for different periods/authors

## ANALYSIS MISSION
Generate revolutionary meta-insights that would be impossible from analyzing individual reports separately. Focus on:

1. **INSTITUTIONAL BIAS FINGERPRINTING**: Systematic patterns that reveal institutional bias
2. **PREDICTIVE FORECASTING**: Data-driven predictions of future bias patterns
3. **HIDDEN CORRELATIONS**: Non-obvious relationships between actors, time periods, events
4. **EVOLUTION DETECTION**: How bias patterns change over time and why
5. **MANIPULATION INDICATORS**: Signs of coordinated bias campaigns or systematic framing

## AVAILABLE INTELLIGENCE
{intelligence_summary}

## ANALYSIS REQUIREMENTS
Provide a comprehensive meta-intelligence analysis with these sections:

### 1. INSTITUTIONAL BIAS FINGERPRINT (3-4 paragraphs)
- Systematic bias patterns that span multiple reports
- Consistent framing asymmetries between actors
- Institutional preferences in coverage and emphasis
- Quantified bias metrics and statistical evidence

### 2. PREDICTIVE INTELLIGENCE FORECASTING (2-3 paragraphs)
- Data-driven predictions for future report bias patterns
- Probability estimates for specific bias manifestations
- Early warning indicators for bias pattern shifts
- Confidence levels and supporting evidence

### 3. HIDDEN CORRELATION DISCOVERY (3-4 paragraphs)
- Non-obvious relationships between variables
- Temporal correlations with external events
- Geographic bias correlations
- Actor treatment correlations with political events

### 4. BIAS EVOLUTION TIMELINE (2-3 paragraphs)
- How bias patterns have evolved over the analyzed period
- Inflection points and pattern shifts
- Causal factors behind evolution changes
- Trajectory analysis and trend projections

### 5. SYSTEMATIC MANIPULATION INDICATORS (2-3 paragraphs)
- Evidence of coordinated bias campaigns
- Consistent framing strategies across reports
- Statistical anomalies suggesting systematic bias
- Institutional policy influence indicators

### 6. REVOLUTIONARY INSIGHTS SUMMARY (2 paragraphs)
- Most significant discoveries from the meta-analysis
- Insights that would be impossible from single-report analysis
- Strategic implications for understanding institutional bias
- Recommendations for future bias detection

## OUTPUT REQUIREMENTS
- Base ALL analysis on the provided data and patterns
- Use specific statistics and evidence from the intelligence summary
- Provide probability estimates and confidence levels where appropriate
- Maintain scientific rigor while delivering revolutionary insights
- Use clear, professional language suitable for academic publication

Generate your revolutionary meta-intelligence analysis:"""

    def generate_meta_intelligence(self, individual_results: List[Dict], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate revolutionary meta-intelligence from individual results and patterns
        
        Args:
            individual_results: List of individual report analysis results
            patterns: Cross-report pattern analysis results
            
        Returns:
            Revolutionary meta-intelligence insights
        """
        print("    🧠 Creating intelligent summary for meta-analysis...")
        
        # Create intelligent summary that fits in single prompt
        intelligence_summary = self.create_intelligent_summary(individual_results, patterns)
        
        print("    🤖 Calling GPT-4o for revolutionary meta-analysis...")
        
        # Single high-value GPT-4o call for meta-intelligence
        try:
            meta_insights = self.call_gpt4o_for_meta_analysis(intelligence_summary)
            
            # Structure the response
            structured_insights = {
                "timestamp": datetime.now().isoformat(),
                "analysis_scope": {
                    "reports_analyzed": len(individual_results),
                    "successful_analyses": len([r for r in individual_results if "error" not in r]),
                    "pattern_categories": len(patterns) if "error" not in patterns else 0,
                    "intelligence_summary_length": len(intelligence_summary)
                },
                "meta_intelligence": meta_insights,
                "confidence_assessment": self.assess_analysis_confidence(individual_results, patterns),
                "methodology": {
                    "approach": "hierarchical_intelligence_synthesis",
                    "ai_model": "gpt-4o",
                    "analysis_type": "meta_pattern_detection",
                    "data_sources": ["quantitative_extraction", "bias_analysis", "temporal_patterns", "geographic_patterns"]
                }
            }
            
            return structured_insights
            
        except Exception as e:
            return {
                "error": f"Meta-intelligence generation failed: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "analysis_scope": {
                    "reports_analyzed": len(individual_results),
                    "intelligence_summary_length": len(intelligence_summary)
                }
            }
    
    def create_intelligent_summary(self, individual_results: List[Dict], patterns: Dict[str, Any]) -> str:
        """
        Create an intelligent summary that captures all key information
        while fitting in a single GPT-4o prompt
        """
        
        # Extract key metrics from individual results
        successful_results = [r for r in individual_results if "error" not in r]
        
        summary_parts = []
        
        # 1. Overall Analysis Scope
        summary_parts.append(f"## ANALYSIS SCOPE")
        summary_parts.append(f"Reports Analyzed: {len(individual_results)}")
        summary_parts.append(f"Successful Analyses: {len(successful_results)}")
        
        if successful_results:
            total_data_points = sum(r["summary"]["total_data_points"] for r in successful_results)
            avg_confidence = sum(r["summary"]["average_confidence"] for r in successful_results) / len(successful_results)
            
            summary_parts.append(f"Total Data Points Extracted: {total_data_points}")
            summary_parts.append(f"Average Confidence Score: {avg_confidence:.3f}")
        
        # 2. Individual Report Summaries
        summary_parts.append(f"\n## INDIVIDUAL REPORT SUMMARIES")
        for result in successful_results[:10]:  # Limit to prevent prompt overflow
            summary_parts.append(f"\n### {result['report_name']}")
            summary_parts.append(f"- Data Points: {result['summary']['total_data_points']}")
            summary_parts.append(f"- Confidence: {result['summary']['average_confidence']:.3f}")
            summary_parts.append(f"- Categories: {result['summary']['categories_found']}")
            summary_parts.append(f"- Actors: {result['summary']['actors_identified']}")
            summary_parts.append(f"- Bias Instances: {result['summary']['bias_paragraphs']}")
            
            # Key quantitative findings
            if result["quantitative_data"]:
                top_incidents = sorted(result["quantitative_data"], 
                                     key=lambda x: x.get("value", 0) if isinstance(x.get("value"), (int, float)) else 0, 
                                     reverse=True)[:3]
                summary_parts.append("- Top Incidents:")
                for incident in top_incidents:
                    summary_parts.append(f"  * {incident.get('category', 'unknown')}: {incident.get('value', 0)} {incident.get('unit', '')} by {incident.get('actor', 'unknown')}")
            
            # Key bias findings
            if result["bias_data"]:
                bias_types = [item.get("bias_flag", "") for item in result["bias_data"]]
                bias_actors = []
                for item in result["bias_data"]:
                    bias_actors.extend(item.get("core_actors", []))
                
                summary_parts.append(f"- Bias Types: {list(set(bias_types))}")
                summary_parts.append(f"- Bias Actors: {list(set(bias_actors))}")
        
        # 3. Cross-Report Pattern Analysis
        if "error" not in patterns:
            summary_parts.append(f"\n## CROSS-REPORT PATTERNS")
            
            # Temporal patterns
            if "temporal_evolution" in patterns and "error" not in patterns["temporal_evolution"]:
                temporal = patterns["temporal_evolution"]
                summary_parts.append(f"\n### Temporal Evolution Patterns")
                
                if "violation_trends" in temporal:
                    summary_parts.append("- Violation Trends:")
                    for category, trend_data in temporal["violation_trends"].items():
                        summary_parts.append(f"  * {category}: {trend_data.get('trend_direction', 'unknown')} ({trend_data.get('change_percent', 0):+.1f}%)")
                
                if "actor_evolution" in temporal:
                    summary_parts.append("- Actor Evolution:")
                    for actor, evolution in temporal["actor_evolution"].items():
                        summary_parts.append(f"  * {actor}: {evolution.get('activity_trend', 'unknown')} ({evolution.get('activity_change_percent', 0):+.1f}%)")
                
                if "ceasefire_impact" in temporal and "error" not in temporal["ceasefire_impact"]:
                    ceasefire = temporal["ceasefire_impact"]
                    summary_parts.append(f"- Ceasefire Impact: {ceasefire.get('average_reduction_percent', 0):.1f}% reduction, effectiveness: {ceasefire.get('ceasefire_effectiveness', 'unknown')}")
            
            # Geographic patterns
            if "geographic_clustering" in patterns and "error" not in patterns["geographic_clustering"]:
                geo = patterns["geographic_clustering"]
                summary_parts.append(f"\n### Geographic Patterns")
                
                if "location_hotspots" in geo:
                    summary_parts.append("- Top Incident Locations:")
                    for location, data in list(geo["location_hotspots"].items())[:5]:
                        summary_parts.append(f"  * {location}: {data.get('incident_count', 0)} incidents")
                
                if "sector_patterns" in geo:
                    summary_parts.append("- Sector Analysis:")
                    for sector, data in geo["sector_patterns"].items():
                        summary_parts.append(f"  * {sector}: {data.get('incident_count', 0)} incidents, avg value: {data.get('average_incident_value', 0)}")
            
            # Actor behavior evolution
            if "actor_behavior_evolution" in patterns:
                summary_parts.append(f"\n### Actor Behavior Evolution")
                for actor, evolution in patterns["actor_behavior_evolution"].items():
                    summary_parts.append(f"- {actor}: {evolution.get('evolution_summary', 'No summary available')}")
            
            # Bias pattern shifts
            if "bias_pattern_shifts" in patterns and "error" not in patterns["bias_pattern_shifts"]:
                bias_shifts = patterns["bias_pattern_shifts"]
                summary_parts.append(f"\n### Bias Pattern Evolution")
                
                if "bias_type_evolution" in bias_shifts:
                    summary_parts.append("- Bias Type Trends:")
                    for bias_type, trend_data in bias_shifts["bias_type_evolution"].items():
                        summary_parts.append(f"  * {bias_type}: {trend_data.get('trend', 'unknown')} (latest: {trend_data.get('latest_count', 0)})")
                
                if "actor_bias_evolution" in bias_shifts:
                    summary_parts.append("- Actor Bias Evolution:")
                    for actor, bias_data in bias_shifts["actor_bias_evolution"].items():
                        summary_parts.append(f"  * {actor}: {bias_data.get('bias_mention_trend', 'unknown')} bias mentions")
        
        # Join all parts
        intelligence_summary = "\n".join(summary_parts)
        
        # Ensure it's not too long for GPT-4o (rough estimate: ~100K tokens max)
        if len(intelligence_summary) > 80000:  # Conservative limit
            # Truncate while preserving structure
            intelligence_summary = intelligence_summary[:80000] + "\n\n[SUMMARY TRUNCATED FOR LENGTH]"
        
        return intelligence_summary
    
    def call_gpt4o_for_meta_analysis(self, intelligence_summary: str) -> str:
        """
        Make the revolutionary GPT-4o call for meta-intelligence generation
        """
        
        # Create the complete prompt
        complete_prompt = self.meta_analysis_prompt.format(
            intelligence_summary=intelligence_summary
        )
        
        # Call GPT-4o with optimized parameters for meta-analysis
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": complete_prompt}],
            temperature=0.3,  # Slightly higher for creative insights while maintaining accuracy
            max_tokens=4000,  # Maximum comprehensive analysis
            top_p=0.9,       # Focus on most likely tokens
            frequency_penalty=0.1,  # Reduce repetition
            presence_penalty=0.2    # Encourage diverse insights
        )
        
        return response.choices[0].message.content
    
    def assess_analysis_confidence(self, individual_results: List[Dict], patterns: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess the confidence level of the meta-analysis
        """
        
        successful_results = [r for r in individual_results if "error" not in r]
        
        # Calculate confidence factors
        data_quality_score = 0
        if successful_results:
            avg_confidence = sum(r["summary"]["average_confidence"] for r in successful_results) / len(successful_results)
            data_quality_score = avg_confidence
        
        pattern_quality_score = 0.5  # Default
        if "error" not in patterns:
            # Count successful pattern analyses
            successful_patterns = sum(1 for key, value in patterns.items() 
                                    if isinstance(value, dict) and "error" not in value)
            total_patterns = len([key for key in patterns.keys() if not key.endswith("_summary")])
            pattern_quality_score = successful_patterns / max(total_patterns, 1)
        
        sample_size_score = min(len(successful_results) / 5.0, 1.0)  # Optimal at 5+ reports
        
        overall_confidence = (data_quality_score + pattern_quality_score + sample_size_score) / 3
        
        return {
            "overall_confidence": round(overall_confidence, 3),
            "data_quality_score": round(data_quality_score, 3),
            "pattern_quality_score": round(pattern_quality_score, 3),
            "sample_size_score": round(sample_size_score, 3),
            "confidence_level": (
                "high" if overall_confidence >= 0.8 else
                "medium" if overall_confidence >= 0.6 else
                "low"
            ),
            "reliability_factors": {
                "reports_analyzed": len(individual_results),
                "successful_analyses": len(successful_results),
                "pattern_categories": len(patterns) if "error" not in patterns else 0
            }
        }


def main():
    """Test the meta-intelligence engine"""
    print("Meta-Intelligence Engine - Test Mode")
    
    # This would be used for testing with sample data
    engine = MetaIntelligenceEngine()
    print(f"✅ Meta-Intelligence Engine initialized successfully")


if __name__ == "__main__":
    main()