#!/usr/bin/env python3
"""
Cross-Report Pattern Analyzer
Synthesizes patterns across multiple individual report analyses
Pure computation - no API calls needed
"""

import json
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import re

class CrossReportAnalyzer:
    """
    Analyzes individual report results to find cross-report patterns
    """
    
    def __init__(self):
        """Initialize the cross-report analyzer"""
        self.temporal_analyzer = TemporalPatternAnalyzer()
        self.geographic_analyzer = GeographicPatternAnalyzer()
        self.actor_analyzer = ActorEvolutionAnalyzer()
        self.bias_analyzer = BiasEvolutionAnalyzer()
        self.violation_analyzer = ViolationTrendAnalyzer()
    
    def synthesize_patterns(self, individual_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main pattern synthesis function
        
        Args:
            individual_results: List of individual report analysis results
            
        Returns:
            Comprehensive cross-report pattern analysis
        """
        print("    🔍 Analyzing temporal patterns...")
        temporal_patterns = self.temporal_analyzer.analyze(individual_results)
        
        print("    🗺️ Analyzing geographic patterns...")
        geographic_patterns = self.geographic_analyzer.analyze(individual_results)
        
        print("    👥 Analyzing actor evolution...")
        actor_patterns = self.actor_analyzer.analyze(individual_results)
        
        print("    🎭 Analyzing bias evolution...")
        bias_patterns = self.bias_analyzer.analyze(individual_results)
        
        print("    ⚖️ Analyzing violation trends...")
        violation_patterns = self.violation_analyzer.analyze(individual_results)
        
        # Compile comprehensive pattern analysis
        pattern_synthesis = {
            "analysis_timestamp": datetime.now().isoformat(),
            "reports_analyzed": len(individual_results),
            "temporal_evolution": temporal_patterns,
            "geographic_clustering": geographic_patterns,
            "actor_behavior_evolution": actor_patterns,
            "bias_pattern_shifts": bias_patterns,
            "violation_trend_analysis": violation_patterns,
            "cross_pattern_correlations": self.find_cross_correlations(
                temporal_patterns, geographic_patterns, actor_patterns, bias_patterns
            )
        }
        
        return pattern_synthesis
    
    def find_cross_correlations(self, temporal, geographic, actor, bias) -> Dict[str, Any]:
        """Find correlations between different pattern types"""
        
        correlations = {
            "temporal_geographic_correlation": self.correlate_temporal_geographic(temporal, geographic),
            "actor_bias_correlation": self.correlate_actor_bias(actor, bias),
            "violation_bias_correlation": self.correlate_violations_bias(temporal, bias)
        }
        
        return correlations
    
    def correlate_temporal_geographic(self, temporal, geographic) -> Dict[str, Any]:
        """Correlate temporal patterns with geographic patterns"""
        return {
            "seasonal_geographic_shifts": "Analysis placeholder",
            "time_location_hotspots": "Analysis placeholder"
        }
    
    def correlate_actor_bias(self, actor, bias) -> Dict[str, Any]:
        """Correlate actor patterns with bias patterns"""
        return {
            "actor_framing_correlation": "Analysis placeholder",
            "bias_actor_emphasis": "Analysis placeholder"
        }
    
    def correlate_violations_bias(self, temporal, bias) -> Dict[str, Any]:
        """Correlate violation patterns with bias patterns"""
        return {
            "violation_emphasis_correlation": "Analysis placeholder",
            "legal_framing_patterns": "Analysis placeholder"
        }


class TemporalPatternAnalyzer:
    """Analyzes temporal patterns across reports"""
    
    def analyze(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal evolution patterns"""
        
        # Extract temporal data from all reports
        temporal_data = []
        for result in individual_results:
            for item in result.get("quantitative_data", []):
                if "temporal_data" in item:
                    temporal_data.append({
                        "report_name": result["report_name"],
                        "category": item.get("category"),
                        "value": item.get("value"),
                        "actor": item.get("actor"),
                        "temporal_data": item["temporal_data"],
                        "confidence": item.get("confidence_score", 0)
                    })
        
        if not temporal_data:
            return {"error": "No temporal data found"}
        
        # Analyze violation trends over time
        violation_trends = self.analyze_violation_trends(temporal_data)
        
        # Analyze actor behavior changes
        actor_evolution = self.analyze_actor_temporal_evolution(temporal_data)
        
        # Analyze seasonal patterns
        seasonal_patterns = self.analyze_seasonal_patterns(temporal_data)
        
        # Analyze ceasefire impact
        ceasefire_impact = self.analyze_ceasefire_impact(temporal_data)
        
        return {
            "violation_trends": violation_trends,
            "actor_evolution": actor_evolution,
            "seasonal_patterns": seasonal_patterns,
            "ceasefire_impact": ceasefire_impact,
            "temporal_summary": self.generate_temporal_summary(temporal_data)
        }
    
    def analyze_violation_trends(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how violations change over time"""
        
        # Group by category and time period
        category_trends = defaultdict(list)
        
        for item in temporal_data:
            category = item["category"]
            reporting_period = item["temporal_data"].get("reporting_period_start", "unknown")
            value = item["value"]
            
            if reporting_period != "unknown" and isinstance(value, (int, float)):
                category_trends[category].append({
                    "period": reporting_period,
                    "value": value,
                    "actor": item["actor"]
                })
        
        # Calculate trends for each category
        trends = {}
        for category, data_points in category_trends.items():
            if len(data_points) >= 2:
                # Sort by period
                sorted_data = sorted(data_points, key=lambda x: x["period"])
                
                # Calculate trend
                values = [d["value"] for d in sorted_data]
                if len(values) >= 2:
                    trend_direction = "increasing" if values[-1] > values[0] else "decreasing"
                    change_percent = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
                    
                    trends[category] = {
                        "trend_direction": trend_direction,
                        "change_percent": round(change_percent, 2),
                        "data_points": len(values),
                        "latest_value": values[-1],
                        "earliest_value": values[0],
                        "periods_covered": [sorted_data[0]["period"], sorted_data[-1]["period"]]
                    }
        
        return trends
    
    def analyze_actor_temporal_evolution(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how actor behavior/attribution changes over time"""
        
        actor_evolution = defaultdict(lambda: defaultdict(list))
        
        for item in temporal_data:
            actor = item["actor"]
            period = item["temporal_data"].get("reporting_period_start", "unknown")
            category = item["category"]
            value = item["value"]
            
            if period != "unknown":
                actor_evolution[actor][period].append({
                    "category": category,
                    "value": value
                })
        
        # Analyze evolution for each actor
        evolution_analysis = {}
        for actor, period_data in actor_evolution.items():
            if len(period_data) >= 2:
                periods = sorted(period_data.keys())
                
                # Calculate activity trends
                total_incidents = {}
                for period in periods:
                    total_incidents[period] = sum(item["value"] for item in period_data[period] if isinstance(item["value"], (int, float)))
                
                if len(total_incidents) >= 2:
                    values = list(total_incidents.values())
                    trend = "increasing" if values[-1] > values[0] else "decreasing"
                    change = ((values[-1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
                    
                    evolution_analysis[actor] = {
                        "activity_trend": trend,
                        "activity_change_percent": round(change, 2),
                        "periods_active": len(periods),
                        "latest_incidents": total_incidents[periods[-1]],
                        "earliest_incidents": total_incidents[periods[0]]
                    }
        
        return evolution_analysis
    
    def analyze_seasonal_patterns(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze seasonal patterns in violations"""
        
        seasonal_data = defaultdict(list)
        
        for item in temporal_data:
            seasonal_period = item["temporal_data"].get("seasonal_period", "unknown")
            value = item["value"]
            category = item["category"]
            
            if seasonal_period != "unknown" and isinstance(value, (int, float)):
                seasonal_data[seasonal_period].append({
                    "category": category,
                    "value": value,
                    "actor": item["actor"]
                })
        
        # Analyze patterns by season
        seasonal_analysis = {}
        for season, data_points in seasonal_data.items():
            total_incidents = sum(d["value"] for d in data_points)
            categories = Counter(d["category"] for d in data_points)
            actors = Counter(d["actor"] for d in data_points)
            
            seasonal_analysis[season] = {
                "total_incidents": total_incidents,
                "incident_count": len(data_points),
                "top_categories": dict(categories.most_common(3)),
                "top_actors": dict(actors.most_common(3)),
                "average_incident_value": round(total_incidents / len(data_points), 2) if data_points else 0
            }
        
        return seasonal_analysis
    
    def analyze_ceasefire_impact(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Analyze impact of ceasefires on violations"""
        
        pre_ceasefire = []
        post_ceasefire = []
        
        for item in temporal_data:
            ceasefire_period = item["temporal_data"].get("ceasefire_period", "unknown")
            
            if ceasefire_period == "pre_ceasefire":
                pre_ceasefire.append(item)
            elif ceasefire_period == "post_ceasefire":
                post_ceasefire.append(item)
        
        if not pre_ceasefire or not post_ceasefire:
            return {"error": "Insufficient ceasefire period data"}
        
        # Calculate impact metrics
        pre_total = sum(item["value"] for item in pre_ceasefire if isinstance(item["value"], (int, float)))
        post_total = sum(item["value"] for item in post_ceasefire if isinstance(item["value"], (int, float)))
        
        pre_avg = pre_total / len(pre_ceasefire) if pre_ceasefire else 0
        post_avg = post_total / len(post_ceasefire) if post_ceasefire else 0
        
        reduction_percent = ((pre_avg - post_avg) / pre_avg) * 100 if pre_avg != 0 else 0
        
        return {
            "pre_ceasefire_incidents": len(pre_ceasefire),
            "post_ceasefire_incidents": len(post_ceasefire),
            "pre_ceasefire_total": pre_total,
            "post_ceasefire_total": post_total,
            "average_reduction_percent": round(reduction_percent, 2),
            "ceasefire_effectiveness": "effective" if reduction_percent > 50 else "limited" if reduction_percent > 0 else "ineffective"
        }
    
    def generate_temporal_summary(self, temporal_data: List[Dict]) -> Dict[str, Any]:
        """Generate overall temporal analysis summary"""
        
        return {
            "total_temporal_data_points": len(temporal_data),
            "time_periods_covered": len(set(item["temporal_data"].get("reporting_period_start") for item in temporal_data)),
            "categories_with_temporal_data": len(set(item["category"] for item in temporal_data)),
            "actors_with_temporal_data": len(set(item["actor"] for item in temporal_data))
        }


class GeographicPatternAnalyzer:
    """Analyzes geographic patterns across reports"""
    
    def analyze(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Analyze geographic clustering patterns"""
        
        # Extract geographic data
        geographic_data = []
        for result in individual_results:
            for item in result.get("quantitative_data", []):
                if "geographic_data" in item:
                    geographic_data.append({
                        "report_name": result["report_name"],
                        "category": item.get("category"),
                        "value": item.get("value"),
                        "actor": item.get("actor"),
                        "geographic_data": item["geographic_data"],
                        "confidence": item.get("confidence_score", 0)
                    })
        
        if not geographic_data:
            return {"error": "No geographic data found"}
        
        # Analyze location hotspots
        location_hotspots = self.analyze_location_hotspots(geographic_data)
        
        # Analyze sector patterns
        sector_patterns = self.analyze_sector_patterns(geographic_data)
        
        # Analyze geographic bias
        geographic_bias = self.analyze_geographic_bias(geographic_data)
        
        return {
            "location_hotspots": location_hotspots,
            "sector_patterns": sector_patterns,
            "geographic_bias": geographic_bias,
            "geographic_summary": self.generate_geographic_summary(geographic_data)
        }
    
    def analyze_location_hotspots(self, geographic_data: List[Dict]) -> Dict[str, Any]:
        """Identify geographic hotspots"""
        
        location_incidents = defaultdict(list)
        
        for item in geographic_data:
            location = item["geographic_data"].get("location", "unknown")
            if location != "unknown":
                location_incidents[location].append({
                    "category": item["category"],
                    "value": item["value"],
                    "actor": item["actor"]
                })
        
        # Rank locations by incident count and severity
        hotspots = {}
        for location, incidents in location_incidents.items():
            total_value = sum(inc["value"] for inc in incidents if isinstance(inc["value"], (int, float)))
            categories = Counter(inc["category"] for inc in incidents)
            actors = Counter(inc["actor"] for inc in incidents)
            
            hotspots[location] = {
                "incident_count": len(incidents),
                "total_incident_value": total_value,
                "top_categories": dict(categories.most_common(3)),
                "top_actors": dict(actors.most_common(3))
            }
        
        # Sort by incident count
        sorted_hotspots = dict(sorted(hotspots.items(), key=lambda x: x[1]["incident_count"], reverse=True))
        
        return sorted_hotspots
    
    def analyze_sector_patterns(self, geographic_data: List[Dict]) -> Dict[str, Any]:
        """Analyze patterns by sector (East/West)"""
        
        sector_data = defaultdict(list)
        
        for item in geographic_data:
            sector = item["geographic_data"].get("sector", "unknown")
            if sector != "unknown":
                sector_data[sector].append(item)
        
        sector_analysis = {}
        for sector, incidents in sector_data.items():
            total_incidents = len(incidents)
            total_value = sum(inc["value"] for inc in incidents if isinstance(inc["value"], (int, float)))
            categories = Counter(inc["category"] for inc in incidents)
            actors = Counter(inc["actor"] for inc in incidents)
            
            sector_analysis[sector] = {
                "incident_count": total_incidents,
                "total_incident_value": total_value,
                "average_incident_value": round(total_value / total_incidents, 2) if total_incidents > 0 else 0,
                "top_categories": dict(categories.most_common(3)),
                "top_actors": dict(actors.most_common(3))
            }
        
        return sector_analysis
    
    def analyze_geographic_bias(self, geographic_data: List[Dict]) -> Dict[str, Any]:
        """Analyze potential geographic bias in reporting"""
        
        # This would involve more complex analysis of how different locations
        # are treated in the reporting
        return {
            "bias_indicators": "Analysis placeholder - would detect if certain locations get more detailed coverage",
            "coverage_disparity": "Analysis placeholder - would measure reporting detail variations by location"
        }
    
    def generate_geographic_summary(self, geographic_data: List[Dict]) -> Dict[str, Any]:
        """Generate geographic analysis summary"""
        
        return {
            "total_geographic_data_points": len(geographic_data),
            "unique_locations": len(set(item["geographic_data"].get("location") for item in geographic_data if item["geographic_data"].get("location") != "unknown")),
            "sectors_covered": len(set(item["geographic_data"].get("sector") for item in geographic_data if item["geographic_data"].get("sector") != "unknown"))
        }


class ActorEvolutionAnalyzer:
    """Analyzes how actor behavior and attribution evolves across reports"""
    
    def analyze(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Analyze actor evolution patterns"""
        
        # Extract actor data across all reports
        actor_data = defaultdict(lambda: defaultdict(list))
        
        for result in individual_results:
            report_name = result["report_name"]
            
            for item in result.get("quantitative_data", []):
                actor = item.get("actor", "unknown")
                category = item.get("category", "unknown")
                
                actor_data[actor][report_name].append({
                    "category": category,
                    "value": item.get("value", 0),
                    "confidence": item.get("confidence_score", 0),
                    "legal_article": item.get("legal_article_violated", "")
                })
        
        # Analyze evolution for each actor
        evolution_analysis = {}
        for actor, report_data in actor_data.items():
            if len(report_data) >= 2:  # Need at least 2 reports for evolution analysis
                evolution_analysis[actor] = self.analyze_single_actor_evolution(actor, report_data)
        
        return evolution_analysis
    
    def analyze_single_actor_evolution(self, actor: str, report_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze evolution for a single actor"""
        
        reports = sorted(report_data.keys())
        
        # Calculate metrics for each report
        report_metrics = {}
        for report in reports:
            incidents = report_data[report]
            total_incidents = len(incidents)
            total_value = sum(inc["value"] for inc in incidents if isinstance(inc["value"], (int, float)))
            avg_confidence = sum(inc["confidence"] for inc in incidents) / total_incidents if total_incidents > 0 else 0
            categories = Counter(inc["category"] for inc in incidents)
            
            report_metrics[report] = {
                "incident_count": total_incidents,
                "total_value": total_value,
                "average_confidence": avg_confidence,
                "top_categories": dict(categories.most_common(3))
            }
        
        # Calculate evolution trends
        incident_counts = [report_metrics[r]["incident_count"] for r in reports]
        confidence_scores = [report_metrics[r]["average_confidence"] for r in reports]
        
        return {
            "reports_mentioned": len(reports),
            "incident_trend": self.calculate_trend(incident_counts),
            "confidence_trend": self.calculate_trend(confidence_scores),
            "latest_metrics": report_metrics[reports[-1]],
            "earliest_metrics": report_metrics[reports[0]],
            "evolution_summary": f"Actor mentioned in {len(reports)} reports with {self.calculate_trend(incident_counts)} incident trend"
        }
    
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        if values[-1] > values[0]:
            return "increasing"
        elif values[-1] < values[0]:
            return "decreasing"
        else:
            return "stable"


class BiasEvolutionAnalyzer:
    """Analyzes how bias patterns evolve across reports"""
    
    def analyze(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Analyze bias evolution patterns"""
        
        # Extract bias data across all reports
        bias_evolution = []
        
        for result in individual_results:
            report_name = result["report_name"]
            
            for bias_item in result.get("bias_data", []):
                bias_evolution.append({
                    "report_name": report_name,
                    "bias_flag": bias_item.get("bias_flag", ""),
                    "bias_reason": bias_item.get("bias_reason", ""),
                    "core_actors": bias_item.get("core_actors", []),
                    "violation_type": bias_item.get("violation_type", [])
                })
        
        if not bias_evolution:
            return {"error": "No bias data found"}
        
        # Analyze bias type evolution
        bias_type_evolution = self.analyze_bias_type_evolution(bias_evolution)
        
        # Analyze actor bias evolution
        actor_bias_evolution = self.analyze_actor_bias_evolution(bias_evolution)
        
        return {
            "bias_type_evolution": bias_type_evolution,
            "actor_bias_evolution": actor_bias_evolution,
            "bias_summary": self.generate_bias_summary(bias_evolution)
        }
    
    def analyze_bias_type_evolution(self, bias_evolution: List[Dict]) -> Dict[str, Any]:
        """Analyze how different types of bias change over time"""
        
        # Group by report and bias type
        report_bias_types = defaultdict(lambda: Counter())
        
        for item in bias_evolution:
            report = item["report_name"]
            bias_type = item["bias_flag"]
            
            if bias_type:
                report_bias_types[report][bias_type] += 1
        
        # Calculate evolution for each bias type
        bias_types = set()
        for report_data in report_bias_types.values():
            bias_types.update(report_data.keys())
        
        evolution_analysis = {}
        for bias_type in bias_types:
            counts = []
            reports = sorted(report_bias_types.keys())
            
            for report in reports:
                counts.append(report_bias_types[report][bias_type])
            
            if len(counts) >= 2:
                trend = self.calculate_trend(counts)
                evolution_analysis[bias_type] = {
                    "trend": trend,
                    "latest_count": counts[-1],
                    "earliest_count": counts[0],
                    "reports_present": len([c for c in counts if c > 0])
                }
        
        return evolution_analysis
    
    def analyze_actor_bias_evolution(self, bias_evolution: List[Dict]) -> Dict[str, Any]:
        """Analyze how bias toward specific actors evolves"""
        
        actor_bias = defaultdict(lambda: defaultdict(list))
        
        for item in bias_evolution:
            report = item["report_name"]
            bias_type = item["bias_flag"]
            actors = item["core_actors"]
            
            for actor in actors:
                actor_bias[actor][report].append(bias_type)
        
        # Analyze evolution for each actor
        evolution_analysis = {}
        for actor, report_data in actor_bias.items():
            if len(report_data) >= 2:
                reports = sorted(report_data.keys())
                bias_counts = []
                
                for report in reports:
                    bias_counts.append(len(report_data[report]))
                
                evolution_analysis[actor] = {
                    "reports_mentioned": len(reports),
                    "bias_mention_trend": self.calculate_trend(bias_counts),
                    "latest_bias_count": bias_counts[-1],
                    "earliest_bias_count": bias_counts[0]
                }
        
        return evolution_analysis
    
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        if values[-1] > values[0]:
            return "increasing"
        elif values[-1] < values[0]:
            return "decreasing"
        else:
            return "stable"
    
    def generate_bias_summary(self, bias_evolution: List[Dict]) -> Dict[str, Any]:
        """Generate bias evolution summary"""
        
        bias_types = Counter(item["bias_flag"] for item in bias_evolution if item["bias_flag"])
        all_actors = []
        for item in bias_evolution:
            all_actors.extend(item["core_actors"])
        actor_mentions = Counter(all_actors)
        
        return {
            "total_bias_instances": len(bias_evolution),
            "bias_types_found": dict(bias_types),
            "most_mentioned_actors": dict(actor_mentions.most_common(5)),
            "reports_with_bias": len(set(item["report_name"] for item in bias_evolution))
        }


class ViolationTrendAnalyzer:
    """Analyzes trends in legal violations across reports"""
    
    def analyze(self, individual_results: List[Dict]) -> Dict[str, Any]:
        """Analyze violation trend patterns"""
        
        # Extract violation data
        violation_data = []
        
        for result in individual_results:
            report_name = result["report_name"]
            
            for item in result.get("quantitative_data", []):
                legal_article = item.get("legal_article_violated", "")
                if legal_article and "UNSCR 1701" in legal_article:
                    violation_data.append({
                        "report_name": report_name,
                        "legal_article": legal_article,
                        "category": item.get("category", ""),
                        "value": item.get("value", 0),
                        "actor": item.get("actor", "")
                    })
        
        if not violation_data:
            return {"error": "No violation data found"}
        
        # Analyze article violation trends
        article_trends = self.analyze_article_trends(violation_data)
        
        # Analyze actor violation patterns
        actor_violation_patterns = self.analyze_actor_violations(violation_data)
        
        return {
            "article_trends": article_trends,
            "actor_violation_patterns": actor_violation_patterns,
            "violation_summary": self.generate_violation_summary(violation_data)
        }
    
    def analyze_article_trends(self, violation_data: List[Dict]) -> Dict[str, Any]:
        """Analyze trends for each UNSCR 1701 article"""
        
        article_by_report = defaultdict(lambda: Counter())
        
        for item in violation_data:
            report = item["report_name"]
            article = item["legal_article"]
            
            article_by_report[report][article] += 1
        
        # Calculate trends for each article
        all_articles = set()
        for report_data in article_by_report.values():
            all_articles.update(report_data.keys())
        
        article_trends = {}
        for article in all_articles:
            counts = []
            reports = sorted(article_by_report.keys())
            
            for report in reports:
                counts.append(article_by_report[report][article])
            
            if len(counts) >= 2:
                trend = self.calculate_trend(counts)
                article_trends[article] = {
                    "trend": trend,
                    "latest_violations": counts[-1],
                    "earliest_violations": counts[0],
                    "total_violations": sum(counts),
                    "reports_violated": len([c for c in counts if c > 0])
                }
        
        return article_trends
    
    def analyze_actor_violations(self, violation_data: List[Dict]) -> Dict[str, Any]:
        """Analyze violation patterns by actor"""
        
        actor_violations = defaultdict(lambda: defaultdict(int))
        
        for item in violation_data:
            actor = item["actor"]
            article = item["legal_article"]
            
            actor_violations[actor][article] += 1
        
        # Analyze patterns for each actor
        actor_patterns = {}
        for actor, article_counts in actor_violations.items():
            total_violations = sum(article_counts.values())
            most_violated = max(article_counts.items(), key=lambda x: x[1]) if article_counts else ("None", 0)
            
            actor_patterns[actor] = {
                "total_violations": total_violations,
                "articles_violated": len(article_counts),
                "most_violated_article": most_violated[0],
                "most_violated_count": most_violated[1],
                "violation_distribution": dict(article_counts)
            }
        
        return actor_patterns
    
    def calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        if values[-1] > values[0]:
            return "increasing"
        elif values[-1] < values[0]:
            return "decreasing"
        else:
            return "stable"
    
    def generate_violation_summary(self, violation_data: List[Dict]) -> Dict[str, Any]:
        """Generate violation analysis summary"""
        
        articles = Counter(item["legal_article"] for item in violation_data)
        actors = Counter(item["actor"] for item in violation_data)
        
        return {
            "total_violations": len(violation_data),
            "unique_articles_violated": len(articles),
            "unique_violating_actors": len(actors),
            "most_violated_articles": dict(articles.most_common(5)),
            "top_violating_actors": dict(actors.most_common(5))
        }


def main():
    """Test the cross-report analyzer"""
    # This would be used for testing with sample data
    print("Cross-Report Pattern Analyzer - Test Mode")


if __name__ == "__main__":
    main()