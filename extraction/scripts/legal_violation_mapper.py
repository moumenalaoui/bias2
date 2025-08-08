#!/usr/bin/env python3
"""
Legal Violation Mapper
Maps violations to specific UN Security Council Resolution articles for legal grounding
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalViolationMapper:
    """
    Maps violations to specific UN Security Council Resolution articles
    for legally grounded citations
    """
    
    def __init__(self, cache_dir: str = "../../resolution_cache"):
        self.cache_dir = Path(cache_dir)
        self.resolution_cache = {}
        self.violation_patterns = {}
        self.load_resolution_cache()
        self.build_violation_patterns()
    
    def load_resolution_cache(self):
        """Load all cached resolution data"""
        if not self.cache_dir.exists():
            logger.warning(f"Resolution cache directory {self.cache_dir} not found")
            return
        
        for cache_file in self.cache_dir.glob("resolution_*.json"):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    resolution_number = data.get("resolution_number")
                    if resolution_number:
                        self.resolution_cache[resolution_number] = data
                        logger.info(f"Loaded resolution {resolution_number} with {len(data.get('articles', {}))} articles")
            except Exception as e:
                logger.error(f"Error loading {cache_file}: {e}")
    
    def build_violation_patterns(self):
        """Build patterns to match violations to UNSCR 1701 articles (all 19 articles)"""
        self.violation_patterns = {
            # UNSCR 1701 as the single yardstick for all violations (19 articles total)
            
            # 1. Cessation of Hostilities (Articles 1, 2, 7, 8, 11, 16)
            "cessation_of_hostilities": {
                "keywords": [
                    "fire", "strikes", "shelling", "rockets", "missiles", "hostilities", 
                    "ceasefire violations", "exchanges of fire", "military operations",
                    "attacks", "fighting", "clashes", "violence", "cessation of hostilities",
                    "offensive military operations", "permanent ceasefire"
                ],
                "articles": ["1", "2", "7", "8", "11", "16"],
                "description": "Violations of ceasefire and cessation of hostilities requirements"
            },
            
            # 2. Sovereignty Violations (Articles 3, 4, 5, 8, 10)
            "sovereignty_violations": {
                "keywords": [
                    "Blue Line crossing", "territorial violations", "incursions", "cross-border",
                    "Lebanese territory", "north of Blue Line", "south of Blue Line",
                    "north of the Blue Line", "south of the Blue Line",
                    "air incursions", "ground incursions", "sea incursions", "sovereignty",
                    "territorial integrity", "borders", "Blue Line", "Israeli forces", "positions",
                    "Lebanese territory", "foreign forces", "Shebaa farms", "international borders"
                ],
                "articles": ["3", "4", "5", "8", "10"],
                "description": "Violations of Lebanese sovereignty and territorial integrity"
            },
            
            # 3. Unauthorized Arms (Articles 8, 14, 15)
            "unauthorized_arms": {
                "keywords": [
                    "unauthorized arms", "weapons cache", "arms smuggling", "weapons in Lebanon",
                    "AOR", "unauthorized weapons", "arms proliferation", "weapons transfer",
                    "unauthorized assets", "weapons uncovered", "arms embargo violations",
                    "arms and related materiel", "military equipment", "paramilitary equipment",
                    "technical training", "arms supply", "weapons sales"
                ],
                "articles": ["8", "14", "15"],
                "description": "Unauthorized arms and weapons in Lebanese territory"
            },
            
            # 4. Restriction of Movement/Access (Articles 11, 12)
            "restriction_of_movement": {
                "keywords": [
                    "restriction of movement", "access denial", "freedom of movement",
                    "humanitarian access", "movement restrictions", "access blocked",
                    "freedom of movement denied", "movement impeded", "hostile activities",
                    "freedom of movement of United Nations personnel"
                ],
                "articles": ["11", "12"],
                "description": "Restrictions on UNIFIL movement and humanitarian access"
            },
            
            # 5. Attacks on UNIFIL (Articles 11, 12, 13)
            "attacks_on_unifil": {
                "keywords": [
                    "attacks on UNIFIL", "UN personnel", "peacekeepers", "force protection",
                    "UNIFIL personnel", "peacekeeper attacks", "UN forces", "UNIFIL forces",
                    "threats to peacekeepers", "UNIFIL security", "United Nations personnel",
                    "UNIFIL facilities", "UNIFIL installations", "UNIFIL equipment"
                ],
                "articles": ["11", "12", "13"],
                "description": "Direct attacks or threats against UNIFIL personnel and facilities"
            },
            
            # 6. Humanitarian Violations (Articles 6, 7, 11, 12)
            "humanitarian_violations": {
                "keywords": [
                    "civilians", "civilian casualties", "humanitarian", "displacement",
                    "refugees", "humanitarian access", "civilian infrastructure",
                    "humanitarian workers", "safe passage", "voluntary return",
                    "humanitarian assistance", "displaced persons", "civilian populations",
                    "humanitarian convoys", "reconstruction", "development"
                ],
                "articles": ["6", "7", "11", "12"],
                "description": "Humanitarian violations and civilian protection"
            },
            
            # 7. UNIFIL Mandate and Operations (Articles 11, 12, 13, 16)
            "unifil_mandate": {
                "keywords": [
                    "UNIFIL mandate", "UNIFIL operations", "UNIFIL deployment", "UNIFIL troops",
                    "UNIFIL force strength", "UNIFIL capabilities", "UNIFIL functions",
                    "UNIFIL assistance", "UNIFIL coordination", "UNIFIL monitoring"
                ],
                "articles": ["11", "12", "13", "16"],
                "description": "UNIFIL mandate, operations, and force deployment"
            },
            
            # 8. Government Control and Authority (Articles 3, 8, 14)
            "government_control": {
                "keywords": [
                    "Government of Lebanon", "Lebanese government", "government authority",
                    "government control", "government consent", "government borders",
                    "government entry points", "government territory", "Lebanese State"
                ],
                "articles": ["3", "8", "14"],
                "description": "Government of Lebanon control and authority violations"
            },
            
            # 9. International Cooperation (Articles 7, 9, 10, 13, 17, 18, 19)
            "international_cooperation": {
                "keywords": [
                    "Security Council", "Secretary-General", "international community",
                    "Member States", "international actors", "cooperation", "compliance",
                    "international peace", "comprehensive peace", "lasting peace"
                ],
                "articles": ["7", "9", "10", "13", "17", "18", "19"],
                "description": "International cooperation and compliance requirements"
            }
        }
    
    def find_violation_articles(self, violation_text: str, violation_type: str = None) -> List[Dict]:
        """
        Find specific UNSCR 1701 articles that are violated based on the violation text
        
        Args:
            violation_text: Text describing the violation
            violation_type: Type of violation (optional)
            
        Returns:
            List of dictionaries with UNSCR 1701 article information
        """
        violations = []
        violation_text_lower = violation_text.lower()
        
        # Only use UNSCR 1701 as the yardstick
        resolution_num = "1701"
        if resolution_num not in self.resolution_cache:
            logger.warning(f"UNSCR {resolution_num} not found in cache")
            return violations
        
        resolution_data = self.resolution_cache[resolution_num]
        articles = resolution_data.get("articles", {})
        
        # If violation type is specified, check only that type
        if violation_type and violation_type in self.violation_patterns:
            patterns_to_check = [violation_type]
        else:
            patterns_to_check = self.violation_patterns.keys()
        
        for pattern_type in patterns_to_check:
            pattern = self.violation_patterns[pattern_type]
            
            # Check if any keywords match
            keyword_matches = []
            for keyword in pattern["keywords"]:
                if keyword.lower() in violation_text_lower:
                    keyword_matches.append(keyword)
            
            if keyword_matches:
                # Check specific articles for this violation type
                relevant_articles = pattern["articles"]
                
                for article_num in relevant_articles:
                    if article_num in articles:
                        article_text = articles[article_num]
                        
                        # Calculate relevance score based on keyword matches
                        relevance_score = self.calculate_relevance_score(
                            violation_text_lower, article_text.lower(), keyword_matches
                        )
                        
                        if relevance_score > 0.3:  # Threshold for relevance
                            violations.append({
                                "resolution_number": resolution_num,
                                "resolution_url": resolution_data.get("url", ""),
                                "article_number": article_num,
                                "article_text": article_text,
                                "violation_type": pattern_type,
                                "matched_keywords": keyword_matches,
                                "relevance_score": relevance_score,
                                "legal_citation": f"UNSCR {resolution_num} (2006), Article {article_num}",
                                "full_citation": f"United Nations Security Council Resolution {resolution_num} (2006), Article {article_num}: {article_text[:200]}..."
                            })
        
        # Sort by relevance score
        violations.sort(key=lambda x: x["relevance_score"], reverse=True)
        return violations
    
    def calculate_relevance_score(self, violation_text: str, article_text: str, keyword_matches: List[str]) -> float:
        """
        Calculate relevance score between violation text and article text
        
        Args:
            violation_text: Lowercase violation text
            article_text: Lowercase article text
            keyword_matches: List of matched keywords
            
        Returns:
            Relevance score between 0 and 1
        """
        score = 0.0
        
        # Base score from keyword matches
        score += len(keyword_matches) * 0.2
        
        # Check for specific terms in article text
        specific_terms = [
            "ceasefire", "hostilities", "withdrawal", "Blue Line", "Lebanese territory",
            "arms", "weapons", "disarmament", "authority", "sovereignty"
        ]
        
        for term in specific_terms:
            if term in violation_text and term in article_text:
                score += 0.1
        
        # Check for actor mentions
        actors = ["israel", "lebanon", "hizbullah", "hezbollah", "idf", "laf", "unifil"]
        for actor in actors:
            if actor in violation_text and actor in article_text:
                score += 0.05
        
        # Normalize score
        return min(score, 1.0)
    
    def get_legal_grounding(self, data_point: Dict) -> Dict:
        """
        Get legal grounding for a data point from quantitative extraction
        
        Args:
            data_point: Dictionary containing extraction data
            
        Returns:
            Dictionary with legal grounding information
        """
        category = data_point.get("category", "")
        quote = data_point.get("quote", "")
        actor = data_point.get("actor", "")
        
        # Map categories to UNSCR 1701 violation types (all 19 articles)
        category_violation_map = {
            # Cessation of Hostilities (Articles 1, 2, 7, 8, 11, 16)
            "air_violations": "cessation_of_hostilities",
            "missile_launches": "cessation_of_hostilities",
            "rocket_launches": "cessation_of_hostilities",
            "shelling_incidents": "cessation_of_hostilities",
            "fire_exchanges": "cessation_of_hostilities",
            "hostilities": "cessation_of_hostilities",
            "ceasefire_violations": "cessation_of_hostilities",
            "military_operations": "cessation_of_hostilities",
            
            # Sovereignty Violations (Articles 3, 4, 5, 8, 10)
            "blue_line_crossings": "sovereignty_violations",
            "territorial_incursions": "sovereignty_violations",
            "air_incursions": "sovereignty_violations",
            "ground_incursions": "sovereignty_violations",
            "sea_incursions": "sovereignty_violations",
            "force_size_idf": "sovereignty_violations",  # Military presence in Lebanese territory
            "tunnels_detected": "sovereignty_violations",
            "force_size_laf": "sovereignty_violations",
            "military_presence": "sovereignty_violations",
            "border_violations": "sovereignty_violations",
            "shebaa_farms": "sovereignty_violations",
            
            # Unauthorized Arms (Articles 8, 14, 15)
            "weapon_caches": "unauthorized_arms",
            "arms_smuggling": "unauthorized_arms",
            "unauthorized_weapons": "unauthorized_arms",
            "weapons_uncovered": "unauthorized_arms",
            "unauthorized_arms": "unauthorized_arms",
            "arms_embargo": "unauthorized_arms",
            "arms_supply": "unauthorized_arms",
            "weapons_sales": "unauthorized_arms",
            
            # Restriction of Movement/Access (Articles 11, 12)
            "movement_restrictions": "restriction_of_movement",
            "access_denial": "restriction_of_movement",
            "humanitarian_access": "restriction_of_movement",
            "freedom_of_movement": "restriction_of_movement",
            "hostile_activities": "restriction_of_movement",
            
            # Attacks on UNIFIL (Articles 11, 12, 13)
            "unifil_attacks": "attacks_on_unifil",
            "peacekeeper_attacks": "attacks_on_unifil",
            "un_personnel_threats": "attacks_on_unifil",
            "unifil_security": "attacks_on_unifil",
            "unifil_facilities": "attacks_on_unifil",
            "unifil_equipment": "attacks_on_unifil",
            
            # Humanitarian Violations (Articles 6, 7, 11, 12)
            "fatalities_total": "humanitarian_violations",
            "fatalities_children": "humanitarian_violations",
            "fatalities_women": "humanitarian_violations",
            "displacement_total": "humanitarian_violations",
            "displacement_returned": "humanitarian_violations",
            "homes_destroyed": "humanitarian_violations",
            "schools_affected": "humanitarian_violations",
            "medical_damage": "humanitarian_violations",
            "civilian_casualties": "humanitarian_violations",
            "humanitarian_assistance": "humanitarian_violations",
            "reconstruction": "humanitarian_violations",
            
            # UNIFIL Mandate and Operations (Articles 11, 12, 13, 16)
            "unifil_deployment": "unifil_mandate",
            "unifil_operations": "unifil_mandate",
            "unifil_troops": "unifil_mandate",
            "unifil_coordination": "unifil_mandate",
            "unifil_monitoring": "unifil_mandate",
            
            # Government Control and Authority (Articles 3, 8, 14)
            "government_authority": "government_control",
            "government_consent": "government_control",
            "government_borders": "government_control",
            "lebanese_state": "government_control",
            
            # International Cooperation (Articles 7, 9, 10, 13, 17, 18, 19)
            "security_council": "international_cooperation",
            "secretary_general": "international_cooperation",
            "international_community": "international_cooperation",
            "member_states": "international_cooperation",
            "compliance": "international_cooperation"
        }
        
        violation_type = category_violation_map.get(category)
        
        # Create violation description
        violation_description = f"{category.replace('_', ' ').title()}: {quote}"
        
        # Find relevant articles
        violations = self.find_violation_articles(violation_description, violation_type)
        
        return {
            "original_data": data_point,
            "violation_type": violation_type,
            "violation_description": violation_description,
            "legal_articles": violations,
            "primary_violation": violations[0] if violations else None,
            "legal_grounding_summary": self.create_legal_summary(violations)
        }
    
    def create_legal_summary(self, violations: List[Dict]) -> str:
        """
        Create a summary of UNSCR 1701 legal grounding
        
        Args:
            violations: List of violation dictionaries
            
        Returns:
            Summary string
        """
        if not violations:
            return "No UNSCR 1701 violations identified"
        
        primary = violations[0]
        summary = f"UNSCR 1701 violation: {primary['legal_citation']} - {primary['violation_type'].replace('_', ' ').title()}"
        
        if len(violations) > 1:
            additional = [v['legal_citation'] for v in violations[1:3]]  # Top 3
            summary += f"\nAdditional UNSCR 1701 violations: {', '.join(additional)}"
        
        return summary
    
    def analyze_bias_with_legal_grounding(self, bias_result: Dict) -> Dict:
        """
        Add legal grounding to bias analysis results
        
        Args:
            bias_result: Dictionary containing bias analysis
            
        Returns:
            Enhanced bias result with legal grounding
        """
        text = bias_result.get("text", "")
        violation_types = bias_result.get("violation_type", [])
        
        legal_violations = []
        for violation_type in violation_types:
            violations = self.find_violation_articles(text, violation_type.lower().replace(" ", "_"))
            legal_violations.extend(violations)
        
        # Remove duplicates and sort by relevance
        unique_violations = {}
        for violation in legal_violations:
            key = f"{violation['resolution_number']}_{violation['article_number']}"
            if key not in unique_violations or violation['relevance_score'] > unique_violations[key]['relevance_score']:
                unique_violations[key] = violation
        
        legal_violations = list(unique_violations.values())
        legal_violations.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Enhance bias result with only essential legal grounding
        enhanced_result = bias_result.copy()
        enhanced_result.update({
            "legal_grounding_summary": self.create_legal_summary(legal_violations)
        })
        
        return enhanced_result
    
    def generate_legal_report(self, data_points: List[Dict], bias_results: List[Dict] = None) -> Dict:
        """
        Generate a comprehensive legal report
        
        Args:
            data_points: List of quantitative data points
            bias_results: List of bias analysis results (optional)
            
        Returns:
            Comprehensive legal report
        """
        legal_groundings = []
        violation_counts = {}
        resolution_counts = {}
        
        # Process quantitative data points
        for data_point in data_points:
            grounding = self.get_legal_grounding(data_point)
            legal_groundings.append(grounding)
            
            if grounding["primary_violation"]:
                violation_type = grounding["primary_violation"]["violation_type"]
                resolution_num = grounding["primary_violation"]["resolution_number"]
                
                violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
                resolution_counts[resolution_num] = resolution_counts.get(resolution_num, 0) + 1
        
        # Process bias results if provided
        bias_legal_groundings = []
        if bias_results:
            for bias_result in bias_results:
                enhanced_bias = self.analyze_bias_with_legal_grounding(bias_result)
                bias_legal_groundings.append(enhanced_bias)
                
                if enhanced_bias["primary_legal_violation"]:
                    violation_type = enhanced_bias["primary_legal_violation"]["violation_type"]
                    resolution_num = enhanced_bias["primary_legal_violation"]["resolution_number"]
                    
                    violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
                    resolution_counts[resolution_num] = resolution_counts.get(resolution_num, 0) + 1
        
        # Generate summary statistics
        total_violations = sum(violation_counts.values())
        
        report = {
            "summary": {
                "total_violations_identified": total_violations,
                "violation_types": violation_counts,
                "resolutions_cited": resolution_counts,
                "most_common_violation": max(violation_counts.items(), key=lambda x: x[1]) if violation_counts else None,
                "most_cited_resolution": max(resolution_counts.items(), key=lambda x: x[1]) if resolution_counts else None
            },
            "quantitative_legal_groundings": legal_groundings,
            "bias_legal_groundings": bias_legal_groundings,
            "detailed_violations": self.generate_detailed_violations(legal_groundings + bias_legal_groundings)
        }
        
        return report
    
    def generate_detailed_violations(self, all_groundings: List[Dict]) -> List[Dict]:
        """
        Generate detailed violation breakdown
        
        Args:
            all_groundings: List of all legal groundings
            
        Returns:
            Detailed violation breakdown
        """
        violation_details = {}
        
        for grounding in all_groundings:
            if "primary_violation" in grounding and grounding["primary_violation"]:
                violation = grounding["primary_violation"]
                key = f"{violation['resolution_number']}_{violation['article_number']}"
                
                if key not in violation_details:
                    violation_details[key] = {
                        "resolution_number": violation["resolution_number"],
                        "article_number": violation["article_number"],
                        "article_text": violation["article_text"],
                        "violation_type": violation["violation_type"],
                        "legal_citation": violation["legal_citation"],
                        "instances": [],
                        "total_instances": 0
                    }
                
                violation_details[key]["instances"].append({
                    "description": grounding.get("violation_description", "Unknown violation"),
                    "relevance_score": violation["relevance_score"],
                    "matched_keywords": violation["matched_keywords"]
                })
                violation_details[key]["total_instances"] += 1
        
        # Convert to list and sort by total instances
        detailed_list = list(violation_details.values())
        detailed_list.sort(key=lambda x: x["total_instances"], reverse=True)
        
        return detailed_list

def main():
    """Test the legal violation mapper"""
    mapper = LegalViolationMapper()
    
    # Test with sample data
    test_violation = "Israeli forces remained present in five positions north of the Blue Line"
    violations = mapper.find_violation_articles(test_violation, "military_presence")
    
    print("Legal Violation Mapper Test")
    print("=" * 50)
    print(f"Test violation: {test_violation}")
    print(f"Found {len(violations)} legal violations:")
    
    for i, violation in enumerate(violations[:3], 1):
        print(f"\n{i}. {violation['legal_citation']}")
        print(f"   Type: {violation['violation_type']}")
        print(f"   Relevance: {violation['relevance_score']:.2f}")
        print(f"   Keywords: {', '.join(violation['matched_keywords'])}")
        print(f"   Article: {violation['article_text'][:100]}...")

if __name__ == "__main__":
    main() 