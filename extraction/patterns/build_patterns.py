#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_patterns.py
PatternBank compiler that loads YAML lexicon and generates regex patterns at runtime.
Makes Fast Path data-driven and generalizable across UN reports.
"""

from __future__ import annotations

import os
import re
import yaml
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

import regex as re

# ------------------------------------------------------------
# Core data structures
# ------------------------------------------------------------

@dataclass
class PatternMatch:
    """Result of a pattern match with metadata"""
    pid: int
    page: int
    family: str
    subtype: str
    value: Optional[float]
    high: Optional[float]
    approx: bool
    unit: Optional[str]
    qualifier: Optional[str]
    actor_raw: Optional[str]
    actor_norm: Optional[str]
    confidence: float
    quote: str
    span: Dict[str, int]
    span_hash: str

@dataclass
class CompiledPattern:
    """A compiled regex pattern with metadata"""
    family: str
    subtype: str
    pattern: re.Pattern
    confidence_base: float
    unit_override: Optional[str] = None

class PatternBank:
    """
    PatternBank loads YAML lexicon and compiles regex patterns for UN report extraction.
    Provides data-driven, generalizable pattern matching.
    """
    
    def __init__(self, lexicon_path: Optional[str] = None):
        self.lexicon_path = lexicon_path or os.path.join(
            os.path.dirname(__file__), "un_lexicon.yaml"
        )
        self.lexicon: Dict[str, Any] = {}
        self.patterns: List[CompiledPattern] = []
        self.actor_matcher: Optional[ActorMatcher] = None
        self._load_lexicon()
        self._compile_patterns()
    
    def _load_lexicon(self):
        """Load the YAML lexicon file"""
        try:
            with open(self.lexicon_path, 'r', encoding='utf-8') as f:
                self.lexicon = yaml.safe_load(f)
            logging.info(f"[PatternBank] loaded lexicon from {self.lexicon_path}")
        except Exception as e:
            logging.error(f"[PatternBank] failed to load lexicon: {e}")
            raise
    
    def _compile_patterns(self):
        """Compile all patterns from the lexicon"""
        families = self.lexicon.get('families', {})
        
        for family_name, family_config in families.items():
            self._compile_family_patterns(family_name, family_config)
        
        # Initialize actor matcher
        actors = self.lexicon.get('actors', {})
        self.actor_matcher = ActorMatcher(actors)
        
        logging.info(f"[PatternBank] compiled {len(self.patterns)} patterns for {len(families)} families")
    
    def _compile_family_patterns(self, family_name: str, family_config: Dict[str, Any]):
        """Compile patterns for a specific family"""
        observer_verbs = family_config.get('observer_verbs', [])
        actor_verbs = family_config.get('actor_verbs', [])
        # Fallback to old format for backward compatibility
        if not observer_verbs and not actor_verbs:
            verbs = family_config.get('verbs', [])
            observer_verbs = verbs
            actor_verbs = verbs
        nouns = family_config.get('nouns', [])
        extras = family_config.get('extras', [])
        skeletons = family_config.get('skeletons', [])
        subtypes = family_config.get('subtypes', ['default'])
        
        # Build numeric grammar
        num_grammar = self._build_numeric_grammar()
        
        # Compile verb-based patterns for observer verbs
        for verb in observer_verbs:
            for noun in nouns:
                self._compile_verb_noun_pattern(family_name, verb, noun, num_grammar, subtypes, verb_type="observer")
        
        # Compile verb-based patterns for actor verbs
        for verb in actor_verbs:
            for noun in nouns:
                self._compile_verb_noun_pattern(family_name, verb, noun, num_grammar, subtypes, verb_type="actor")
        
        # Compile skeleton patterns
        for skeleton in skeletons:
            self._compile_skeleton_pattern(family_name, skeleton, num_grammar, subtypes)
        
        # Compile extra patterns
        for extra in extras:
            self._compile_extra_pattern(family_name, extra, num_grammar, subtypes)
        
        # Compile special patterns for date_mentions
        if family_name == 'date_mentions':
            self._compile_date_patterns(family_name, family_config, subtypes)
    
    def _build_numeric_grammar(self) -> str:
        """Build the numeric grammar regex pattern"""
        numbers = self.lexicon.get('numbers', {})
        qualifiers = numbers.get('qualifiers', [])
        magnitudes = numbers.get('magnitudes', [])
        ranges = numbers.get('ranges', [])
        
        # Qualifier pattern
        qual_pattern = ""
        if qualifiers:
            qual_pattern = rf"(?P<qualifier>\b(?:{'|'.join(re.escape(q) for q in qualifiers)})\b)?\s*"
        
        # Range pattern
        range_pattern = rf"(?:(?P<r1>\d{{1,4}})\s*(?:{'|'.join(re.escape(r) for r in ranges)})\s*(?P<r2>\d{{1,4}}))"
        
        # Digit pattern
        digit_pattern = r"(?P<num>\d{1,3}(?:[ ,]\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
        
        # Word number pattern
        word_pattern = ""
        if magnitudes:
            word_pattern = rf"|(?P<wordnum>\b(?:{'|'.join(re.escape(m) for m in magnitudes)})\b)"
        
        # Combined numeric pattern
        num_pattern = rf"{qual_pattern}(?:{range_pattern}|{digit_pattern}{word_pattern})"
        
        return num_pattern
    
    def _compile_verb_noun_pattern(self, family: str, verb: str, noun: str, num_grammar: str, subtypes: List[str], verb_type: str = "observer"):
        """Compile a verb-noun pattern with verb type awareness"""
        # Adjust confidence based on verb type
        confidence_base = 0.9 if verb_type == "observer" else 0.8
        
        # Basic pattern: [qualifier] [number] [noun] [verb]
        pattern_str = rf"{num_grammar}\s+(?P<noun>{re.escape(noun)})\s+(?P<verb>{re.escape(verb)})"
        
        try:
            compiled = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
            self.patterns.append(CompiledPattern(
                family=family,
                subtype=subtypes[0] if subtypes else 'default',
                pattern=compiled,
                confidence_base=confidence_base
            ))
        except Exception as e:
            logging.warning(f"[PatternBank] failed to compile pattern for {family}:{verb}:{noun}: {e}")
        
        # Inverse pattern: [verb] [number] [noun]
        pattern_str2 = rf"(?P<verb>{re.escape(verb)})\s+{num_grammar}\s+(?P<noun>{re.escape(noun)})"
        
        try:
            compiled2 = re.compile(pattern_str2, re.IGNORECASE | re.MULTILINE)
            self.patterns.append(CompiledPattern(
                family=family,
                subtype=subtypes[0] if subtypes else 'default',
                pattern=compiled2,
                confidence_base=confidence_base
            ))
        except Exception as e:
            logging.warning(f"[PatternBank] failed to compile inverse pattern for {family}:{verb}:{noun}: {e}")
    
    def _compile_skeleton_pattern(self, family: str, skeleton: str, num_grammar: str, subtypes: List[str]):
        """Compile a skeleton pattern with placeholders and bounded gaps"""
        # Replace placeholders
        pattern_str = skeleton.replace('{NUM}', num_grammar)
        pattern_str = pattern_str.replace('{verb}', r'(?P<verb>\w+(?:\s+\w+)*)')
        pattern_str = pattern_str.replace('{noun}', r'(?P<noun>\w+(?:\s+\w+)*)')
        pattern_str = pattern_str.replace('{actor}', r'(?P<actor>\w+(?:\s+\w+)*)')
        
        # Replace unbounded gaps with bounded gaps
        pattern_str = re.sub(r'\s*\.\*\s*', r'\\s*(?:.{0,80}?)\\s*', pattern_str)
        
        try:
            compiled = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
            self.patterns.append(CompiledPattern(
                family=family,
                subtype=subtypes[0] if subtypes else 'default',
                pattern=compiled,
                confidence_base=0.7
            ))
        except Exception as e:
            logging.warning(f"[PatternBank] failed to compile skeleton pattern for {family}: {e}")
    
    def _compile_extra_pattern(self, family: str, extra: str, num_grammar: str, subtypes: List[str]):
        """Compile an extra pattern"""
        # Add numeric grammar to extra patterns that need it
        if '{NUM}' in extra:
            pattern_str = extra.replace('{NUM}', num_grammar)
        else:
            pattern_str = extra
        
        # Handle regex patterns that are already in regex format vs literal text
        # If the pattern contains regex syntax, treat it as a regex pattern
        # Otherwise, escape it as literal text
        if any(char in pattern_str for char in ['(', ')', '|', '[', ']', '*', '+', '?', '.', '^', '$']):
            # This looks like a regex pattern, compile it directly
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                self.patterns.append(CompiledPattern(
                    family=family,
                    subtype=subtypes[0] if subtypes else 'default',
                    pattern=compiled,
                    confidence_base=0.6
                ))
            except Exception as e:
                logging.warning(f"[PatternBank] failed to compile regex extra pattern for {family}: {e}")
        else:
            # This looks like literal text, escape it
            try:
                escaped_pattern = re.escape(pattern_str)
                compiled = re.compile(escaped_pattern, re.IGNORECASE | re.MULTILINE)
                self.patterns.append(CompiledPattern(
                    family=family,
                    subtype=subtypes[0] if subtypes else 'default',
                    pattern=compiled,
                    confidence_base=0.6
                ))
            except Exception as e:
                logging.warning(f"[PatternBank] failed to compile literal extra pattern for {family}: {e}")
    
    def _compile_date_patterns(self, family: str, family_config: Dict[str, Any], subtypes: List[str]):
        """Compile date mention patterns"""
        patterns = family_config.get('patterns', [])
        
        for pattern_str in patterns:
            try:
                compiled = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
                self.patterns.append(CompiledPattern(
                    family=family,
                    subtype=subtypes[0] if subtypes else 'default',
                    pattern=compiled,
                    confidence_base=0.5
                ))
            except Exception as e:
                logging.warning(f"[PatternBank] failed to compile date pattern: {e}")
    
    def find_candidates(self, text: str, pid: int, page: int = -1) -> List[PatternMatch]:
        """
        Find all pattern matches in text and return as PatternMatch objects.
        """
        candidates = []
        dedup_key: Set[Tuple[int, str, str]] = set()
        
        for pattern in self.patterns:
            for match in pattern.pattern.finditer(text):
                # Extract match data
                match_data = self._extract_match_data(match, pattern, text, pid, page)
                if not match_data:
                    continue
                
                # Deduplicate
                key = (pid, pattern.family, match_data['span_hash'])
                if key in dedup_key:
                    continue
                dedup_key.add(key)
                
                # Create PatternMatch
                candidate = PatternMatch(
                    pid=pid,
                    page=page,
                    family=pattern.family,
                    subtype=pattern.subtype,
                    value=match_data['value'],
                    high=match_data['high'],
                    approx=match_data['approx'],
                    unit=match_data['unit'],
                    qualifier=match_data['qualifier'],
                    actor_raw=match_data['actor_raw'],
                    actor_norm=match_data['actor_norm'],
                    confidence=match_data['confidence'],
                    quote=match_data['quote'],
                    span=match_data['span'],
                    span_hash=match_data['span_hash']
                )
                candidates.append(candidate)
        
        return candidates
    
    def _extract_match_data(self, match: re.Match, pattern: CompiledPattern, text: str, pid: int, page: int) -> Optional[Dict[str, Any]]:
        """Extract structured data from a regex match"""
        import hashlib
        
        # Extract numeric data
        value, high, approx, qualifier = self._extract_number(match)
        
        # Extract actor if present
        actor_raw, actor_norm = None, None
        if self.actor_matcher:
            actor_raw, actor_norm = self.actor_matcher.find_first(text)
        
        # Calculate confidence using enhanced scoring
        confidence = pattern.confidence_base
        
        # +0.5 for verb+object both matched
        if self._has_verb_and_object(text, pattern.family):
            confidence += 0.5
        
        # +0.3 if actor alias matched
        if actor_raw:
            confidence += 0.3
        
        # +0.2 if directionality phrase matched
        if self._has_directionality_phrase(text):
            confidence += 0.2
        
        # Extract quote
        start, end = match.span()
        quote = self._extract_quote(text, start, end)
        
        # Generate span hash
        span_hash = f"sha1:{hashlib.sha1(f'{pid}|{pattern.family}|{start}|{end}|{text[start:end]}'.encode()).hexdigest()}"
        
        # Determine unit based on family
        unit = self._get_unit_for_family(pattern.family)
        
        return {
            'value': value,
            'high': high,
            'approx': approx,
            'qualifier': qualifier,
            'actor_raw': actor_raw,
            'actor_norm': actor_norm,
            'confidence': confidence,
            'quote': quote,
            'span': {'start': start, 'end': end},
            'span_hash': span_hash,
            'unit': unit
        }
    
    def _extract_number(self, match: re.Match) -> Tuple[Optional[float], Optional[float], bool, str]:
        """Extract numeric value from match groups"""
        import math
        
        qual = (match.groupdict().get("qualifier") or "").strip()
        
        # Range?
        r1, r2 = match.groupdict().get("r1"), match.groupdict().get("r2")
        if r1 and r2:
            try:
                low = float(r1.replace(",", ""))
                high = float(r2.replace(",", ""))
                return low, high, True, qual
            except ValueError:
                pass
        
        # Single digit number?
        num = match.groupdict().get("num")
        if num:
            try:
                val = float(num.replace(",", ""))
                return val, None, False, qual
            except ValueError:
                pass
        
        # Word number
        wordnum = match.groupdict().get("wordnum")
        if wordnum:
            val, approx = self._wordnum_to_value(wordnum)
            return val, None, approx, qual
        
        return None, None, False, qual
    
    def _wordnum_to_value(self, word: str) -> Tuple[float, bool]:
        """Map word-numbers to approximate numeric values"""
        w = word.lower()
        approx = True
        if w.startswith("dozen"):
            return 12.0, approx
        if w.startswith("score"):
            return 20.0, approx
        if w.startswith("hundred"):
            return 100.0, approx
        if w.startswith("thousand"):
            return 1000.0, approx
        if w in ("million", "millions"):
            return 1_000_000.0, approx
        if w in ("billion", "billions"):
            return 1_000_000_000.0, approx
        return math.nan, approx
    
    def _extract_quote(self, text: str, start: int, end: int, ctx: int = 120) -> str:
        """Extract quote with context"""
        a = max(0, start - ctx)
        b = min(len(text), end + ctx)
        return text[a:b].strip()
    
    def _has_verb_and_object(self, text: str, family: str) -> bool:
        """Check if text contains both verb and object for the family"""
        family_data = self.lexicon.get("families", {}).get(family, {})
        verbs = family_data.get("verbs", [])
        nouns = family_data.get("nouns", [])
        
        text_lower = text.lower()
        
        # Check for family-specific context to avoid over-extraction
        family_context = {
            "trajectories": ["trajectory", "projectile", "rocket", "missile", "fired", "launched"],
            "air_strikes": ["air strike", "airstrike", "air attack", "bombing", "air raid"],
            "casualties_civilians": ["civilian", "killed", "died", "casualt", "death"],
            "shelling_occasions": ["shelling", "opening fire", "occasion", "bombardment"],
            "airspace_violations": ["airspace", "overflight", "violation", "penetrated"],
            "blue_line_violations": ["blue line", "violation", "incident", "crossing"],
            "weapons_seized": ["weapon", "seized", "confiscated", "found", "discovered"],
            "displacement": ["displaced", "evacuated", "fled", "refugee"]
        }
        
        # Must have family-specific context
        context_words = family_context.get(family, [])
        has_context = any(word in text_lower for word in context_words)
        
        if not has_context:
            return False
        
        has_verb = any(verb.lower() in text_lower for verb in verbs)
        has_noun = any(noun.lower() in text_lower for noun in nouns)
        
        return has_verb and has_noun
    
    def _has_directionality_phrase(self, text: str) -> bool:
        """Check if text contains directionality phrases"""
        directionality_phrases = [
            "from south to north",
            "from north to south", 
            "did not cross the blue line",
            "crossed the blue line",
            "north of the blue line",
            "south of the blue line"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in directionality_phrases)
    
    def _get_unit_for_family(self, family: str) -> str:
        """Get the appropriate unit for a fact family"""
        unit_mapping = {
            "trajectories": "count",
            "air_strikes": "count", 
            "casualties_civilians": "count",
            "shelling_occasions": "count",
            "airspace_violations": "count",
            "blue_line_violations": "count",
            "weapons_seized": "count",
            "displacement": "count",
            "distance": "km",
            "finance": "USD",
            "date_mentions": None
        }
        return unit_mapping.get(family, "count")

# ------------------------------------------------------------
# Actor matcher (reused from fast_path_extractor)
# ------------------------------------------------------------

class ActorMatcher:
    """Matches actor aliases and returns canonical names"""
    
    def __init__(self, aliases: Dict[str, List[str]]):
        self.aliases = aliases or {}
        self._ahocorasick = None
        try:
            import ahocorasick
            A = ahocorasick.Automaton()
            for norm, alist in self.aliases.items():
                for a in alist:
                    A.add_word(a.lower(), (norm, a))
            A.make_automaton()
            self._ahocorasick = A
            self._regex = None
        except Exception:
            # Fallback: union regex
            parts = []
            for norm, alist in self.aliases.items():
                for a in alist:
                    parts.append(re.escape(a))
            if parts:
                self._regex = re.compile(r"(?i)\b(" + "|".join(sorted(parts, key=len, reverse=True)) + r")\b")
            else:
                self._regex = None

    def find_first(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        t = text.lower()
        if self._ahocorasick is not None:
            for _, (norm, raw) in self._ahocorasick.iter(t):
                return norm, raw
            return None, None
        if self._regex is not None:
            m = self._regex.search(text)
            if m:
                raw = m.group(1)
                # map back to canonical
                for norm, alist in self.aliases.items():
                    for a in alist:
                        if a.lower() == raw.lower():
                            return norm, a
        return None, None

# ------------------------------------------------------------
# Factory function
# ------------------------------------------------------------

def create_pattern_bank(lexicon_path: Optional[str] = None) -> PatternBank:
    """Factory function to create a PatternBank instance"""
    return PatternBank(lexicon_path)

if __name__ == "__main__":
    # Test the PatternBank
    bank = create_pattern_bank()
    print(f"Loaded {len(bank.patterns)} patterns")
    
    # Test with sample text
    test_text = "UNIFIL radars detected 220 trajectories of projectiles fired from south to north of the Blue Line"
    candidates = bank.find_candidates(test_text, pid=1)
    print(f"Found {len(candidates)} candidates in test text")
    for c in candidates:
        print(f"  {c.family}.{c.subtype}: {c.value} {c.unit} (confidence: {c.confidence:.2f})")
