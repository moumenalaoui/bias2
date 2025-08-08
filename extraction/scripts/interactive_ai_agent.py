#!/usr/bin/env python3
"""
Interactive AI Agent for UN Report Bias Analysis
Provides conversational interface to ask questions about analysis results
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import openai
from pathlib import Path
from dotenv import load_dotenv

class InteractiveAIAgent:
    """
    Interactive AI agent that can answer questions about UN report bias analysis results
    based on Entman's framing theory and quantitative data.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the interactive AI agent."""
        # Load environment variables
        load_dotenv()
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        # Load analysis data
        self.quantitative_data = []
        self.bias_data = []
        self.ai_analysis = None
        self.data_loaded = False
        
        # Interactive prompt template
        self.interactive_prompt_template = """You are an expert analyst specializing in media bias analysis using Entman's framing theory (1993). You have access to comprehensive analysis results from a UN Security Council report and can answer questions about the findings.

## YOUR EXPERTISE
- **Entman's Framing Theory**: Deep understanding of selection bias, framing bias, and omission bias
- **UN Security Council Resolutions**: Expertise in UNSCR 1701 and legal compliance
- **Quantitative Analysis**: Statistical interpretation of extracted data
- **Bias Detection**: Identification of media bias patterns
- **Legal Assessment**: Evaluation of international law violations

## AVAILABLE DATA
You have access to:

### QUANTITATIVE DATA:
{quantitative_summary}

### BIAS ANALYSIS DATA:
{bias_summary}

### LEGAL VIOLATION SUMMARY:
{legal_summary}

### PREVIOUS AI ANALYSIS:
{ai_analysis_summary}

## RESPONSE GUIDELINES
- **Be conversational but professional**
- **Base answers on the actual data provided**
- **Use specific numbers and quotes when relevant**
- **Apply Entman's framing theory when discussing bias**
- **Provide actionable insights when appropriate**
- **Acknowledge limitations of the data when relevant**
- **Be objective and balanced in your analysis**

## USER QUESTION
{user_question}

Please provide a comprehensive, data-driven answer:"""

    def load_analysis_data(self, quantitative_file: str, bias_file: str, ai_analysis_file: Optional[str] = None):
        """Load all analysis data for the interactive session."""
        try:
            # Load quantitative data
            self.quantitative_data = self.load_quantitative_data(quantitative_file)
            
            # Load bias data
            self.bias_data = self.load_bias_data(bias_file)
            
            # Load AI analysis if available
            if ai_analysis_file and os.path.exists(ai_analysis_file):
                with open(ai_analysis_file, 'r', encoding='utf-8') as f:
                    self.ai_analysis = json.load(f)
            
            self.data_loaded = True
            print(f"✅ Loaded {len(self.quantitative_data)} quantitative data points and {len(self.bias_data)} bias analysis entries")
            
        except Exception as e:
            print(f"❌ Error loading data: {str(e)}")
            self.data_loaded = False

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

    def generate_quantitative_summary(self) -> str:
        """Generate a summary of quantitative data for the prompt."""
        if not self.quantitative_data:
            return "No quantitative data available."
        
        # Group by category
        categories = {}
        actors = {}
        violations = {}
        
        for item in self.quantitative_data:
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
        
        summary = f"Total data points: {len(self.quantitative_data)}\n\n"
        
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

    def generate_bias_summary(self) -> str:
        """Generate a summary of bias analysis data for the prompt."""
        if not self.bias_data:
            return "No bias analysis data available."
        
        # Bias type distribution
        bias_types = {}
        actors_mentioned = set()
        violation_types = set()
        
        for item in self.bias_data:
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
        
        summary = f"Total paragraphs analyzed: {len(self.bias_data)}\n\n"
        
        # Bias type breakdown
        summary += "BIAS TYPE DISTRIBUTION:\n"
        for bias_type, count in bias_types.items():
            summary += f"- {bias_type}: {count} paragraphs\n"
        
        # Actors mentioned
        summary += f"\nACTORS MENTIONED: {', '.join(sorted(actors_mentioned))}\n"
        
        # Sample bias reasons
        summary += "\nSAMPLE BIAS REASONS:\n"
        for i, item in enumerate(self.bias_data[:3]):  # First 3 examples
            summary += f"- Paragraph {item.get('paragraph_id', 'unknown')}: {item.get('bias_reason', 'No reason provided')[:200]}...\n"
        
        return summary

    def generate_legal_summary(self) -> str:
        """Generate a summary of legal violations."""
        # Collect all legal violations
        violations = {}
        
        # From quantitative data
        for item in self.quantitative_data:
            violation = item.get('legal_article_violated', 'unknown')
            if violation not in violations:
                violations[violation] = 0
            violations[violation] += 1
        
        # From bias data
        for item in self.bias_data:
            legal_summary = item.get('legal_grounding_summary', '')
            if legal_summary and legal_summary != "No UNSCR 1701 violations identified":
                violations[legal_summary] = violations.get(legal_summary, 0) + 1
        
        summary = "LEGAL VIOLATION SUMMARY:\n"
        for violation, count in violations.items():
            summary += f"- {violation}: {count} instances\n"
        
        return summary

    def generate_ai_analysis_summary(self) -> str:
        """Generate a summary of the previous AI analysis."""
        if not self.ai_analysis:
            return "No previous AI analysis available."
        
        analysis_text = self.ai_analysis.get('analysis', '')
        if len(analysis_text) > 500:
            analysis_text = analysis_text[:500] + "..."
        
        return f"Previous AI Analysis Summary:\n{analysis_text}"

    def ask_question(self, question: str) -> str:
        """Ask a question and get an AI response based on the loaded data."""
        if not self.data_loaded:
            return "❌ No analysis data loaded. Please load data first."
        
        try:
            # Generate summaries for the prompt
            quantitative_summary = self.generate_quantitative_summary()
            bias_summary = self.generate_bias_summary()
            legal_summary = self.generate_legal_summary()
            ai_analysis_summary = self.generate_ai_analysis_summary()
            
            # Create the interactive prompt
            prompt = self.interactive_prompt_template.format(
                quantitative_summary=quantitative_summary,
                bias_summary=bias_summary,
                legal_summary=legal_summary,
                ai_analysis_summary=ai_analysis_summary,
                user_question=question
            )
            
            # Call GPT-4o for response
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # Lower temperature for more consistent responses
                max_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"❌ Error generating response: {str(e)}"

    def get_suggested_questions(self) -> List[str]:
        """Get a list of suggested questions users can ask."""
        return [
            "What are the main patterns of bias in this report?",
            "Which actor has the most violations according to the data?",
            "How does Entman's framing theory apply to this analysis?",
            "What are the most significant legal violations found?",
            "Can you compare the actions of different actors?",
            "What recommendations would you make based on this analysis?",
            "How reliable is the quantitative data in this report?",
            "What are the limitations of this bias analysis?",
            "Which UNSCR 1701 articles are most frequently violated?",
            "How does the selection bias manifest in this report?"
        ]

def main():
    """Main function for command-line interactive session."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive AI Agent for UN Report Bias Analysis")
    parser.add_argument("--quantitative", type=str, required=True, help="Path to quantitative results JSONL")
    parser.add_argument("--bias", type=str, required=True, help="Path to bias analysis results JSONL")
    parser.add_argument("--ai-analysis", type=str, help="Path to AI analysis report JSON")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = InteractiveAIAgent(api_key=args.api_key)
    
    # Load data
    print("🤖 Loading analysis data...")
    agent.load_analysis_data(args.quantitative, args.bias, args.ai_analysis)
    
    if not agent.data_loaded:
        print("❌ Failed to load data. Exiting.")
        sys.exit(1)
    
    print("✅ Data loaded successfully!")
    print("\n🎯 Interactive AI Agent Ready!")
    print("Ask me anything about the analysis results.")
    print("Type 'quit' to exit, 'help' for suggested questions.\n")
    
    # Interactive loop
    while True:
        try:
            question = input("❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\n💡 Suggested questions:")
                for i, q in enumerate(agent.get_suggested_questions(), 1):
                    print(f"{i}. {q}")
                print()
                continue
            
            if not question:
                continue
            
            print("\n🤖 AI Agent is thinking...")
            response = agent.ask_question(question)
            print(f"\n💬 {response}\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main() 