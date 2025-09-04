#!/usr/bin/env python3
"""
UN Report Analysis - Fast Path
Ultra-fast quantitative extraction with ≤3 LLM calls + optional bias analysis
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import subprocess
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="UN Report Analysis - Fast Path",
    page_icon="",
    layout="wide"
)

# API Key Configuration
st.sidebar.markdown("**OpenAI API Configuration**")
st.sidebar.markdown("**Required for LLM-powered analysis**")

# Check if API key is already set in environment
existing_key = os.getenv('OPENAI_API_KEY', '')
if existing_key:
    st.sidebar.success("✅ API key found in environment")
    api_key = existing_key
else:
    # Get API key from user
    api_key = st.sidebar.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    
    if not api_key:
        st.sidebar.warning("⚠️ API key required to use LLM features")
        st.sidebar.markdown("""
        **To get started:**
        1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
        2. Create an API key
        3. Enter it above
        4. Your key is stored locally and never shared
        """)
        st.stop()
    else:
        # Set the API key in environment for this session
        os.environ['OPENAI_API_KEY'] = api_key
        st.sidebar.success("✅ API key configured for this session")
    
    # Cost estimation
    with st.sidebar.expander("Cost Estimation", expanded=False):
        st.markdown("""
        **Typical costs per analysis:**
        - **Fast Path (≤3 calls)**: ~$0.01-0.03
        - **With Bias Analysis**: ~$0.05-0.15
        - **With AI Agent**: ~$0.01-0.05 per question
        
        *Costs are estimates and may vary based on document size and complexity.*
        """)

# Minimalistic CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 300;
        color: #374151;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.025em;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .stButton > button {
        background: #f3f4f6;
        color: #374151;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 400;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        background: #e5e7eb;
        border-color: #9ca3af;
    }
    
    .stFileUploader > div > div {
        border: 2px dashed #d1d5db;
        background: #f9fafb;
        border-radius: 8px;
    }
    
    .section-header {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #1e3a8a !important;
        margin: 2rem 0 1rem 0 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #e5e7eb !important;
    }
</style>
""", unsafe_allow_html=True)

def run_fast_path_extraction(pdf_file, output_dir, include_quantitative=True, include_bias_analysis=False, process_all_paragraphs=True, min_calls=3, max_calls=3):
    """Run the Fast Path extraction pipeline with universal output structure and configurable LLM calls"""
    try:
        # Save PDF to extraction directory
        current_dir = Path.cwd()
        project_root = current_dir.parent if current_dir.name == "UI" else current_dir
        
        pdf_dest = project_root / "extraction" / "reports_pdf" / pdf_file.name
        pdf_dest.parent.mkdir(parents=True, exist_ok=True)
        
        with open(pdf_dest, "wb") as f:
            f.write(pdf_file.getbuffer())
        
        # Create universal output directory structure
        universal_output_dir = project_root / "extraction" / "universal_output"
        pdf_conversion_dir = universal_output_dir / "pdf_conversion"
        quantitative_dir = universal_output_dir / "quantitative_extraction"
        bias_analysis_dir = universal_output_dir / "bias_analysis"
        
        # Clear and recreate universal output directories
        if universal_output_dir.exists():
            shutil.rmtree(universal_output_dir)
        universal_output_dir.mkdir(parents=True, exist_ok=True)
        pdf_conversion_dir.mkdir(parents=True, exist_ok=True)
        quantitative_dir.mkdir(parents=True, exist_ok=True)
        bias_analysis_dir.mkdir(parents=True, exist_ok=True)
        
        st.info(f"Created fresh universal output directory: {universal_output_dir}")
        
        # Run hybrid extractor to convert PDF to JSONL
        scripts_dir = project_root / "extraction" / "scripts"
        
        # Convert PDF to JSONL and save to universal output
        hybrid_result = subprocess.run([
            sys.executable, "hybrid_extractor.py", str(pdf_dest), str(pdf_conversion_dir)
        ], capture_output=True, text=True, cwd=scripts_dir, timeout=300)
        
        if hybrid_result.returncode != 0:
            return False, f"PDF conversion failed: {hybrid_result.stderr}"
        
        st.info(f"PDF converted to JSONL successfully")
        
        # Find the generated JSONL file in the universal output
        jsonl_files = list(pdf_conversion_dir.glob("*.jsonl"))
        if not jsonl_files:
            st.error(f"No JSONL file found in {pdf_conversion_dir}")
            return False, "No JSONL file generated"
        
        jsonl_file = jsonl_files[0]
        st.info(f"Found JSONL file: {jsonl_file.name}")
        
        # Run quantitative extraction if requested
        if include_quantitative:
            fast_result = subprocess.run([
                sys.executable, "ultra_fast_quantitative_extractor.py",
                "--input", str(jsonl_file), 
                "--output_dir", str(quantitative_dir), 
                "--max_windows", str(max_calls)
            ], capture_output=True, text=True, cwd=scripts_dir, timeout=300)
            
            if fast_result.returncode != 0:
                return False, f"Fast extraction failed: {fast_result.stderr}"
            
            st.info(f"Quantitative extraction completed with {min_calls}-{max_calls} LLM calls")
            
            # Check quantitative results
            results_file = quantitative_dir / "ultra_fast_extraction_results.jsonl"
            summary_file = quantitative_dir / "ultra_fast_extraction_summary.json"
            
            if not (results_file.exists() and summary_file.exists()):
                return False, "Quantitative results files not found"
            
            # Load summary data
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            # Skip quantitative extraction
            st.info("Skipping quantitative extraction as requested")
            summary = {"total_paragraphs_processed": 0, "total_llm_calls_used": 0}
        
        # If bias analysis is requested, run it now
        if include_bias_analysis:
            st.info(f"Running bias analysis on {jsonl_file.name}...")
            
            bias_success, bias_output = run_bias_analysis(jsonl_file, bias_analysis_dir, process_all_paragraphs, max_calls)
            if bias_success:
                st.success(f"Bias analysis completed and saved to universal output")
                # Add bias analysis info to summary
                summary['bias_analysis_completed'] = True
                summary['bias_analysis_path'] = str(bias_output)
            else:
                st.warning(f"Bias analysis failed: {bias_output}")
                summary['bias_analysis_completed'] = False
        
        # Add universal output paths and LLM call info to summary
        summary['universal_output_dir'] = str(universal_output_dir)
        summary['pdf_conversion_dir'] = str(pdf_conversion_dir)
        summary['quantitative_dir'] = str(quantitative_dir)
        summary['bias_analysis_dir'] = str(bias_analysis_dir)
        
        # Only add quantitative file path if quantitative extraction was enabled
        if include_quantitative:
            summary['quantitative_file'] = str(results_file)
        else:
            summary['quantitative_file'] = None
            
        summary['llm_calls_config'] = {'min_calls': min_calls, 'max_calls': max_calls, 'actual_calls': summary.get('llm_calls_used', 0)}
        
        return True, summary
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def run_bias_analysis(jsonl_file_path, output_dir, process_all_paragraphs=True, max_calls=None):
    """Run bias analysis on the extracted JSONL data"""
    try:
        st.info("Starting bias analysis...")
        
        # Get the project root
        current_dir = Path.cwd()
        project_root = current_dir.parent if current_dir.name == "UI" else current_dir
        scripts_dir = project_root / "extraction" / "scripts"
        
        # Run bias analysis based on user preferences
        if process_all_paragraphs:
            # Process all paragraphs - no LLM call limit
            st.info("Bias analysis will process ALL paragraphs in the report (comprehensive analysis)")
            
            bias_result = subprocess.run([
                sys.executable, "call_api_bias_optimized.py", 
                "--input", str(jsonl_file_path),
                "--output", str(output_dir),
                "--max-concurrent", "30",
                "--batch-size", "50"
            ], capture_output=True, text=True, cwd=scripts_dir, timeout=1800)  # Increased timeout for full processing
        else:
            # Process limited number of paragraphs based on LLM call limit
            max_paragraphs_for_bias = max_calls if max_calls else 10  # Default to 10 if no limit specified
            
            st.info(f"Bias analysis will process up to {max_paragraphs_for_bias} paragraphs using {max_calls} LLM calls (limited processing)")
            
            bias_result = subprocess.run([
                sys.executable, "call_api_bias_optimized.py", 
                "--input", str(jsonl_file_path),
                "--output", str(output_dir),
                "--max-paragraphs", str(max_paragraphs_for_bias),
                "--max-concurrent", "30",
                "--batch-size", "50",
                "--max-llm-calls", str(max_calls)
            ], capture_output=True, text=True, cwd=scripts_dir, timeout=900)
        
        if bias_result.returncode != 0:
            error_msg = f"Bias analysis failed: {bias_result.stderr}"
            if "timeout" in error_msg.lower():
                error_msg += "\n\nTip: Try reducing the number of LLM calls or increasing the timeout."
            return False, error_msg
        
        # Check for bias analysis results
        bias_output_file = output_dir / "text_bias_analysis_results.jsonl"
        bias_summary_file = output_dir / "text_bias_analysis_results_summary.json"
        
        if bias_output_file.exists():
            st.success("Bias analysis completed successfully!")
            
            # Try to load and display summary if available
            if bias_summary_file.exists():
                try:
                    with open(bias_summary_file, 'r') as f:
                        summary = json.load(f)
                    st.info(f"Processed {summary.get('total_paragraphs_processed', 0)} paragraphs using {summary.get('total_llm_calls_used', 0)} LLM calls")
                except Exception as e:
                    st.warning(f"Could not load bias analysis summary: {e}")
            
            return True, str(bias_output_file)
        else:
            return False, "Bias analysis completed but no results file found"
            
    except Exception as e:
        return False, f"Bias analysis error: {str(e)}"

def run_ai_analysis(quantitative_file, bias_file, output_dir):
    """Run AI analysis agent on the extracted data"""
    try:
        st.info("Starting AI analysis agent...")
        
        # Get the project root
        current_dir = Path.cwd()
        project_root = current_dir.parent if current_dir.name == "UI" else current_dir
        scripts_dir = project_root / "extraction" / "scripts"
        
        # Run AI analysis agent
        ai_result = subprocess.run([
            sys.executable, "ai_analysis_agent.py", 
            "--quantitative", str(quantitative_file),
            "--bias", str(bias_file),
            "--output", str(output_dir / "ai_analysis_report.json")
        ], capture_output=True, text=True, cwd=scripts_dir, timeout=900)
        
        if ai_result.returncode != 0:
            return False, f"AI analysis failed: {ai_result.stderr}"
        
        # Check for AI analysis results
        ai_output_file = output_dir / "ai_analysis_report.json"
        if ai_output_file.exists():
            st.success("AI analysis completed successfully!")
            return True, str(ai_output_file)
        else:
            return False, "AI analysis completed but no results file found"
            
    except Exception as e:
        return False, f"AI analysis error: {str(e)}"

def normalize_extracted_data(input_file: str, output_file: str) -> Dict[str, Any]:
    """Apply comprehensive data normalization to extracted facts"""
    try:
        # Get the project root and add scripts directory to path
        current_dir = Path.cwd()
        project_root = current_dir.parent if current_dir.name == "UI" else current_dir
        scripts_dir = project_root / "extraction" / "scripts"
        sys.path.insert(0, str(scripts_dir))
        
        # Import the fact normalizer directly
        from fact_normalizer import normalize_facts_from_file
        
        # Process the file
        stats = normalize_facts_from_file(input_file, output_file)
        
        return {
            'success': True,
            'stats': stats,
            'message': f"Data normalization complete: {stats['original_facts']} → {stats['normalized_facts']} facts"
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'message': f"Normalization failed: {str(e)}"
        }

def display_ai_analysis_results(ai_file_path):
    """Display AI analysis results"""
    try:
        st.markdown("**AI Analysis Report**")
        
        # Load AI analysis data
        with open(ai_file_path, 'r', encoding='utf-8') as f:
            ai_data = json.load(f)
        
        if "error" in ai_data:
            st.error(f"AI Analysis Error: {ai_data['error']}")
            return
        
        # Display data summary
        if 'data_summary' in ai_data:
            st.markdown("**Summary**")
            summary = ai_data['data_summary']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", summary.get('quantitative_points', 0))
            with col2:
                st.metric("Bias Paragraphs", summary.get('bias_paragraphs', 0))
            with col3:
                st.metric("Categories", summary.get('categories_found', 0))
            with col4:
                st.metric("Actors", summary.get('actors_identified', 0))
        
        # Display the main analysis
        st.markdown("**Analysis**")
        
        # Parse the analysis text into sections
        analysis_text = ai_data.get('analysis', '')
        if analysis_text:
            # Split by common section headers
            sections = analysis_text.split('\n\n')
            
            for section in sections:
                if section.strip():
                    # Check if it's a section header
                    if section.startswith('###') or section.startswith('##') or section.startswith('#'):
                        st.markdown(section)
                    elif section.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.')):
                        # Numbered sections
                        st.markdown(f"**{section.split('.')[0]}.** {section.split('.', 1)[1] if '.' in section else section}")
                    else:
                        # Regular paragraphs
                        st.write(section)
                        st.markdown("---")
        
        # Download AI analysis results
        with open(ai_file_path, 'r', encoding='utf-8') as f:
            ai_content = f.read()
        
        st.download_button(
            label="Download AI Analysis (JSON)",
            data=ai_content,
            file_name="ai_analysis_report.json",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error displaying AI analysis: {e}")
        st.info(f"Debug: File path: {ai_file_path}")
        st.info(f"Debug: Error details: {str(e)}")

def display_bias_results(bias_file_path):
    """Display comprehensive bias analysis results with 3 main tabs"""
    try:
        st.markdown("**🔍 Bias Analysis Results**")
        
        # Load bias analysis data
        bias_data = []
        with open(bias_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bias_data.append(json.loads(line))
        
        if not bias_data:
            st.warning("No bias analysis data found")
            return
        
        # Load summary if available
        summary_file = Path(bias_file_path).parent / "text_bias_analysis_results_summary.json"
        summary = {}
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Total Paragraphs", summary.get('total_paragraphs_processed', len(bias_data)))
        with col2:
            st.metric("🤖 LLM Calls Used", summary.get('total_llm_calls_used', 'N/A'))
        with col3:
            st.metric("⏱️ Processing Time", f"{summary.get('processing_time_minutes', 0):.2f} min")
        with col4:
            avg_time = summary.get('average_time_per_paragraph', 0)
            st.metric("⚡ Avg Time/Paragraph", f"{avg_time:.2f}s")
        
        st.success(f"✅ Successfully analyzed {len(bias_data)} paragraphs")
        
        # Create the 3 main tabs as requested
        tab1, tab2, tab3 = st.tabs(["📊 Dataset Overview", "📋 Detailed Results", "📈 Bias Analysis Summary"])
        
        with tab1:
            st.markdown("**📊 Dataset Overview - CSV Table**")
            
            # Convert bias data to DataFrame for CSV display
            try:
                import pandas as pd
                
                # Create a simplified DataFrame with key columns
                csv_data = []
                for entry in bias_data:
                    csv_row = {
                        'paragraph_id': entry.get('paragraph_id', ''),
                        'bias_flag': entry.get('bias_flag', ''),
                        'bias_reason': entry.get('bias_reason', '')[:200] + '...' if len(entry.get('bias_reason', '')) > 200 else entry.get('bias_reason', ''),
                        'core_actors': ', '.join(entry.get('core_actors', [])),
                        'violation_type': ', '.join(entry.get('violation_type', [])),
                        'location': entry.get('location', ''),
                        'occurrence_date': entry.get('occurrence_date', ''),
                        'text_preview': entry.get('text', '')[:100] + '...' if len(entry.get('text', '')) > 100 else entry.get('text', '')
                    }
                    csv_data.append(csv_row)
                
                df_csv = pd.DataFrame(csv_data)
                
                # Display the CSV table
                st.markdown("**Bias Analysis Results Table**")
                st.dataframe(df_csv, use_container_width=True, height=400)
                
                # Download CSV button
                csv_string = df_csv.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_string,
                    file_name="bias_analysis_results.csv",
                    mime="text/csv"
                )
                
                # Quick statistics
                st.markdown("**Quick Statistics**")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Rows", len(df_csv))
                with col2:
                    bias_types = df_csv['bias_flag'].nunique()
                    st.metric("Bias Types", bias_types)
                with col3:
                    total_actors = len(set([actor for actors in df_csv['core_actors'] if actors for actor in actors.split(', ')]))
                    st.metric("Unique Actors", total_actors)
                with col4:
                    bias_cases = len(df_csv[df_csv['bias_flag'] != 'none'])
                    st.metric("Bias Cases", bias_cases)
                
            except Exception as e:
                st.error(f"Error creating CSV table: {e}")
                # Fallback to simple display
                st.markdown("**Raw Data Preview**")
                for i, entry in enumerate(bias_data[:5]):
                    st.write(f"**Entry {i+1}:** {entry.get('paragraph_id', 'Unknown')} - {entry.get('bias_flag', 'Unknown')}")
        
        with tab2:
            st.markdown("**📋 Detailed Results - All Text Captured**")
            
            # Option to show subset or all results
            show_all = st.checkbox("Show all paragraphs (may be slow for large datasets)", value=False)
            
            if show_all:
                paragraphs_to_show = bias_data
                st.info(f"Showing all {len(paragraphs_to_show)} paragraphs")
            else:
                paragraphs_to_show = bias_data[:10]
                st.info(f"Showing first 10 paragraphs (check 'Show all' to see all {len(bias_data)} paragraphs)")
            
            for i, entry in enumerate(paragraphs_to_show):
                with st.expander(f"📄 Paragraph {entry.get('paragraph_id', 'Unknown')} - {entry.get('bias_flag', 'Unknown').title()} Bias"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📝 Full Text Content**")
                        text_content = entry.get('text', 'No text available')
                        # Show full text without truncation
                        st.text_area("", text_content, height=150, key=f"full_text_{i}")
                        
                        st.markdown("**🏷️ Bias Classification**")
                        bias_flag = entry.get('bias_flag', 'Unknown')
                        if bias_flag == 'none':
                            st.success("✅ No bias detected")
                        elif bias_flag == 'framing':
                            st.warning("⚠️ Framing bias detected")
                        elif bias_flag == 'selection':
                            st.warning("⚠️ Selection bias detected")
                        elif bias_flag == 'omission':
                            st.warning("⚠️ Omission bias detected")
                        else:
                            st.info(f"ℹ️ {bias_flag}")
                        
                        st.markdown("**👥 Core Actors**")
                        actors = entry.get('core_actors', [])
                        if actors:
                            for actor in actors:
                                st.write(f"• {actor}")
                        else:
                            st.write("No actors identified")
                    
                    with col2:
                        st.markdown("**🔍 Full Bias Analysis**")
                        bias_reason = entry.get('bias_reason', 'No analysis available')
                        # Show full analysis without truncation
                        st.text_area("", bias_reason, height=150, key=f"full_reason_{i}")
                        
                        st.markdown("**⚖️ Violation Types**")
                        violations = entry.get('violation_type', [])
                        if violations:
                            for violation in violations:
                                st.write(f"• {violation}")
                        else:
                            st.write("No violations identified")
                        
                        st.markdown("**📍 Location & Date**")
                        location = entry.get('location', 'Not specified')
                        date = entry.get('occurrence_date', 'Not specified')
                        st.info(f"📍 {location}")
                        st.info(f"📅 {date}")
                        
                        # Show Entman framing analysis if available
                        entman_analysis = entry.get('entman_framing_analysis', {})
                        if entman_analysis:
                            st.markdown("**🎯 Entman Framing Analysis**")
                            for key, value in entman_analysis.items():
                                if value:
                                    st.write(f"• **{key.replace('_', ' ').title()}**: {value}")
        
        with tab3:
            st.markdown("**📈 Bias Analysis Summary - Graphs & Analytics**")
            
            # Bias type distribution chart
            st.markdown("**Bias Type Distribution**")
            bias_counts = {}
            for entry in bias_data:
                bias_type = entry.get('bias_flag', 'unknown')
                bias_counts[bias_type] = bias_counts.get(bias_type, 0) + 1
            
            if bias_counts:
                try:
                    import plotly.express as px
                    df_bias = pd.DataFrame(list(bias_counts.items()), columns=['Bias Type', 'Count'])
                    
                    # Create bar chart
                    fig = px.bar(
                        df_bias,
                        x='Bias Type',
                        y='Count',
                        title="Bias Categories Distribution",
                        labels={'x': 'Bias Type', 'y': 'Count'},
                        color='Count',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(
                        xaxis_tickangle=-45,
                        showlegend=False,
                        height=400
                    )
                    fig.update_traces(
                        text=df_bias['Count'],
                        textposition='outside'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error creating chart: {e}")
            
            # Actor analysis
            st.markdown("**Actor Analysis**")
            actor_counts = {}
            for entry in bias_data:
                actors = entry.get('core_actors', [])
                for actor in actors:
                    actor_counts[actor] = actor_counts.get(actor, 0) + 1
            
            if actor_counts:
                # Show top 10 actors
                top_actors = sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                df_actors = pd.DataFrame(top_actors, columns=['Actor', 'Mentions'])
                
                try:
                    fig = px.bar(
                        df_actors,
                        x='Mentions',
                        y='Actor',
                        orientation='h',
                        title="Top 10 Most Mentioned Actors",
                        color='Mentions',
                        color_continuous_scale='blues'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating actor chart: {e}")
                
                # Actor table
                st.markdown("**Actor Mentions Table**")
                st.dataframe(df_actors, use_container_width=True)
            
            # Violation type analysis
            st.markdown("**Violation Type Analysis**")
            violation_counts = {}
            for entry in bias_data:
                violations = entry.get('violation_type', [])
                for violation in violations:
                    violation_counts[violation] = violation_counts.get(violation, 0) + 1
            
            if violation_counts:
                top_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                df_violations = pd.DataFrame(top_violations, columns=['Violation Type', 'Count'])
                
                try:
                    fig = px.pie(
                        df_violations,
                        values='Count',
                        names='Violation Type',
                        title="Top 10 Violation Types",
                        color_discrete_sequence=px.colors.qualitative.Set2
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error creating violation chart: {e}")
                
                # Violation table
                st.markdown("**Violation Types Table**")
                st.dataframe(df_violations, use_container_width=True)
            
            # Location analysis
            st.markdown("**Location Analysis**")
            location_counts = {}
            for entry in bias_data:
                location = entry.get('location', 'Not specified')
                if location and location != 'Not specified':
                    location_counts[location] = location_counts.get(location, 0) + 1
            
            if location_counts:
                top_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                df_locations = pd.DataFrame(top_locations, columns=['Location', 'Count'])
                
                st.markdown("**Top Locations**")
                st.dataframe(df_locations, use_container_width=True)
            
            # Summary statistics
            st.markdown("**📊 Summary Statistics**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Bias Cases", sum(1 for entry in bias_data if entry.get('bias_flag') != 'none'))
            with col2:
                st.metric("Framing Bias", sum(1 for entry in bias_data if entry.get('bias_flag') == 'framing'))
            with col3:
                st.metric("Selection Bias", sum(1 for entry in bias_data if entry.get('bias_flag') == 'selection'))
            with col4:
                st.metric("Omission Bias", sum(1 for entry in bias_data if entry.get('bias_flag') == 'omission'))
        
    except Exception as e:
        st.error(f"Error displaying bias results: {e}")
        st.info(f"Debug: File path: {bias_file_path}")
        st.info(f"Debug: Error details: {str(e)}")
        
        # Download bias results
        try:
            with open(bias_file_path, 'r', encoding='utf-8') as f:
                bias_content = f.read()
            
            st.download_button(
                label="Download Bias Analysis (JSONL)",
                data=bias_content,
                file_name="bias_analysis_results.jsonl",
                mime="application/json"
            )
        except Exception as download_error:
            st.warning(f"Could not prepare download: {download_error}")

def display_results(data, output_dir):
    """Display the extraction results with rich visualizations"""
    st.markdown(f"**Extraction completed: {data.get('total_data_points', 0)} data points found**")
    
    # Summary Dashboard
    st.markdown("**Summary**")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", data.get('total_data_points', 0))
    with col2:
        st.metric("Categories", data.get('categories_found', 0))
    with col3:
        st.metric("LLM Calls", data.get('llm_calls_used', 0))
    with col4:
        confidence = data.get('average_confidence', 0.0)
        st.metric("Confidence", f"{confidence:.1%}")
    
    # Summary statistics
    st.markdown("**Details**")
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.text(f"Method: {data.get('extraction_method', 'Unknown')}")
        st.text(f"Timestamp: {data.get('extraction_timestamp', 'Unknown')}")
        st.text(f"Categories: {data.get('categories_found', 0)}")
    
    with summary_col2:
        st.text(f"Total Points: {data.get('total_data_points', 0)}")
        st.text(f"Confidence: {data.get('average_confidence', 0.0):.1%}")
        st.text(f"LLM Calls: {data.get('llm_calls_used', 0)}")
    
    # Enhanced visualizations section
    st.markdown("**Data Visualizations**")
    
    # Category distribution
    if 'category_distribution' in data and data['category_distribution']:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Points by Category**")
            category_data = data['category_distribution']
            
            # Convert to DataFrame for better display
            df_cat = pd.DataFrame(list(category_data.items()), columns=['Category', 'Count'])
            df_cat = df_cat.sort_values('Count', ascending=False)
            
            st.dataframe(df_cat, use_container_width=True)
        
        with col2:
            # Create a pie chart for category distribution
            try:
                import plotly.express as px
                fig_pie = px.pie(
                    df_cat, 
                    values='Count', 
                    names='Category',
                    title="Category Distribution (Pie Chart)"
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True, key=f"category_pie_chart_{id(fig_pie)}")
            except ImportError:
                st.warning("Plotly not available")
    
    # Performance metrics
    st.markdown("**Performance**")
    perf_col1, perf_col2 = st.columns(2)
    
    with perf_col1:
        llm_calls = data.get('llm_calls_used', 0)
        max_llm_calls = data.get('llm_calls_config', {}).get('max_calls', 3)
        st.metric("LLM Calls", f"{llm_calls}/{max_llm_calls}")
        # Ensure progress value is between 0.0 and 1.0
        progress_value = min(llm_calls / max_llm_calls, 1.0) if max_llm_calls > 0 else 0.0
        st.progress(progress_value)
    
    with perf_col2:
        total_points = data.get('total_data_points', 0)
        if total_points > 0 and llm_calls > 0:
            points_per_call = total_points / llm_calls
            st.metric("Rate", f"{points_per_call:.1f} points/call")
        else:
            st.metric("Rate", "N/A")
    
    # Enhanced category chart
    if 'category_distribution' in data and data['category_distribution']:
        st.markdown("**Detailed Category Analysis**")
        try:
            import plotly.express as px
            
            # Create enhanced bar chart
            fig_bar = px.bar(
                df_cat,
                x='Category',
                y='Count',
                title="Data Points by Category (Enhanced View)",
                labels={'x': 'Category', 'y': 'Number of Data Points'},
                color='Count',  # Color by count
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(
                xaxis_tickangle=-45,
                showlegend=False,
                height=500
            )
            fig_bar.update_traces(
                text=df_cat['Count'],
                textposition='outside'
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"category_bar_chart_{id(fig_bar)}")
            
        except ImportError:
            st.warning("Plotly not available - showing basic chart")
            # Fallback to basic chart
            st.bar_chart(df_cat.set_index('Category'))
    
    # Load and display actual data points
    st.markdown("**Extracted Data Points**")
    
    # Use the universal output path from the data
    if 'quantitative_dir' in data:
        quantitative_dir = Path(data['quantitative_dir'])
        # Use normalized file if available, otherwise fall back to original
        normalized_file = quantitative_dir / "normalized_extraction_results.jsonl"
        original_file = quantitative_dir / "ultra_fast_extraction_results.jsonl"
        results_file = normalized_file if normalized_file.exists() else original_file
        
        # Show which file is being used
        if normalized_file.exists():
            st.info(f"📊 Displaying **normalized data** from: `{normalized_file.name}`")
        else:
            st.info(f"📊 Displaying **original data** from: `{original_file.name}`")
    else:
        # Fallback to the old method
        results_file = Path(output_dir) / "ultra_fast_extraction_results.jsonl"
    
    if results_file.exists():
        try:
            data_points = []
            with open(results_file, 'r') as f:
                for line in f:
                    if line.strip():
                        data_points.append(json.loads(line))
            
            if data_points:
                # Create table with better data handling
                table_data = []
                for dp in data_points:
                    # Handle value field more robustly
                    value = dp.get('value', 'N/A')
                    if value is not None and value != 'N/A':
                        # Keep the original value as is (don't force conversion)
                        display_value = str(value)
                    else:
                        display_value = 'N/A'
                    
                    table_data.append({
                        'Category': dp.get('category', 'Unknown'),
                        'Value': display_value,
                        'Unit': dp.get('unit', ''),
                        'Actor': dp.get('actor', 'Unknown'),
                        'Legal Article': dp.get('legal_article_violated', 'N/A'),
                        'Quote': dp.get('quote', '')[:100] + "..." if len(dp.get('quote', '')) > 100 else dp.get('quote', ''),
                        'Confidence': f"{dp.get('confidence_score', 0):.1%}"
                    })
                
                df = pd.DataFrame(table_data)
                st.dataframe(df, use_container_width=True)
                
                # Data analysis
                st.markdown("**Analysis**")
                
                # Value distribution analysis
                if 'Value' in df.columns:
                    value_col1, value_col2 = st.columns(2)
                    
                    with value_col1:
                        st.markdown("**Value Distribution**")
                        try:
                            numeric_values = pd.to_numeric(df['Value'], errors='coerce').dropna()
                            text_values = df[df['Value'].apply(lambda x: not str(x).replace('.', '').replace('-', '').isdigit())]
                            
                            if len(numeric_values) > 0:
                                # Create normalized values plot
                                import numpy as np
                                
                                # Normalize the values using min-max normalization
                                min_val = numeric_values.min()
                                max_val = numeric_values.max()
                                normalized_values = (numeric_values - min_val) / (max_val - min_val) if max_val != min_val else numeric_values * 0
                                
                                # Create a scatter plot of normalized values
                                fig_normalized = px.scatter(
                                    x=range(len(normalized_values)),
                                    y=normalized_values,
                                    title="Normalized Values Distribution",
                                    labels={'x': 'Data Point Index', 'y': 'Normalized Value (0-1)'},
                                    color=normalized_values,
                                    color_continuous_scale='viridis'
                                )
                                fig_normalized.update_layout(
                                    yaxis=dict(range=[0, 1]),
                                    showlegend=False
                                )
                                st.plotly_chart(fig_normalized, use_container_width=True, key=f"normalized_values_chart_{id(fig_normalized)}")
                                
                                # Also show a histogram of normalized values
                                fig_hist_norm = px.histogram(
                                    normalized_values,
                                    title="Normalized Values Histogram",
                                    labels={'value': 'Normalized Value', 'count': 'Frequency'},
                                    nbins=min(10, len(normalized_values)),
                                    color_discrete_sequence=['#2E8B57']
                                )
                                fig_hist_norm.update_layout(
                                    xaxis=dict(range=[0, 1])
                                )
                                st.plotly_chart(fig_hist_norm, use_container_width=True, key=f"normalized_histogram_chart_{id(fig_hist_norm)}")
                                
                                if len(text_values) > 0:
                                    st.text(f"Text values: {len(text_values)}")
                            else:
                                st.text("No numeric values")
                        except ImportError:
                            st.warning("Plotly not available")
                    
                    with value_col2:
                        st.markdown("**Statistics**")
                        if len(numeric_values) > 0:
                            st.metric("Numeric", len(numeric_values))
                            st.metric("Text", len(text_values) if 'text_values' in locals() else 0)
                            st.metric("Sum", f"{numeric_values.sum():,}")
                            st.metric("Average", f"{numeric_values.mean():.1f}")
                            
                            # Normalized statistics
                            st.markdown("**Normalized Stats**")
                            st.metric("Min (Original)", f"{numeric_values.min():.1f}")
                            st.metric("Max (Original)", f"{numeric_values.max():.1f}")
                            st.metric("Avg (Normalized)", f"{normalized_values.mean():.3f}")
                            st.metric("Std (Normalized)", f"{normalized_values.std():.3f}")
                        else:
                            st.text("No numeric data")
                
                # Violation Type Scatter Plots - Actual Numeric Values
                st.markdown("**📊 Violation Type Scatter Plots - Actual Numeric Values**")
                
                # Extract numeric values by violation type
                violation_data = {}
                if 'Value' in df.columns and 'Category' in df.columns:
                    for category in df['Category'].unique():
                        if pd.notna(category):
                            category_data = df[df['Category'] == category]
                            numeric_values = pd.to_numeric(category_data['Value'], errors='coerce').dropna()
                            if len(numeric_values) > 0:
                                violation_data[category] = numeric_values
                
                if violation_data:
                    # Create scatter plots for each violation type
                    num_categories = len(violation_data)
                    if num_categories <= 4:
                        # Show all in one row if 4 or fewer categories
                        cols = st.columns(num_categories)
                        for i, (category, values) in enumerate(violation_data.items()):
                            with cols[i]:
                                fig_scatter = px.scatter(
                                    x=range(len(values)),
                                    y=values,
                                    title=f"{category} Violations",
                                    labels={'x': 'Data Point', 'y': 'Value'},
                                    color=values,
                                    color_continuous_scale='viridis'
                                )
                                fig_scatter.update_layout(
                                    height=300,
                                    showlegend=False
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True, key=f"violation_scatter_{category}_{id(fig_scatter)}")
                                
                                # Show summary stats
                                st.caption(f"Count: {len(values)} | Avg: {values.mean():.1f} | Max: {values.max():.1f}")
                    else:
                        # Show in multiple rows if more than 4 categories
                        for i, (category, values) in enumerate(violation_data.items()):
                            if i % 2 == 0:
                                cols = st.columns(2)
                            
                            with cols[i % 2]:
                                fig_scatter = px.scatter(
                                    x=range(len(values)),
                                    y=values,
                                    title=f"{category} Violations",
                                    labels={'x': 'Data Point', 'y': 'Value'},
                                    color=values,
                                    color_continuous_scale='viridis'
                                )
                                fig_scatter.update_layout(
                                    height=300,
                                    showlegend=False
                                )
                                st.plotly_chart(fig_scatter, use_container_width=True, key=f"violation_scatter_{category}_{id(fig_scatter)}")
                                
                                # Show summary stats
                                st.caption(f"Count: {len(values)} | Avg: {values.mean():.1f} | Max: {values.max():.1f}")
                
                # Combined violation comparison
                if len(violation_data) > 1:
                    st.markdown("**📈 Combined Violation Types Comparison**")
                    
                    # Create a combined scatter plot
                    combined_data = []
                    for category, values in violation_data.items():
                        for i, value in enumerate(values):
                            combined_data.append({
                                'Category': category,
                                'Value': value,
                                'Index': i
                            })
                    
                    if combined_data:
                        df_combined = pd.DataFrame(combined_data)
                        fig_combined = px.scatter(
                            df_combined,
                            x='Index',
                            y='Value',
                            color='Category',
                            title="All Violation Types - Numeric Values Comparison",
                            labels={'Index': 'Data Point Index', 'Value': 'Violation Value'},
                            hover_data=['Category', 'Value']
                        )
                        fig_combined.update_layout(height=400)
                        st.plotly_chart(fig_combined, use_container_width=True, key=f"combined_violations_{id(fig_combined)}")
                        
                        # Summary table
                        st.markdown("**Summary by Violation Type**")
                        summary_data = []
                        for category, values in violation_data.items():
                            summary_data.append({
                                'Violation Type': category,
                                'Count': len(values),
                                'Average': f"{values.mean():.1f}",
                                'Max': f"{values.max():.1f}",
                                'Min': f"{values.min():.1f}",
                                'Total': f"{values.sum():.1f}"
                            })
                        
                        df_summary = pd.DataFrame(summary_data)
                        st.dataframe(df_summary, use_container_width=True)
                
                # Actor analysis
                if 'Actor' in df.columns:
                    st.markdown("**Actor Analysis**")
                    actor_counts = df['Actor'].value_counts()
                    if len(actor_counts) > 0:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Frequency**")
                            st.dataframe(actor_counts.reset_index().rename(columns={'index': 'Actor', 'Actor': 'Count'}))
                        
                        with col2:
                            try:
                                fig_actor = px.pie(
                                    values=actor_counts.values,
                                    names=actor_counts.index,
                                    title="Data Points by Actor"
                                )
                                st.plotly_chart(fig_actor, use_container_width=True, key=f"actor_analysis_chart_{id(fig_actor)}")
                            except ImportError:
                                st.warning("Plotly not available")
                
                # Download buttons
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv_data,
                        file_name="fast_path_results.csv",
                        mime="text/csv"
                    )
                
                with download_col2:
                    # Also provide JSON download
                    json_data = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="Download Results (JSON)",
                        data=json_data,
                        file_name="fast_path_results.json",
                        mime="application/json"
                    )
            else:
                st.warning("No data points found")
        except Exception as e:
            st.error(f"Error loading data: {e}")
    else:
        st.error("Results file not found")

def run_interactive_ai_agent(quantitative_file, bias_file, ai_analysis_file, output_dir):
    """Initialize and run the interactive AI agent"""
    try:
        st.info("Initializing Interactive AI Agent...")
        
        # Get the project root
        current_dir = Path.cwd()
        project_root = current_dir.parent if current_dir.name == "UI" else current_dir
        scripts_dir = project_root / "extraction" / "scripts"
        
        # For now, we'll use the existing AI analysis agent and create an interactive interface
        # The interactive functionality will be built into the Streamlit interface
        
        st.success("Interactive AI Agent initialized successfully!")
        return True, {
            'quantitative_file': str(quantitative_file),
            'bias_file': str(bias_file),
            'ai_analysis_file': str(ai_analysis_file) if ai_analysis_file else None
        }
        
    except Exception as e:
        return False, f"Interactive AI Agent error: {str(e)}"

# This function has been removed as it's duplicated in the main flow

def get_ai_response(question, quantitative_file, bias_file, ai_analysis_file=None):
    """Get an interactive AI response to a question"""
    try:
        # Import the interactive agent
        try:
            # Get the project root and add scripts directory to path
            current_dir = Path.cwd()
            project_root = current_dir.parent if current_dir.name == "UI" else current_dir
            scripts_dir = project_root / "extraction" / "scripts"
            sys.path.insert(0, str(scripts_dir))
            from interactive_ai_agent import InteractiveAIAgent
        except ImportError:
            return "❌ Interactive AI agent not available. Please ensure the system is properly installed."
        
        # Initialize agent
        agent = InteractiveAIAgent()
        
        # Load data - only pass non-None files
        files_to_load = []
        if quantitative_file:
            files_to_load.append(quantitative_file)
        if bias_file:
            files_to_load.append(bias_file)
        if ai_analysis_file:
            files_to_load.append(ai_analysis_file)
            
        if not files_to_load:
            return f"❌ No analysis data files found. Please run an analysis first. (quantitative_file: {quantitative_file}, bias_file: {bias_file}, ai_analysis_file: {ai_analysis_file})"
            
        agent.load_analysis_data(*files_to_load)
        
        if not agent.data_loaded:
            return "❌ Failed to load analysis data"
        
        # Get response
        response = agent.ask_question(question)
        return response
        
    except Exception as e:
        return f"❌ Error getting AI response: {str(e)}"

def main():
    # Initialize session state for chat history
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []
    
    st.markdown('<h1 class="main-title">UN Report Analysis Platform</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Developed by <strong>Moumen Alaoui</strong> | <a href="mailto:moumenalaoui@proton.me">moumenalaoui@proton.me</a> | <a href="https://www.linkedin.com/in/moumenalaoui">LinkedIn</a></p>', unsafe_allow_html=True)

    
    # Initialize session state for storing results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'bias_results' not in st.session_state:
        st.session_state.bias_results = None
    if 'ai_analysis_results' not in st.session_state:
        st.session_state.ai_analysis_results = None
    if 'extraction_completed' not in st.session_state:
        st.session_state.extraction_completed = False
    
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload a UN Security Council report (PDF)",
        type=['pdf']
    )
    
    if uploaded_file:
        # Simple file confirmation
        st.markdown(f"**File:** {uploaded_file.name}")
        
        # Minimal file info
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Size: {uploaded_file.size / 1024:.1f} KB")
        with col2:
            st.text(f"Type: {uploaded_file.type}")
        
        # Analysis options
        st.markdown("**Analysis Configuration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_quantitative = st.checkbox(
                "Include Quantitative Extraction",
                value=True,
                help="Extract quantitative data (violations, actors, locations, dates) from the report. This is the core analysis."
            )
            
            include_bias_analysis = st.checkbox(
                "Include Bias Analysis",
                help="Analyze the report for bias, framing, and language patterns using Entman's theory. Provides deeper insights into reporting bias."
            )
        
        with col2:
            include_ai_agent = st.checkbox(
                "Include AI Agent Analysis",
                help="Get comprehensive AI-powered analysis combining quantitative data and bias analysis. Requires at least one analysis type to be enabled."
            )
            
            # Show bias analysis specific options
            if include_bias_analysis:
                st.markdown("**Bias Analysis Options**")
                process_all_paragraphs = st.checkbox(
                    "Process All Paragraphs",
                    value=True,
                    help="Process all paragraphs in the report (recommended for comprehensive analysis). Uncheck to limit processing."
                )
        
        # Note: Interactive AI Chat is always available after analysis
        st.info("**Interactive AI Chat will be available after analysis completion**")
        
        # LLM Call Configuration (for quantitative extraction)
        if include_quantitative:
            st.markdown("**LLM Call Configuration (Quantitative Extraction)**")
            
            # Simple preset selection
            call_presets = {
                "Ultra-Fast (1-2 calls)": {"min": 1, "max": 2},
                "Fast (3-5 calls)": {"min": 3, "max": 5},
                "Standard (6-8 calls)": {"min": 6, "max": 8},
                "Research (9-15 calls)": {"min": 9, "max": 15},
                "Custom": {"min": 1, "max": 20}
            }
            
            selected_preset = st.selectbox("Quantitative Strategy:", list(call_presets.keys()))
            
            if selected_preset == "Custom":
                col1, col2 = st.columns(2)
                with col1:
                    min_calls = st.slider("Min calls:", 1, 20, 3)
                with col2:
                    max_calls = st.slider("Max calls:", min_calls, 20, 5)
            else:
                preset = call_presets[selected_preset]
                min_calls = preset["min"]
                max_calls = preset["max"]
        else:
            # Default values when quantitative extraction is disabled
            min_calls = 3
            max_calls = 5
        
        # Simple estimates
        avg_calls = (min_calls + max_calls) / 2
        
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Calls: {min_calls}-{max_calls}")
        with col2:
            st.text(f"Avg: {avg_calls:.1f}")
        
        if include_ai_agent and not include_bias_analysis:
            st.warning("AI Agent requires Bias Analysis")
            include_ai_agent = False
        

        
        # Run extraction
        if st.button("Run Extraction"):
            # Get project root directory
            current_dir = Path.cwd()
            project_root = current_dir.parent if current_dir.name == "UI" else current_dir
            
            # Clear previous session results and files
            st.session_state.extraction_completed = False
            st.session_state.analysis_results = None
            st.session_state.bias_results = None
            st.session_state.ai_analysis_results = None
            st.session_state.ai_chat_history = []  # Clear AI chat history
            
            # Clear old output files
            try:
                import shutil
                universal_output_dir = project_root / "extraction" / "universal_output"
                if universal_output_dir.exists():
                    shutil.rmtree(universal_output_dir)
                    universal_output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                st.warning(f"Could not clear old results: {e}")
            
            with st.spinner("Processing..."):
                # Create temporary directory
                temp_dir = tempfile.mkdtemp(prefix=f"fast_path_{uploaded_file.name.replace('.pdf', '')}_")
                temp_path = Path(temp_dir)
                
                try:
                    st.info(f"Processing {uploaded_file.name}...")
                    
                    # Run extraction
                    success, output = run_fast_path_extraction(
                        uploaded_file, temp_path, 
                        include_quantitative=include_quantitative,
                        include_bias_analysis=include_bias_analysis,
                        process_all_paragraphs=process_all_paragraphs if include_bias_analysis else True,
                        min_calls=min_calls, max_calls=max_calls
                    )
                    
                    if success and isinstance(output, dict):
                        # Store results in session state (for potential future use)
                        st.session_state.analysis_results = output
                        
                        # Apply data normalization (only if quantitative extraction was enabled)
                        if include_quantitative:
                            quant_file = output.get('quantitative_file')
                            if quant_file and os.path.exists(quant_file):
                                normalized_file = os.path.join(os.path.dirname(quant_file), 'normalized_extraction_results.jsonl')
                                normalization_result = normalize_extracted_data(str(quant_file), str(normalized_file))
                                
                                if normalization_result['success']:
                                    st.success(normalization_result['message'])
                                    # Update the output to use normalized file
                                    output['quantitative_file'] = normalized_file
                                else:
                                    st.warning(f"Normalization failed: {normalization_result['message']}")
                        
                        # Display results
                        display_results(output, temp_path)
                        
                        # Mark extraction as completed for this session
                        st.session_state.extraction_completed = True
                        
                        # Store bias analysis results for later display
                        if include_bias_analysis and output.get('bias_analysis_completed', False):
                            bias_file_path = output.get('bias_analysis_path')
                            if bias_file_path:
                                st.session_state.bias_results = bias_file_path
                                
                                # Run AI analysis if requested
                                if include_ai_agent:
                                    st.markdown("**AI Agent Analysis**")
                                    
                                    # Get the quantitative results file path (if quantitative extraction was enabled)
                                    quantitative_file = None
                                    if include_quantitative:
                                        quantitative_file_path = output.get('quantitative_dir')
                                        if quantitative_file_path:
                                            quantitative_file = Path(quantitative_file_path) / "ultra_fast_extraction_results.jsonl"
                                            if not quantitative_file.exists():
                                                st.warning("Quantitative results file not found for AI analysis")
                                                quantitative_file = None
                                    
                                    # Run AI analysis with available data
                                    ai_success, ai_output = run_ai_analysis(quantitative_file, bias_file_path, temp_path)
                                    
                                    if ai_success:
                                        display_ai_analysis_results(ai_output)
                                        st.session_state.ai_analysis_results = ai_output
                                    else:
                                        st.error(f"AI analysis failed: {ai_output}")
                            else:
                                st.error("Bias analysis path not found in output")
                        elif include_bias_analysis:
                            st.warning("Bias analysis was requested but failed to complete")
                        
                        # Download summary
                        json_str = json.dumps(output, indent=2)
                        st.download_button(
                            label="Download Summary (JSON)",
                            data=json_str,
                            file_name="fast_path_summary.json",
                            mime="application/json",
                            key=f"summary_download_{id(json_str)}"
                        )

                        # AI Chat interface is now available separately below

                    else:
                        st.error(f"Extraction failed: {output}")
                        
                finally:
                    # Clean up
                    try:
                        shutil.rmtree(temp_path)
                    except:
                        pass
        
        # No need to display results again - they're already shown above
        pass
    
    # Interactive AI Chat with NO RELOAD using st.form - ALWAYS AVAILABLE
    # This is the EXACT method from your working code
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []
    
    # Show AI chat interface if we have any analysis results
    # Use normalized file if available, otherwise fall back to original
    # Get the project root directory
    current_dir = Path.cwd()
    project_root = current_dir.parent if current_dir.name == "UI" else current_dir
    
    normalized_file = project_root / "extraction" / "universal_output" / "quantitative_extraction" / "normalized_extraction_results.jsonl"
    original_file = project_root / "extraction" / "universal_output" / "quantitative_extraction" / "ultra_fast_extraction_results.jsonl"
    quant_file = normalized_file if normalized_file.exists() else original_file
    
    # Bias analysis is in its own directory, AI analysis might be in quantitative_extraction
    bias_file = project_root / "extraction" / "universal_output" / "bias_analysis" / "text_bias_analysis_results.jsonl"
    ai_file = project_root / "extraction" / "universal_output" / "quantitative_extraction" / "ai_analysis_report.json"
    
    # Show bias analysis results only if they're from the current session
    if bias_file.exists() and st.session_state.get('extraction_completed', False):
        st.markdown("---")
        st.markdown('<h2 class="section-header">🔍 Bias Analysis Results</h2>', unsafe_allow_html=True)
        st.markdown("Comprehensive bias analysis using Entman's framing theory and advanced AI techniques.")
        display_bias_results(bias_file)
    
    # ALWAYS show AI agent - it will work with whatever files are available
    st.markdown("---")
    st.markdown('<h2 class="section-header">💬 Interactive AI Agent</h2>', unsafe_allow_html=True)
    st.markdown("Ask questions about your analysis results and get intelligent, data-driven answers!")
    
    # Display chat history
    for message in st.session_state.ai_chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Use st.form to prevent page reload - EXACT method from working code
    with st.form("ai_chat_form", clear_on_submit=True):
        user_question = st.text_input("Ask a question about the analysis...", key="chat_input")
        submit_button = st.form_submit_button("🤖 Ask AI", type="primary")
        
        if submit_button and user_question:
            # Add user message to chat history
            st.session_state.ai_chat_history.append({"role": "user", "content": user_question})
            
            # Get AI response
            with st.spinner("🤖 AI Agent is thinking..."):
                # Only pass files that exist
                quant_path = str(quant_file) if quant_file.exists() else None
                bias_path = str(bias_file) if bias_file.exists() else None
                ai_path = str(ai_file) if ai_file.exists() else None
                
                # Debug info
                st.info(f"Debug: quant_file={quant_file}, exists={quant_file.exists()}, quant_path={quant_path}")
                st.info(f"Debug: bias_file={bias_file}, exists={bias_file.exists()}, bias_path={bias_path}")
                st.info(f"Debug: ai_file={ai_file}, exists={ai_file.exists()}, ai_path={ai_path}")
                
                response = get_ai_response(user_question, quant_path, bias_path, ai_path)
            
            # Add AI response to chat history
            st.session_state.ai_chat_history.append({"role": "assistant", "content": response})
            
            # Force rerun to update chat display
            st.rerun()
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", key="clear_chat"):
        st.session_state.ai_chat_history = []
        st.rerun()
    
    # Suggested questions
    with st.expander("💡 Suggested Questions", expanded=False):
        st.markdown("""
        **Try asking:**
        - What are the main patterns of bias in this report?
        - Which actor has the most violations according to the data?
        - How does Entman's framing theory apply to this analysis?
        - What are the most significant legal violations found?
        - Can you compare the actions of different actors?
        - What recommendations would you make based on this analysis?
        - How reliable is the quantitative data in this report?
        - What are the limitations of this bias analysis?
        - Which UNSCR 1701 articles are most frequently violated?
        - How does the selection bias manifest in this report?
        - What framing bias indicators were detected?
        - How does the language analysis reveal bias patterns?
        - Which paragraphs show the strongest evidence of omission bias?
        - Can you analyze the moral evaluation patterns in the framing?
        - What systemic bias indicators were found?
                 """)

    # Professional footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em; margin-top: 2rem;'>
        <p><strong>UN Report Analysis Platform</strong> | Developed by <strong>Moumen Alaoui</strong></p>
        <p><a href="mailto:moumenalaoui@proton.me" style="color: #666;">moumenalaoui@proton.me</a> | 
           <a href="https://www.linkedin.com/in/moumenalaoui" style="color: #666;">LinkedIno </a></p>
        <p style="font-size: 0.8em; margin-top: 1rem;">AI-Powered Intelligence Platform for Advanced Policy Analysis</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 