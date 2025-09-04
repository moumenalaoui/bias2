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
st.sidebar.markdown("## 🔑 OpenAI API Configuration")
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
    with st.sidebar.expander("💰 Cost Estimation", expanded=False):
        st.markdown("""
        **Typical costs per analysis:**
        - **Fast Path (≤3 calls)**: ~$0.01-0.03
        - **With Bias Analysis**: ~$0.05-0.15
        - **With AI Agent**: ~$0.01-0.05 per question
        
        **Model Pricing:**
        - GPT-4o: $5/1M input tokens, $15/1M output tokens
        - GPT-4o-mini: $0.15/1M input tokens, $0.60/1M output tokens
        
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

def run_fast_path_extraction(pdf_file, output_dir, include_bias_analysis=False, min_calls=3, max_calls=3):
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
            "python3", "hybrid_extractor.py", str(pdf_dest), str(pdf_conversion_dir)
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
        
        # Run ultra-fast extraction with user-configured LLM calls
        fast_result = subprocess.run([
            "python3", "ultra_fast_quantitative_extractor.py",
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
        
        # If bias analysis is requested, run it now
        if include_bias_analysis:
            st.info(f"Running bias analysis on {jsonl_file.name}...")
            
            bias_success, bias_output = run_bias_analysis(jsonl_file, bias_analysis_dir)
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
        summary['quantitative_file'] = str(results_file)  # Add the quantitative file path
        summary['llm_calls_config'] = {'min_calls': min_calls, 'max_calls': max_calls, 'actual_calls': summary.get('llm_calls_used', 0)}
        
        return True, summary
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def run_bias_analysis(jsonl_file_path, output_dir):
    """Run bias analysis on the extracted JSONL data"""
    try:
        st.info("Starting bias analysis...")
        
        # Get the project root
        current_dir = Path.cwd()
        project_root = current_dir.parent if current_dir.name == "UI" else current_dir
        scripts_dir = project_root / "extraction" / "scripts"
        
        # Run bias analysis
        bias_result = subprocess.run([
            "python3", "call_api_bias_optimized.py", 
            "--input", str(jsonl_file_path),
            "--output", str(output_dir),
            "--test"  # Start with test mode for faster results
        ], capture_output=True, text=True, cwd=scripts_dir, timeout=600)
        
        if bias_result.returncode != 0:
            return False, f"Bias analysis failed: {bias_result.stderr}"
        
        # Check for bias analysis results
        bias_output_file = output_dir / "text_bias_analysis_results.jsonl"
        if bias_output_file.exists():
            st.success("Bias analysis completed successfully!")
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
            "python3", "ai_analysis_agent.py", 
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
    """Display bias analysis results with simple histogram"""
    try:
        st.markdown("**Bias Analysis Results**")
        
        # Load bias analysis data
        bias_data = []
        with open(bias_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    bias_data.append(json.loads(line))
        
        if not bias_data:
            st.warning("No bias analysis data found")
            return
        
        st.success(f"Found {len(bias_data)} bias analysis entries")
        
        # Display bias data with proper field mapping
        for i, entry in enumerate(bias_data[:10]):  # Show first 10 entries
            with st.expander(f"Entry {i+1}: {entry.get('paragraph_id', 'Unknown')}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Text Snippet**")
                    text_content = entry.get('text', 'No text available')
                    if len(text_content) > 200:
                        st.text(text_content[:200] + "...")
                    else:
                        st.text(text_content)
                    
                    st.markdown("**🏷️ Bias Type**")
                    bias_flag = entry.get('bias_flag', 'Unknown')
                    if bias_flag == 'none':
                        st.success("No bias detected")
                    elif bias_flag == 'framing':
                        st.warning("Framing bias detected")
                    elif bias_flag == 'selection':
                        st.warning("Selection bias detected")
                    elif bias_flag == 'omission':
                        st.warning("Omission bias detected")
                    else:
                        st.info(bias_flag)
                    
                    st.markdown("**Core Actors**")
                    actors = entry.get('core_actors', [])
                    if actors:
                        for actor in actors:
                            st.write(f"• {actor}")
                    else:
                        st.write("No actors identified")
                
                with col2:
                    st.markdown("**Bias Analysis**")
                    bias_reason = entry.get('bias_reason', 'No analysis available')
                    st.text(bias_reason[:300] + "..." if len(bias_reason) > 300 else bias_reason)
                    
                    st.markdown("**Violation Types**")
                    violations = entry.get('violation_type', [])
                    if violations:
                        for violation in violations:
                            st.write(f"• {violation}")
                    else:
                        st.write("No violations identified")
                    
                    st.markdown("**📍 Location**")
                    location = entry.get('location', 'Not specified')
                    st.info(location)
        
        # Summary statistics with simple histogram
        st.markdown("**Bias Analysis Summary**")
        
        # Count bias types
        bias_counts = {}
        for entry in bias_data:
            bias_type = entry.get('bias_flag', 'unknown')
            bias_counts[bias_type] = bias_counts.get(bias_type, 0) + 1
        
        if bias_counts:
            # Show bias distribution as a simple histogram
            st.markdown("**Bias Category Distribution**")
            
            # Create histogram chart
            try:
                import plotly.express as px
                df_bias = pd.DataFrame(list(bias_counts.items()), columns=['Bias Type', 'Count'])
                
                # Create histogram
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
                st.plotly_chart(fig, use_container_width=True, key=f"value_distribution_chart_{id(fig)}")
                
                # Simple text summary below the chart
                st.markdown("**Summary:**")
                for bias_type, count in bias_counts.items():
                    if bias_type == 'none':
                        st.write(f"• **No Bias Detected**: {count} paragraphs")
                    else:
                        st.write(f"• **{bias_type.title()} Bias**: {count} paragraphs")
                        
            except ImportError:
                st.info("Plotly not available for charts - showing text summary only")
                for bias_type, count in bias_counts.items():
                    if bias_type == 'none':
                        st.success(f"No Bias: {count}")
                    else:
                        st.warning(f"{bias_type.title()}: {count}")
        
        # Download bias results
        with open(bias_file_path, 'r', encoding='utf-8') as f:
            bias_content = f.read()
        
        st.download_button(
            label="Download Bias Analysis (JSONL)",
            data=bias_content,
            file_name="bias_analysis_results.jsonl",
            mime="application/json"
        )
        
    except Exception as e:
        st.error(f"Error displaying bias results: {e}")
        st.info(f"Debug: File path: {bias_file_path}")
        st.info(f"Debug: Error details: {str(e)}")

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
        max_llm_calls = 3
        st.metric("LLM Calls", f"{llm_calls}/{max_llm_calls}")
        st.progress(llm_calls / max_llm_calls)
    
    with perf_col2:
        total_points = data.get('total_data_points', 0)
        if total_points > 0:
            points_per_call = total_points / llm_calls
            st.metric("Rate", f"{points_per_call:.1f} points/call")
    
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
                                fig_hist = px.histogram(
                                    numeric_values,
                                    title="Numeric Values Distribution",
                                    labels={'value': 'Value', 'count': 'Frequency'},
                                    nbins=min(10, len(numeric_values))
                                )
                                st.plotly_chart(fig_hist, use_container_width=True, key=f"value_histogram_chart_{id(fig_hist)}")
                                
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
                        else:
                            st.text("No numeric data")
                
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
    st.markdown('<p class="subtitle">Developed by Moumen Alaoui</p>', unsafe_allow_html=True)
    
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
        
        # Bias analysis option
        st.markdown("**Analysis Configuration**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_bias_analysis = st.checkbox(
                "Include Bias Analysis",
                help="Enable this to also analyze the report for bias, framing, and language patterns. This will add additional processing time but provide deeper insights."
            )
            
            include_ai_agent = st.checkbox(
                "Include AI Agent Analysis",
                help="Enable this to get comprehensive AI-powered analysis combining quantitative data and bias analysis. Requires bias analysis to be enabled."
            )
        
        # Note: Interactive AI Chat is always available after analysis
        st.info("**Interactive AI Chat will be available after analysis completion**")
        
        # LLM Call Configuration
        st.markdown("**LLM Call Configuration**")
        
        # Simple preset selection
        call_presets = {
            "Ultra-Fast (1-2 calls)": {"min": 1, "max": 2},
            "Fast (3-5 calls)": {"min": 3, "max": 5},
            "Standard (6-8 calls)": {"min": 6, "max": 8},
            "Research (9-15 calls)": {"min": 9, "max": 15},
            "Custom": {"min": 1, "max": 20}
        }
        
        selected_preset = st.selectbox("Strategy:", list(call_presets.keys()))
        
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
            with st.spinner("Processing..."):
                # Create temporary directory
                temp_dir = tempfile.mkdtemp(prefix=f"fast_path_{uploaded_file.name.replace('.pdf', '')}_")
                temp_path = Path(temp_dir)
                
                try:
                    st.info(f"Processing {uploaded_file.name}...")
                    
                    # Run extraction
                    success, output = run_fast_path_extraction(uploaded_file, temp_path, include_bias_analysis, min_calls, max_calls)
                    
                    if success and isinstance(output, dict):
                        # Store results in session state (for potential future use)
                        st.session_state.analysis_results = output
                        
                        # Apply data normalization
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
                        
                        # Run bias analysis if requested
                        if include_bias_analysis and output.get('bias_analysis_completed', False):
                            st.markdown("**Bias Analysis Results**")
                            bias_file_path = output.get('bias_analysis_path')
                            if bias_file_path:
                                display_bias_results(bias_file_path)
                                st.session_state.bias_results = bias_file_path
                                
                                # Run AI analysis if requested
                                if include_ai_agent:
                                    st.markdown("**AI Agent Analysis**")
                                    
                                    # Get the quantitative results file path
                                    quantitative_file_path = output.get('quantitative_dir')
                                    if quantitative_file_path:
                                        quantitative_file = Path(quantitative_file_path) / "ultra_fast_extraction_results.jsonl"
                                        if quantitative_file.exists():
                                            ai_success, ai_output = run_ai_analysis(quantitative_file, bias_file_path, temp_path)
                                            
                                            if ai_success:
                                                display_ai_analysis_results(ai_output)
                                                st.session_state.ai_analysis_results = ai_output
                                            else:
                                                st.error(f"AI analysis failed: {ai_output}")
                                        else:
                                            st.error("Quantitative results file not found for AI analysis")
                                    else:
                                        st.error("Quantitative directory path not found")
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
        """)

if __name__ == "__main__":
    main() 