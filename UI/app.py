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
import time
import logging
from typing import Tuple

# Add the parent directory to the path so we can import our scripts
sys.path.append(str(Path(__file__).parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="UN Report Analysis Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional minimalist styling
st.markdown("""
<style>
    /* Clean white background */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Main container */
    .main > div {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Headers - Washington Institute style - FORCE OVERRIDE */
    .main-header {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #1e3a8a !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
        letter-spacing: -0.025em !important;
    }
    
    .section-header {
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        color: #1e3a8a !important;
        margin: 2rem 0 1rem 0 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 1px solid #e5e7eb !important;
    }
    
    /* Force override any h1, h2, h3 elements */
    h1, h2, h3 {
        color: #1e3a8a !important;
    }
    
    /* Buttons - Washington Institute style - FORCE OVERRIDE */
    .stButton > button {
        background: #1e3a8a !important;
        color: white !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 0.025em !important;
    }
    
    .stButton > button:hover {
        background: #1e40af !important;
        transform: translateY(-1px) !important;
    }
    
    /* Force override for primary buttons */
    button[kind="primary"], button[data-testid="baseButton-primary"] {
        background: #1e3a8a !important;
        color: white !important;
        border: none !important;
    }
    
    button[kind="primary"]:hover, button[data-testid="baseButton-primary"]:hover {
        background: #1e40af !important;
    }
    
    /* File uploader */
    .stFileUploader > div {
        border: 2px dashed #d1d5db;
        border-radius: 8px;
        padding: 2rem;
        background: #fafafa;
        text-align: center;
        transition: all 0.2s ease;
    }
    
    /* Tabs - Clean minimal */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: transparent;
        border-radius: 0;
        color: #6b7280;
        font-weight: 500;
        font-size: 0.875rem;
        border: none;
        border-bottom: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: transparent !important;
        color: #1e3a8a !important;
        border-bottom: 2px solid #1e3a8a !important;
    }
    
    /* Metrics - Clean cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e5e7eb;
        padding: 1.5rem;
        border-radius: 8px;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: #1e3a8a;
        border-radius: 2px;
    }
    
    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
</style>
""", unsafe_allow_html=True)

def check_environment():
    """Check if the required environment and files exist"""
    issues = []
    warnings = []
    
    # Check if bias_env2 exists
    if not Path("../bias_env2").exists():
        issues.append("Virtual environment 'bias_env2' not found")
    
    # Check if required scripts exist
    required_scripts = [
        "../extraction/scripts/gpt_quantitative_extractor.py",
        "../extraction/scripts/call_api_bias_optimized.py",
        "../extraction/scripts/actor_disambiguation.py",
        "../extraction/scripts/simple_analysis.py",
        "../extraction/scripts/hybrid_extractor.py",
        "../extraction/scripts/batch_resolution_fetcher.py"
    ]
    
    for script in required_scripts:
        if not Path(script).exists():
            issues.append(f"Required script not found: {script}")
    
    # Check if .env file exists
    if not Path("../.env").exists():
        issues.append("OpenAI API key file (.env) not found")
    else:
        # Check if API key is valid
        try:
            from dotenv import load_dotenv
            load_dotenv("../.env")
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                issues.append("OpenAI API key not found in .env file")
            elif len(api_key) < 20:
                issues.append("OpenAI API key appears to be invalid (too short)")
        except Exception as e:
            warnings.append(f"Could not validate API key: {str(e)}")
    
    # Check if required directories exist
    required_dirs = [
        "../extraction/reports_pdf",
        "../extraction/hybrid_output",
        "../extraction/JSONL_outputs",
        "../precision_hybrid_results"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            warnings.append(f"Directory not found (will be created): {dir_path}")
    
    return issues, warnings

def test_api_connection():
    """Test OpenAI API connection"""
    try:
        from dotenv import load_dotenv
        import openai
        
        load_dotenv("../.env")
        api_key = os.getenv('OPENAI_API_KEY')
        
        if not api_key:
            return False, "No API key found"
        
        # Test with a simple API call
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            temperature=0,  # Maximum precision
            top_p=0.9,  # Focus on most likely tokens
            frequency_penalty=0.1,  # Reduce repetition
            presence_penalty=0.1  # Encourage diverse coverage
        )
        
        return True, "API connection successful"
    except Exception as e:
        return False, f"API connection failed: {str(e)}"

def run_quantitative_extraction(pdf_file, temp_dir, max_workers=6, batch_size=5):
    """Run the quantitative extraction pipeline with improved error handling"""
    try:
        # Save uploaded file to the expected location
        pdf_dest = Path("../extraction/reports_pdf") / pdf_file.name
        pdf_dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists and has proper permissions
        if pdf_dest.exists():
            # Check if file is writable
            if not os.access(pdf_dest, os.W_OK):
                # Try to fix permissions
                try:
                    os.chmod(pdf_dest, 0o644)
                except Exception as e:
                    return False, f"File {pdf_file.name} exists but is not writable. Please check file permissions: {str(e)}"
        
        # Save the uploaded file content
        try:
            with open(pdf_dest, "wb") as f:
                f.write(pdf_file.getbuffer())
        except PermissionError:
            return False, f"Permission denied when trying to write to {pdf_file.name}. Please check file permissions or try a different filename."
        except Exception as e:
            return False, f"Error saving file {pdf_file.name}: {str(e)}"
        
        # Run the pipeline with improved parameters
        cmd = [
            "bash", "run_bias2.sh", str(pdf_dest.resolve())
        ]
        
        # Set environment variables for better performance
        env = dict(os.environ, 
                  VIRTUAL_ENV=str(Path("../bias_env2").resolve()),
                  MAX_WORKERS=str(max_workers),
                  BATCH_SIZE=str(batch_size),
                  PYTHONPATH=str(Path("../").resolve()))
        
        # Run with timeout and better error handling
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path("../").resolve(),
            env=env,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Pipeline failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\n\nError output:\n{result.stderr}"
            if result.stdout:
                error_msg += f"\n\nStandard output:\n{result.stdout}"
            return False, error_msg
        
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        return False, "Pipeline timed out after 30 minutes"
    except Exception as e:
        return False, f"Error running quantitative extraction: {str(e)}"

def run_bias_analysis(jsonl_file, output_dir):
    """Run bias analysis on the JSONL file with improved error handling"""
    try:
        # Convert paths to absolute paths
        jsonl_abs = Path(jsonl_file).resolve()
        output_abs = Path(output_dir).resolve()
        
        # Ensure output directory exists
        output_abs.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python3", "call_api_bias_optimized.py",
            "--input", str(jsonl_abs),
            "--output", str(output_abs),
            "--max-concurrent", "6",
            "--batch-size", "10"
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path("../extraction/scripts").resolve(),
            env=dict(os.environ, 
                    VIRTUAL_ENV=str(Path("../../bias_env2").resolve()),
                    PYTHONPATH=str(Path("../../").resolve())),
            timeout=900  # 15 minute timeout
        )
        
        if result.returncode != 0:
            error_msg = f"Bias analysis failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\n\nError output:\n{result.stderr}"
            return False, error_msg
        
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        return False, "Bias analysis timed out after 15 minutes"
    except Exception as e:
        return False, f"Error running bias analysis: {str(e)}"

def jsonl_to_csv(jsonl_file, csv_file):
    """Convert JSONL file to CSV with improved error handling"""
    try:
        data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        continue
        
        if not data:
            return False, "No valid data found in JSONL file"
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        return True, len(data)
        
    except Exception as e:
        return False, f"Error converting to CSV: {str(e)}"

def create_download_link(file_path, link_text, key=None):
    """Create a download link for a file"""
    try:
        with open(file_path, "rb") as f:
            data = f.read()
        
        # Generate unique key if not provided
        if key is None:
            key = f"download_{Path(file_path).stem}_{hash(link_text)}"
        
        return st.download_button(
            label=link_text,
            data=data,
            file_name=Path(file_path).name,
            mime="text/csv",
            key=key
        )
    except Exception as e:
        st.error(f"Error creating download link: {str(e)}")
        return None

def get_actual_paragraph_count(jsonl_file_path):
    """Get actual paragraph count from JSONL file for more accurate estimates"""
    try:
        if not Path(jsonl_file_path).exists():
            return None
        
        count = 0
        with open(jsonl_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    except Exception:
        return None

def estimate_processing_time(file_size_mb, batch_size, max_workers, actual_paragraphs=None):
    """Estimate processing time based on file size and configuration with realistic performance"""
    # Realistic base estimation: 8-12 seconds per API call including rate limiting
    base_time_per_request = 10.0  # More realistic based on actual performance
    
    # Use actual paragraph count if available, otherwise estimate based on file size
    if actual_paragraphs is not None:
        estimated_paragraphs = actual_paragraphs
    else:
        # Estimate paragraphs based on file size (more accurate)
        if file_size_mb < 0.5:
            estimated_paragraphs = 10
        elif file_size_mb < 1:
            estimated_paragraphs = 25
        elif file_size_mb < 2:
            estimated_paragraphs = 50
        elif file_size_mb < 5:
            estimated_paragraphs = 100
        elif file_size_mb < 10:
            estimated_paragraphs = 200
        else:
            estimated_paragraphs = 300
    
    # Account for rate limiting and conservative parallel processing
    rate_limit_factor = 1.5  # Conservative due to rate limiting
    parallel_efficiency = 0.6  # Conservative efficiency with rate limits
    
    # Calculate time based on actual workers and batch size
    effective_workers = min(max_workers, 4)  # Conservative cap
    requests_per_batch = batch_size
    
    # Apply batching optimization factor (more conservative)
    batching_efficiency = 1.2 if batch_size > 1 else 1.0
    
    # Calculate total requests needed
    total_requests = estimated_paragraphs / requests_per_batch
    
    # Calculate time in minutes
    estimated_time = (total_requests * base_time_per_request * rate_limit_factor) / (effective_workers * parallel_efficiency * batching_efficiency * 60)
    
    # Add buffer for processing overhead
    estimated_time = estimated_time * 1.2
    
    return estimated_time, estimated_paragraphs

def run_multi_report_analysis(uploaded_files, temp_path: Path) -> Tuple[bool, str]:
    """Run multi-report intelligence analysis"""
    try:
        # Import the batch processor
        try:
            # Add the multi_report_analysis directory to sys.path
            multi_report_dir = Path("../multi_report_analysis").resolve()
            if str(multi_report_dir) not in sys.path:
                sys.path.insert(0, str(multi_report_dir))
            
            # Also add the extraction/scripts directory
            extraction_scripts_dir = Path("../extraction/scripts").resolve()
            if str(extraction_scripts_dir) not in sys.path:
                sys.path.insert(0, str(extraction_scripts_dir))
            
            # Now import the batch processor
            import batch_processor
            MultiReportBatchProcessor = batch_processor.MultiReportBatchProcessor
            
        except ImportError as e:
            return False, f"Multi-report analysis module import failed: {str(e)}"
        except Exception as e:
            return False, f"Multi-report analysis system error: {str(e)}"
        
        # Save uploaded files to temporary directory
        pdf_paths = []
        report_names = []
        
        for uploaded_file in uploaded_files:
            # Create safe filename
            safe_name = uploaded_file.name.replace(' ', '_').replace('(', '').replace(')', '')
            pdf_path = temp_path / safe_name
            
            # Save file
            with open(pdf_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            pdf_paths.append(str(pdf_path))
            report_names.append(Path(safe_name).stem)
        
        # Initialize and run batch processor
        processor = MultiReportBatchProcessor()
        results = processor.process_multiple_reports(pdf_paths, report_names)
        
        if "error" in results:
            return False, f"Multi-report analysis failed: {results['error']}"
        
        return True, f"Multi-report analysis completed successfully. {results['successful_analyses']} reports processed."
        
    except Exception as e:
        return False, f"Multi-report analysis error: {str(e)}"

def run_ai_analysis(quantitative_file, bias_file, output_dir):
    """Run the AI analysis agent to generate intelligent insights"""
    try:
        # Convert paths to absolute paths
        quant_abs = Path(quantitative_file).resolve()
        bias_abs = Path(bias_file).resolve()
        output_abs = Path(output_dir).resolve()
        
        # Ensure output directory exists
        output_abs.mkdir(parents=True, exist_ok=True)
        
        # Run AI analysis
        cmd = [
            "python3", "ai_analysis_agent.py",
            "--quantitative", str(quant_abs),
            "--bias", str(bias_abs),
            "--output", str(output_abs / "ai_analysis_report.json")
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path("../extraction/scripts").resolve(),
            env=dict(os.environ, 
                    VIRTUAL_ENV=str(Path("../../bias_env2").resolve()),
                    PYTHONPATH=str(Path("../../").resolve())),
            timeout=300  # 5 minutes timeout
        )
        
        if result.returncode != 0:
            error_msg = f"AI analysis failed with return code {result.returncode}"
            if result.stderr:
                error_msg += f"\n\nError output:\n{result.stderr}"
            return False, error_msg
        
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        return False, "AI analysis timed out after 5 minutes"
    except Exception as e:
        return False, f"Error running AI analysis: {str(e)}"

def get_ai_response(question, quantitative_file, bias_file, ai_analysis_file=None):
    """Get an interactive AI response to a question"""
    try:
        # Import the interactive agent
        try:
            sys.path.append(str(Path("../extraction/scripts").resolve()))
            from interactive_ai_agent import InteractiveAIAgent
        except ImportError:
            return "❌ Interactive AI agent not available. Please ensure the system is properly installed."
        
        # Initialize agent
        agent = InteractiveAIAgent()
        
        # Load data
        agent.load_analysis_data(quantitative_file, bias_file, ai_analysis_file)
        
        if not agent.data_loaded:
            return "❌ Failed to load analysis data"
        
        # Get response
        response = agent.ask_question(question)
        return response
        
    except Exception as e:
        return f"❌ Error getting AI response: {str(e)}"

def display_multi_report_results():
    """Display multi-report analysis results"""
    try:
        # Look for the most recent multi-report results
        results_dir = Path("../precision_hybrid_results")
        
        # Find the most recent complete analysis file
        analysis_files = list(results_dir.glob("complete_multi_report_analysis_*.json"))
        if not analysis_files:
            st.warning("No multi-report analysis results found.")
            return
        
        # Get the most recent file
        latest_file = max(analysis_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        st.success(f"📊 Multi-Report Analysis Results (from {latest_file.name})")
        
        # Display summary statistics
        summary = results.get("summary_stats", {})
        if summary and "error" not in summary:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Reports Analyzed", summary.get("reports_analyzed", 0))
            with col2:
                st.metric("Data Points", summary.get("total_data_points", 0))
            with col3:
                st.metric("Categories", summary.get("unique_categories", 0))
            with col4:
                st.metric("Avg Confidence", f"{summary.get('average_confidence_across_all', 0):.3f}")
        
        # Create tabs for different result types
        tab1, tab2, tab3, tab4 = st.tabs(["Individual Reports", "Cross-Report Patterns", "Meta-Intelligence", "Downloads"])
        
        with tab1:
            st.markdown("### Individual Report Results")
            individual_results = results.get("individual_analyses", [])
            
            for result in individual_results:
                if "error" not in result:
                    with st.expander(f"📄 {result['report_name']}", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Data Points", result["summary"]["total_data_points"])
                        with col2:
                            st.metric("Confidence", f"{result['summary']['average_confidence']:.3f}")
                        with col3:
                            st.metric("Categories", result["summary"]["categories_found"])
        
        with tab2:
            st.markdown("### Cross-Report Pattern Analysis")
            patterns = results.get("cross_report_patterns", {})
            
            if "error" not in patterns:
                # Temporal patterns
                if "temporal_evolution" in patterns:
                    st.markdown("#### 📅 Temporal Evolution")
                    temporal = patterns["temporal_evolution"]
                    if "violation_trends" in temporal:
                        st.write("**Violation Trends:**")
                        for category, trend in temporal["violation_trends"].items():
                            direction = trend.get("trend_direction", "unknown")
                            change = trend.get("change_percent", 0)
                            st.write(f"• {category}: {direction} ({change:+.1f}%)")
                
                # Geographic patterns
                if "geographic_clustering" in patterns:
                    st.markdown("#### 🗺️ Geographic Patterns")
                    geo = patterns["geographic_clustering"]
                    if "location_hotspots" in geo:
                        st.write("**Top Incident Locations:**")
                        for location, data in list(geo["location_hotspots"].items())[:5]:
                            st.write(f"• {location}: {data.get('incident_count', 0)} incidents")
            else:
                st.warning("Pattern analysis not available")
        
        with tab3:
            st.markdown("### Meta-Intelligence Insights")
            meta = results.get("meta_intelligence", {})
            
            if "error" not in meta:
                if "meta_intelligence" in meta:
                    st.markdown(meta["meta_intelligence"])
                else:
                    st.write("Meta-intelligence analysis completed successfully.")
                    
                # Display confidence assessment
                if "confidence_assessment" in meta:
                    conf = meta["confidence_assessment"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Overall Confidence", conf.get("overall_confidence", 0))
                    with col2:
                        st.metric("Confidence Level", conf.get("confidence_level", "unknown").title())
                    with col3:
                        st.metric("Pattern Quality", conf.get("pattern_quality_score", 0))
            else:
                st.warning(f"Meta-intelligence analysis failed: {meta.get('error', 'Unknown error')}")
        
        with tab4:
            st.markdown("### Download Results")
            
            # Download complete results
            if latest_file.exists():
                st.download_button(
                    label="📥 Download Complete Analysis (JSON)",
                    data=open(latest_file, 'rb').read(),
                    file_name=latest_file.name,
                    mime="application/json",
                    key="complete_analysis_download"
                )
            
            # Download individual report results
            individual_dir = results_dir / "individual_reports"
            if individual_dir.exists():
                st.markdown("**Individual Report Downloads:**")
                for report_dir in individual_dir.iterdir():
                    if report_dir.is_dir():
                        st.write(f"📁 {report_dir.name}")
                        # List available files for download
                        for file in report_dir.glob("*.jsonl"):
                            if file.exists():
                                st.download_button(
                                    label=f"📥 {file.name}",
                                    data=open(file, 'rb').read(),
                                    file_name=f"{report_dir.name}_{file.name}",
                                    mime="application/json",
                                    key=f"download_{report_dir.name}_{file.stem}"
                                )
    
    except Exception as e:
        st.error(f"Error displaying multi-report results: {str(e)}")

def main():
    # Initialize session state for chat history
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []
    
    # Custom CSS for minimalistic, professional design
    st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background: #ffffff;
        padding: 0;
    }
    
    .stApp {
        background: #fafafa;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-weight: 500;
        color: #1f2937;
        letter-spacing: -0.025em;
    }
    
    .main-header {
        color: #111827 !important;
        font-size: 2.25rem !important;
        font-weight: 600 !important;
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .section-header {
        color: #374151 !important;
        font-size: 1.375rem !important;
        font-weight: 600 !important;
        margin-bottom: 1.5rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #e5e7eb;
    }
    
    /* Cards and Containers */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #ffffff;
        border-bottom: 1px solid #e5e7eb;
        padding: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 0;
        font-weight: 500;
        color: #6b7280;
        border-bottom: 2px solid transparent;
        transition: all 0.2s ease;
        padding: 1rem 1.5rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: transparent;
        color: #111827;
        border-bottom: 2px solid #3b82f6;
        box-shadow: none;
    }
    
    /* Metrics and Stats */
    .metric-container {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid #d1d5db;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    .stButton > button:hover {
        transform: none;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Primary Button */
    .stButton > button[data-baseweb="button"] {
        background: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }
    
    .stButton > button[data-baseweb="button"]:hover {
        background: #2563eb;
        border-color: #2563eb;
    }
    
    /* Secondary Button */
    .stButton > button:not([data-baseweb="button"]) {
        background: #ffffff;
        color: #374151;
        border: 1px solid #d1d5db;
    }
    
    .stButton > button:not([data-baseweb="button"]):hover {
        background: #f9fafb;
        border-color: #9ca3af;
    }
    
    /* Form Elements */
    .stTextInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #d1d5db;
        background: #ffffff;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Checkboxes and Sliders */
    .stCheckbox > label {
        font-weight: 500;
        color: #374151;
    }
    
    .stSlider > div > div > div > div {
        background: #3b82f6;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: #3b82f6;
        border-radius: 2px;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Success/Error Messages */
    .stAlert {
        border-radius: 6px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Chat Messages */
    .stChatMessage {
        background: #ffffff;
        border-radius: 6px;
        margin: 8px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
        font-weight: 500;
    }
    
    /* File Uploader */
    .stFileUploader > div {
        border-radius: 6px;
        border: 2px dashed #d1d5db;
        background: #ffffff;
        transition: all 0.2s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #3b82f6;
        background: #f8fafc;
    }
    
    /* Custom Spinner */
    .stSpinner > div {
        border: 2px solid #e5e7eb;
        border-top: 2px solid #3b82f6;
        border-radius: 50%;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6b7280;
        font-size: 0.875rem;
        margin-top: 3rem;
        padding: 2rem 0;
        border-top: 1px solid #e5e7eb;
        background: #ffffff;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.875rem !important;
        }
        
        .section-header {
            font-size: 1.25rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">UN Security Council Report Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6b7280; font-size: 1rem; margin-top: -1rem; font-weight: 500;">Intelligence & Bias Analysis Platform</p>', unsafe_allow_html=True)
    
    # Sidebar with clean design
    with st.sidebar:
        st.markdown("### Configuration")
        
        # Environment check
        st.markdown("### 🛠️ Environment Status")
        issues, warnings = check_environment()
        
        if issues:
            st.error("❌ Critical Issues Found:")
            for issue in issues:
                st.markdown(f"• {issue}")
            st.warning("Please fix these issues before proceeding.")
            return
        else:
            st.success("✅ Environment Ready")
        
        if warnings:
            st.warning("⚠️ Warnings:")
            for warning in warnings:
                st.markdown(f"• {warning}")
        
        # API connection test
        st.markdown("### 🔌 API Connection")
        api_success, api_message = test_api_connection()
        if api_success:
            st.success("✅ OpenAI API Connected")
        else:
            st.error(f"❌ API Issue: {api_message}")
        
        # Add a separator
        st.markdown("---")
        
        # Quick stats if results exist
        quant_file = Path("../precision_hybrid_results/precision_hybrid_extraction_results.jsonl")
        bias_file = Path("../precision_hybrid_results/text_bias_analysis_results.jsonl")
        
        if quant_file.exists() and bias_file.exists():
            st.markdown("### 📊 Quick Stats")
            try:
                # Count quantitative data points
                with open(quant_file, 'r') as f:
                    quant_count = sum(1 for line in f if line.strip())
                
                # Count bias analysis entries
                with open(bias_file, 'r') as f:
                    bias_count = sum(1 for line in f if line.strip())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📈 Data Points", quant_count)
                with col2:
                    st.metric("🎯 Bias Entries", bias_count)
                    
            except Exception:
                pass
    
    # Analysis mode selection
    st.markdown('<h2 class="section-header">Analysis Mode</h2>', unsafe_allow_html=True)
    
    analysis_mode = st.radio(
        "Choose your analysis approach:",
        ["Single Report Analysis", "Multi-Report Intelligence"],
        help="Single report for detailed analysis, Multi-report for cross-report pattern analysis"
    )
    
    if analysis_mode == "Single Report Analysis":
        # Main content with clean design
        st.markdown('<h3 class="section-header">Upload UN Report PDF</h3>', unsafe_allow_html=True)
        
        # Clean file uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a UN Security Council report PDF for analysis"
        )
    
    else:  # Multi-Report Intelligence
        # Multi-report upload section
        st.markdown('<h3 class="section-header">Upload Multiple UN Reports</h3>', unsafe_allow_html=True)
        
        st.markdown("""
        **Multi-Report Analysis** : Upload 2-20 reports to unlock:
        - **Cross-report pattern detection** - Find bias evolution over time
        - **Predictive intelligence** - Forecast future bias patterns  
        - **Hidden correlations** - Discover non-obvious relationships
        - **Institutional bias fingerprinting** - Detect systematic patterns
        """)
        
        uploaded_files = st.file_uploader(
            "Choose multiple PDF files (2-20 reports recommended)",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload multiple UN Security Council reports for cross-report intelligence analysis"
        )
        
        if uploaded_files:
            if len(uploaded_files) < 2:
                st.warning("⚠️ Multi-report analysis requires at least 2 reports. Please upload more files or switch to single report mode.")
                uploaded_file = None
            elif len(uploaded_files) > 20:
                st.warning("⚠️ For optimal performance, please limit to 20 reports maximum.")
                uploaded_file = None
            else:
                st.success(f"✅ {len(uploaded_files)} reports ready for multi-report analysis!")
                
                # Display file summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Reports", len(uploaded_files))
                with col2:
                    total_size = sum(f.size for f in uploaded_files) / (1024 * 1024)
                    st.metric("Total Size", f"{total_size:.1f} MB")
                with col3:
                    estimated_time = len(uploaded_files) * 4  # Rough estimate: 4 min per report
                    st.metric("Est. Time", f"{estimated_time} min")
                
                # Show file list
                with st.expander("📋 Uploaded Files", expanded=False):
                    for i, file in enumerate(uploaded_files, 1):
                        st.write(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
                
                uploaded_file = uploaded_files  # Set for processing
        else:
            uploaded_file = None
    
    # Initialize result variables (default to False when no files uploaded)
    results_exist = False
    ai_results_exist = False
    
    # Show analysis options for both modes when files are uploaded
    if uploaded_file is not None:
        # Initialize result variables for both modes
        quant_file = Path("../precision_hybrid_results/precision_hybrid_extraction_results.jsonl")
        bias_file = Path("../precision_hybrid_results/text_bias_analysis_results.jsonl")
        ai_file = Path("../precision_hybrid_results/ai_analysis_report.json")
        
        results_exist = quant_file.exists() and bias_file.exists()
        ai_results_exist = ai_file.exists() if results_exist else False
        
        # Single report file info (only for single report mode)
        if analysis_mode == "Single Report Analysis":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", uploaded_file.type)
            
            if results_exist:
                st.success("✅ Analysis results found! You can view them in the tabs below or run a new analysis.")
        
        # Analysis options with clean design
        st.markdown('<h2 class="section-header">Analysis Options</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            run_quantitative = st.checkbox("Quantitative Extraction", value=True, help="Extract structured quantitative data using GPT-4o")
        with col2:
            run_bias = st.checkbox("Bias Analysis", value=True, help="Analyze bias using Entman's framing theory")
        with col3:
            enable_ai_analysis = st.checkbox("AI Analysis", value=False, help="Generate intelligent insights using AI agent (requires both quantitative and bias analysis)")
        
        # Performance options with clean design
        st.markdown('<h3 class="section-header">Performance Options</h3>', unsafe_allow_html=True)
        
        # Performance optimization info
        with st.expander("Performance Optimization Features", expanded=False):
            st.markdown("""
            **Recent Optimizations:**
            - **Smart Batching**: Process multiple paragraphs per API call (3x faster)
            - **Parallel Processing**: Up to 4 concurrent workers (conservative for stability)
            - **Qualitative Value Cleaning**: Automatic conversion of 'daily', 'large-scale' to 'N/A'
            - **Rate Limit Management**: Automatic retry and backoff with 30-second delays
            - **Memory Optimization**: Efficient processing of large files
            
            **Performance Impact:**
            - **Small reports**: 3-8 minutes (was 10-15 minutes)
            - **Large reports**: 8-20 minutes (was 30+ minutes)
            - **Cost reduction**: 60-90% less expensive
            - **Stability**: Reduced rate limit errors
            """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            batch_size = st.slider("Paragraphs per API Call", min_value=1, max_value=5, value=3, 
                                 help="Number of paragraphs per API call (higher = faster, lower = more precise)")
        with col2:
            max_workers = st.slider("Parallel Workers", min_value=1, max_value=4, value=3,
                                  help="Number of parallel API calls (higher = faster but may hit rate limits)")
        with col3:
            parallel_processing = st.checkbox("Enable Parallel Processing", value=True, 
                                            help="Run quantitative and bias analysis simultaneously")
        
        # Show estimated time with clean design
        if run_quantitative and analysis_mode == "📄 Single Report Analysis":
            file_size_mb = uploaded_file.size / (1024 * 1024)
            
            # Try to get actual paragraph count from hybrid extraction if available
            jsonl_file_path = Path("../extraction/JSONL_outputs") / f"{uploaded_file.name.replace('.pdf', '')}-structured.jsonl"
            actual_paragraphs = get_actual_paragraph_count(jsonl_file_path)
            
            estimated_time, estimated_paragraphs = estimate_processing_time(file_size_mb, batch_size, max_workers, actual_paragraphs)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated Time", f"{estimated_time:.0f} minutes")
            with col2:
                if actual_paragraphs is not None:
                    st.metric("Actual Paragraphs", f"{actual_paragraphs}")
                else:
                    st.metric("Estimated Paragraphs", f"{estimated_paragraphs}")
            
            st.info(f"File size: {file_size_mb:.1f} MB")
        elif run_quantitative and analysis_mode == "Multi-Report Intelligence":
            # Show multi-report estimation (already displayed above in the upload section)
            st.info("💡 Processing time and resource estimates are shown above in the file summary.")
        
        # Run analysis button - different for single vs multi-report
        if analysis_mode == "Single Report Analysis":
            button_text = "Start Analysis"
            button_help = "Run analysis on the uploaded report"
        else:
            button_text = "Start Multi-Report Analysis"
            button_help = "Run cross-report analysis with pattern detection and meta-intelligence"
        
        if st.button(button_text, type="primary", use_container_width=True, help=button_help):
            if not (run_quantitative or run_bias):
                st.error("Please select at least one analysis type.")
                return
            
            # Handle multi-report vs single-report processing
            if analysis_mode == "Multi-Report Intelligence":
                # Multi-report processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Progress tracking for multi-report
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        status_text.text("Starting Multi-Report Intelligence Analysis...")
                        progress_bar.progress(10)
                        
                        # Run multi-report analysis
                        success, output = run_multi_report_analysis(uploaded_file, temp_path)
                        
                        if success:
                            progress_bar.progress(100)
                            status_text.text("✅ Multi-Report Intelligence Analysis Complete!")
                            st.success(output)
                            st.balloons()
                            
                            # Display multi-report results
                            display_multi_report_results()
                        else:
                            st.error(f"Multi-report analysis failed: {output}")
                            return
                            
                    except Exception as e:
                        st.error(f"Multi-report analysis error: {str(e)}")
                        return
            
            else:
                # Single-report processing (existing logic)
                # Create temporary directory for processing
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Run analyses
                    if run_quantitative and run_bias and parallel_processing:
                        status_text.text("🔄 Running quantitative extraction and bias analysis in parallel...")
                        
                        # Create status containers for each process
                        quant_status = st.empty()
                        bias_status = st.empty()
                        
                        quant_status.info("📊 Quantitative extraction: Starting...")
                        bias_status.info("🎯 Bias analysis: Waiting for hybrid extraction...")
                        
                        # Run quantitative extraction with better progress tracking
                        progress_bar.progress(10)
                        quant_status.info("📊 Quantitative extraction: Starting hybrid extraction...")
                        
                        progress_bar.progress(20)
                        quant_status.info("📊 Quantitative extraction: Running GPT-4o extraction...")
                        
                        quant_success, quant_output = run_quantitative_extraction(uploaded_file, temp_path, max_workers, batch_size)
                        
                        if not quant_success:
                            st.error(f"Quantitative extraction failed: {quant_output}")
                            return
                        
                        progress_bar.progress(60)
                        quant_status.success("📊 Quantitative extraction: Completed")
                        
                        # Run bias analysis with better progress tracking
                        progress_bar.progress(65)
                        bias_status.info("🎯 Bias analysis: Starting bias analysis...")
                        
                        jsonl_file = Path("../extraction/JSONL_outputs") / f"{uploaded_file.name.replace('.pdf', '')}-structured.jsonl"
                        
                        if not jsonl_file.exists():
                            st.error(f"JSONL file not found: {jsonl_file}")
                            return
                        
                        bias_success, bias_output = run_bias_analysis(jsonl_file, Path("../precision_hybrid_results"))
                        
                        if not bias_success:
                            st.error(f"Bias analysis failed: {bias_output}")
                            return
                        
                        progress_bar.progress(85)
                        bias_status.success("🎯 Bias analysis: Completed")
                        status_text.text("✅ Both analyses completed")
                    
                    else:
                        # Run single analysis
                        if run_quantitative:
                            status_text.text("🔄 Running quantitative extraction...")
                            progress_bar.progress(20)
                            
                            success, output = run_quantitative_extraction(uploaded_file, temp_path, max_workers, batch_size)
                            
                            if not success:
                                st.error(f"Quantitative extraction failed: {output}")
                                return
                            
                            progress_bar.progress(80)
                            status_text.text("✅ Quantitative extraction completed")
                        
                        if run_bias:
                            status_text.text("🔄 Running bias analysis...")
                            progress_bar.progress(60)
                            
                            # Find the JSONL file to analyze
                            jsonl_file = Path("../extraction/JSONL_outputs") / f"{uploaded_file.name.replace('.pdf', '')}-structured.jsonl"
                            
                            if not jsonl_file.exists():
                                st.error(f"JSONL file not found: {jsonl_file}")
                                return
                            
                            success, output = run_bias_analysis(jsonl_file, Path("../precision_hybrid_results"))
                            
                            if not success:
                                st.error(f"Bias analysis failed: {output}")
                                return
                            
                            progress_bar.progress(100)
                            status_text.text("✅ Bias analysis completed")
                    
                    # Run AI analysis if requested
                    if enable_ai_analysis and run_quantitative and run_bias:
                        status_text.text("🤖 Generating AI analysis...")
                        progress_bar.progress(90)
                        
                        # Find the result files
                        quant_file = Path("../precision_hybrid_results/precision_hybrid_extraction_results.jsonl")
                        bias_file = Path("../precision_hybrid_results/text_bias_analysis_results.jsonl")
                        
                        if not quant_file.exists():
                            st.error(f"Quantitative results file not found: {quant_file}")
                            return
                        
                        if not bias_file.exists():
                            st.error(f"Bias analysis results file not found: {bias_file}")
                            return
                        
                        # Run AI analysis
                        ai_success, ai_output = run_ai_analysis(quant_file, bias_file, Path("../precision_hybrid_results"))
                        
                        if not ai_success:
                            st.error(f"AI analysis failed: {ai_output}")
                            return
                        
                        progress_bar.progress(100)
                        status_text.text("✅ AI analysis completed")
                    
                    # Analysis complete
                    st.success("🎉 Analysis completed successfully!")
                    
                    # Display results
                    st.markdown('<h2 class="section-header">📈 Analysis Results</h2>', unsafe_allow_html=True)
                    
                    # Results tabs
                    tab1, tab2, tab3, tab4 = st.tabs(["Quantitative Results", "Bias Analysis", "AI Analysis", "Downloads"])
                    
                    with tab1:
                        if run_quantitative:
                            quant_file = Path("../precision_hybrid_results/precision_hybrid_extraction_results.jsonl")
                            if quant_file.exists():
                                # Convert to CSV
                                csv_file = temp_path / "quantitative_results.csv"
                                success, result = jsonl_to_csv(quant_file, csv_file)
                                
                                if success:
                                    st.success(f"✅ Extracted {result} quantitative data points")
                                    
                                    # Show sample data
                                    df = pd.read_csv(csv_file)
                                    st.subheader("Sample Data")
                                    st.dataframe(df.head(10), use_container_width=True)
                                    
                                    # Show statistics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Records", len(df))
                                    with col2:
                                        st.metric("Categories", df['category'].nunique())
                                    with col3:
                                        st.metric("Actors", df['actor'].nunique())
                                    with col4:
                                        st.metric("Avg Confidence", f"{df['confidence_score'].mean():.2f}")
                                    
                                    # Show legal grounding if available
                                    if 'legal_grounding_summary' in df.columns:
                                        st.subheader("⚖️ Legal Grounding")
                                        legal_summaries = df['legal_grounding_summary'].dropna().unique()
                                        for summary in legal_summaries[:5]:  # Show first 5
                                            st.info(summary)
                                else:
                                    st.error(f"Error processing quantitative results: {result}")
                            else:
                                st.warning("Quantitative results file not found")
                        else:
                            st.info("Quantitative extraction was not selected")
                    
                    with tab2:
                        if run_bias:
                            bias_file = Path("../precision_hybrid_results/text_bias_analysis_results.jsonl")
                            if bias_file.exists():
                                # Convert to CSV
                                csv_file = temp_path / "bias_analysis.csv"
                                success, result = jsonl_to_csv(bias_file, csv_file)
                                
                                if success:
                                    st.success(f"✅ Analyzed {result} paragraphs for bias")
                                    
                                    # Show sample data
                                    df = pd.read_csv(csv_file)
                                    st.subheader("Sample Bias Analysis")
                                    st.dataframe(df.head(10), use_container_width=True)
                                    
                                    # Show bias statistics
                                    if 'bias_flag' in df.columns:
                                        bias_counts = df['bias_flag'].value_counts()
                                        st.subheader("Bias Distribution")
                                        st.bar_chart(bias_counts)
                                else:
                                    st.error(f"Error processing bias results: {result}")
                            else:
                                st.warning("Bias analysis results file not found")
                        else:
                            st.info("Bias analysis was not selected")
                    
                    with tab3:
                        if enable_ai_analysis and run_quantitative and run_bias:
                            ai_file = Path("../precision_hybrid_results/ai_analysis_report.json")
                            quant_file = Path("../precision_hybrid_results/precision_hybrid_extraction_results.jsonl")
                            bias_file = Path("../precision_hybrid_results/text_bias_analysis_results.jsonl")
                            
                            if ai_file.exists() and quant_file.exists() and bias_file.exists():
                                try:
                                    with open(ai_file, 'r', encoding='utf-8') as f:
                                        ai_data = json.load(f)
                                    
                                    if "error" in ai_data:
                                        st.error(f"AI Analysis Error: {ai_data['error']}")
                                    else:
                                        st.success("✅ AI Analysis Generated Successfully")
                                        
                                        # Display the analysis
                                        st.subheader("🤖 AI Analysis Report")
                                        st.markdown(ai_data.get('analysis', 'No analysis content available'))
                                        
                                        # Show data summary
                                        if 'data_summary' in ai_data:
                                            st.subheader("📊 Analysis Summary")
                                            summary = ai_data['data_summary']
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Quantitative Points", summary.get('quantitative_points', 0))
                                            with col2:
                                                st.metric("Bias Paragraphs", summary.get('bias_paragraphs', 0))
                                            with col3:
                                                st.metric("Categories Found", summary.get('categories_found', 0))
                                            with col4:
                                                st.metric("Actors Identified", summary.get('actors_identified', 0))
                                        
                                        # Show timestamp
                                        if 'timestamp' in ai_data:
                                            st.caption(f"Analysis generated: {ai_data['timestamp']}")
                                        
                                        # Interactive AI Chat with NO RELOAD using st.form
                                        st.markdown("---")
                                        st.subheader("💬 Ask the AI Agent")
                                        st.markdown("Ask questions about the analysis results and get intelligent, data-driven answers!")
                                        
                                        # Display chat history
                                        for message in st.session_state.ai_chat_history:
                                            with st.chat_message(message["role"]):
                                                st.markdown(message["content"])
                                        
                                        # Use st.form to prevent page reload
                                        with st.form("ai_chat_form", clear_on_submit=True):
                                            user_question = st.text_input("Ask a question about the analysis...", key="chat_input")
                                            submit_button = st.form_submit_button("🤖 Ask AI", type="primary")
                                            
                                            if submit_button and user_question:
                                                # Add user message to chat history
                                                st.session_state.ai_chat_history.append({"role": "user", "content": user_question})
                                                
                                                # Get AI response
                                                with st.spinner("🤖 AI Agent is thinking..."):
                                                    response = get_ai_response(user_question, quant_file, bias_file, ai_file)
                                                
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
                                
                                except Exception as e:
                                    st.error(f"Error loading AI analysis: {str(e)}")
                            else:
                                st.warning("AI analysis files not found. Please run the analysis first.")
                        else:
                            st.info("AI analysis requires both quantitative extraction and bias analysis to be selected.")
                    
                    with tab4:
                        st.subheader("📥 Download Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if run_quantitative:
                                quant_csv = temp_path / "quantitative_results.csv"
                                if quant_csv.exists():
                                    create_download_link(quant_csv, "📊 Download Quantitative Results (CSV)", key="quant_download_1")
                        
                        with col2:
                            if run_bias:
                                bias_csv = temp_path / "bias_analysis.csv"
                                if bias_csv.exists():
                                    create_download_link(bias_csv, "🎯 Download Bias Analysis (CSV)", key="bias_download_1")
                        
                        with col3:
                            if enable_ai_analysis and run_quantitative and run_bias:
                                ai_file = Path("../precision_hybrid_results/ai_analysis_report.json")
                                if ai_file.exists():
                                    create_download_link(ai_file, "🤖 Download AI Analysis (JSON)", key="ai_download_1")
                

        # Display results if they exist (either from current run or previous runs)
        if results_exist:
            st.markdown('<h2 class="section-header">📈 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Results tabs
            tab1, tab2, tab3, tab4 = st.tabs(["Quantitative Results", "Bias Analysis", "AI Analysis", "Downloads"])
            
            # Create a temporary path for CSV files
            temp_path = Path("/tmp")
            
            with tab1:
                # Convert to CSV
                csv_file = temp_path / "quantitative_results.csv"
                success, result = jsonl_to_csv(quant_file, csv_file)
                
                if success:
                    st.success(f"✅ Extracted {result} quantitative data points")
                    
                    # Show sample data
                    df = pd.read_csv(csv_file)
                    st.subheader("Sample Data")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Categories", df['category'].nunique())
                    with col3:
                        st.metric("Actors", df['actor'].nunique())
                    with col4:
                        st.metric("Avg Confidence", f"{df['confidence_score'].mean():.2f}")
                    
                    # Show legal grounding if available
                    if 'legal_grounding_summary' in df.columns:
                        st.subheader("⚖️ Legal Grounding")
                        legal_summaries = df['legal_grounding_summary'].dropna().unique()
                        for summary in legal_summaries[:5]:  # Show first 5
                            st.info(summary)
                else:
                    st.error(f"Error processing quantitative results: {result}")
            
            with tab2:
                # Convert to CSV
                csv_file = temp_path / "bias_analysis.csv"
                success, result = jsonl_to_csv(bias_file, csv_file)
                
                if success:
                    st.success(f"✅ Analyzed {result} paragraphs for bias")
                    
                    # Show sample data
                    df = pd.read_csv(csv_file)
                    st.subheader("Sample Bias Analysis")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show bias statistics
                    if 'bias_flag' in df.columns:
                        bias_counts = df['bias_flag'].value_counts()
                        st.subheader("Bias Distribution")
                        st.bar_chart(bias_counts)
                else:
                    st.error(f"Error processing bias results: {result}")
            
            with tab3:
                if ai_results_exist:
                    try:
                        with open(ai_file, 'r', encoding='utf-8') as f:
                            ai_data = json.load(f)
                        
                        if "error" in ai_data:
                            st.error(f"AI Analysis Error: {ai_data['error']}")
                        else:
                            st.success("✅ AI Analysis Generated Successfully")
                            
                            # Display the analysis
                            st.subheader("🤖 AI Analysis Report")
                            st.markdown(ai_data.get('analysis', 'No analysis content available'))
                            
                            # Show data summary
                            if 'data_summary' in ai_data:
                                st.subheader("📊 Analysis Summary")
                                summary = ai_data['data_summary']
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Quantitative Points", summary.get('quantitative_points', 0))
                                with col2:
                                    st.metric("Bias Paragraphs", summary.get('bias_paragraphs', 0))
                                with col3:
                                    st.metric("Categories Found", summary.get('categories_found', 0))
                                with col4:
                                    st.metric("Actors Identified", summary.get('actors_identified', 0))
                            
                            # Show timestamp
                            if 'timestamp' in ai_data:
                                st.caption(f"Analysis generated: {ai_data['timestamp']}")
                            
                            # Interactive AI Chat with NO RELOAD using st.form
                            st.markdown("---")
                            st.subheader("💬 Ask the AI Agent")
                            st.markdown("Ask questions about the analysis results and get intelligent, data-driven answers!")
                            
                            # Display chat history
                            for message in st.session_state.ai_chat_history:
                                with st.chat_message(message["role"]):
                                    st.markdown(message["content"])
                            
                            # Use st.form to prevent page reload
                            with st.form("ai_chat_form", clear_on_submit=True):
                                user_question = st.text_input("Ask a question about the analysis...", key="chat_input")
                                submit_button = st.form_submit_button("🤖 Ask AI", type="primary")
                                
                                if submit_button and user_question:
                                    # Add user message to chat history
                                    st.session_state.ai_chat_history.append({"role": "user", "content": user_question})
                                    
                                    # Get AI response
                                    with st.spinner("🤖 AI Agent is thinking..."):
                                        response = get_ai_response(user_question, quant_file, bias_file, ai_file)
                                    
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
                        
                    except Exception as e:
                        st.error(f"Error loading AI analysis: {str(e)}")
                else:
                    st.info("AI analysis not found. Enable 'Generate AI Analysis' and run the analysis to get AI insights.")
            
            with tab4:
                st.subheader("📥 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    quant_csv = temp_path / "quantitative_results.csv"
                    if quant_csv.exists():
                        create_download_link(quant_csv, "📊 Download Quantitative Results (CSV)", key="quant_download_2")
                
                with col2:
                    bias_csv = temp_path / "bias_analysis.csv"
                    if bias_csv.exists():
                        create_download_link(bias_csv, "🎯 Download Bias Analysis (CSV)", key="bias_download_2")
                
                with col3:
                    if ai_results_exist:
                        create_download_link(ai_file, "🤖 Download AI Analysis (JSON)", key="ai_download_2")
    
    # Instructions
    with st.expander("📖 How to Use"):
        st.markdown("""
        ### Step-by-Step Instructions:
        
        1. **Upload PDF**: Select a UN Security Council report PDF file
        2. **Choose Analysis**: Select which types of analysis to run
        3. **Configure Performance**: Adjust batch size and workers for optimal performance
        4. **Start Analysis**: Click the "Start Analysis" button
        5. **Review Results**: Check the results in the tabs
        6. **Download**: Download the results as CSV files
        
        ### What You'll Get:
        
        **Quantitative Results CSV:**
        - Structured data points extracted from the report
        - Categories: fatalities, displacement, missile launches, etc.
        - Actors: IDF, Hizbullah, UNIFIL, etc.
        - Values, confidence scores, and legal violations
        
        **Bias Analysis CSV:**
        - Bias detection using Entman's framing theory
        - Framing bias, selection bias, omission bias
        - Actor coverage analysis
        - Language analysis and bias indicators
        
        ### Performance Tips:
        - **Optimized defaults**: Batch size 5, workers 6
        - **Small reports**: 2-5 minutes (was 10-15 minutes)
        - **Large reports**: 5-15 minutes (was 30+ minutes)
        - **Cost reduction**: 60-90% less expensive with batching
        - **Smart batching**: Automatically processes multiple paragraphs per API call
        - **Rate limit management**: Automatic retry and backoff
        
        ### Requirements:
        - OpenAI API key in `.env` file
        - Virtual environment `bias_env2` activated
        - All required Python packages installed
        """)
    
    # Clean footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>UN Report Analysis Pipeline | Created by Moumen Alaoui | <a href='mailto:moumenalaoui@proton.me'>moumenalaoui@proton.me</a></div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 