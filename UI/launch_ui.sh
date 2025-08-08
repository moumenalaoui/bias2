#!/bin/bash

# 🚀 UN Report Analysis Pipeline - UI Launch Script
# Ensures proper environment setup and launches the Streamlit UI

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Launching UN Report Analysis Pipeline UI${NC}"
echo "=========================================="

# Check if we're in the right directory
if [[ ! -f "app.py" ]]; then
    echo -e "${RED}❌ Error: app.py not found. Please run this script from the UI directory.${NC}"
    exit 1
fi

# Check if virtual environment exists
if [[ ! -d "../bias_env2" ]]; then
    echo -e "${RED}❌ Error: Virtual environment 'bias_env2' not found.${NC}"
    echo "Please ensure the virtual environment is set up correctly."
    exit 1
fi

# Activate virtual environment
echo -e "${BLUE}🔧 Activating virtual environment...${NC}"
source ../bias_env2/bin/activate

# Check if required packages are installed
echo -e "${BLUE}📦 Checking required packages...${NC}"
python -c "import streamlit, pandas, openai; print('✅ All required packages found')" 2>/dev/null || {
    echo -e "${YELLOW}⚠️ Installing missing packages...${NC}"
    pip install -r requirements.txt
    pip install -r ../extraction/scripts/requirements_gpt.txt
}

# Check if .env file exists
if [[ ! -f "../.env" ]]; then
    echo -e "${RED}❌ Error: .env file not found. Please ensure your OpenAI API key is configured.${NC}"
    exit 1
fi

# Test API connection
echo -e "${BLUE}🔗 Testing API connection...${NC}"
python -c "
import os
from dotenv import load_dotenv
import openai

load_dotenv('../.env')
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise Exception('No API key found')

client = openai.OpenAI(api_key=api_key)
response = client.chat.completions.create(
    model='gpt-4o',
    messages=[{'role': 'user', 'content': 'Test'}],
    max_tokens=5
)
print('✅ API connection successful')
" 2>/dev/null || {
    echo -e "${RED}❌ API connection failed. Please check your OpenAI API key.${NC}"
    exit 1
}

# Create required directories if they don't exist
echo -e "${BLUE}📁 Creating required directories...${NC}"
mkdir -p ../extraction/reports_pdf
mkdir -p ../extraction/hybrid_output
mkdir -p ../extraction/JSONL_outputs
mkdir -p ../precision_hybrid_results

echo -e "${GREEN}✅ Environment ready!${NC}"
echo ""
echo -e "${BLUE}🌐 Starting Streamlit UI...${NC}"
echo "The UI will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit with better output handling
exec streamlit run app.py --server.port 8501 --server.address localhost --server.headless true 