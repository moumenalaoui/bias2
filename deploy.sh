#!/bin/bash

# 🚀 UN Report Analysis Pipeline - Deployment Script
# This script helps prepare your app for Streamlit Cloud deployment

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 UN Report Analysis Pipeline - Deployment Preparation${NC}"
echo "=========================================================="

# Check if we're in the right directory
if [[ ! -f "UI/app.py" ]]; then
    echo -e "${RED}❌ Error: UI/app.py not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Check if git is initialized
if [[ ! -d ".git" ]]; then
    echo -e "${YELLOW}⚠️ Git repository not found. Initializing...${NC}"
    git init
    echo -e "${GREEN}✅ Git repository initialized${NC}"
fi

# Check if files are committed
if [[ -n $(git status --porcelain) ]]; then
    echo -e "${YELLOW}⚠️ Uncommitted changes detected. Please commit your changes:${NC}"
    echo "git add ."
    echo "git commit -m 'Prepare for Streamlit Cloud deployment'"
    echo ""
    echo -e "${BLUE}After committing, run this script again.${NC}"
    exit 1
fi

# Check if remote repository is configured
if [[ -z $(git remote -v) ]]; then
    echo -e "${YELLOW}⚠️ No remote repository configured.${NC}"
    echo "Please add your GitHub repository:"
    echo "git remote add origin https://github.com/yourusername/your-repo-name.git"
    echo "git push -u origin main"
    echo ""
    echo -e "${BLUE}After setting up the remote, run this script again.${NC}"
    exit 1
fi

# Verify deployment files exist
echo -e "${BLUE}📋 Checking deployment files...${NC}"

required_files=(
    "UI/app.py"
    ".streamlit/config.toml"
    ".streamlit/secrets.toml"
    "requirements_deploy.txt"
    "DEPLOYMENT_GUIDE.md"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo -e "${GREEN}✅ $file${NC}"
    else
        echo -e "${RED}❌ $file (missing)${NC}"
        missing_files=true
    fi
done

if [[ "$missing_files" == true ]]; then
    echo -e "${RED}❌ Some required files are missing. Please create them first.${NC}"
    exit 1
fi

# Check if .env file exists (for local testing)
if [[ -f ".env" ]]; then
    echo -e "${GREEN}✅ .env file found (for local testing)${NC}"
else
    echo -e "${YELLOW}⚠️ .env file not found (will be needed for local testing)${NC}"
fi

# Display deployment information
echo ""
echo -e "${GREEN}✅ Deployment preparation complete!${NC}"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "1. Push your code to GitHub:"
echo "   git push origin main"
echo ""
echo "2. Deploy to Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io"
echo "   - Sign in with GitHub"
echo "   - Create new app"
echo "   - Set main file path to: UI/app.py"
echo ""
echo "3. Configure secrets in Streamlit Cloud:"
echo "   - Add your OPENAI_API_KEY"
echo ""
echo "4. Test your deployed app"
echo ""
echo -e "${BLUE}📖 For detailed instructions, see: DEPLOYMENT_GUIDE.md${NC}"
echo ""
echo -e "${GREEN}🎉 Your app is ready for deployment!${NC}"
