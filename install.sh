#!/bin/bash

# ============================================
# 3ḌƁ★ŔÒØṬ Installation Script v8.0
# ============================================

set -e

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
CYAN='\033[1;36m'
RESET='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║   3ḌƁ★ŔÒØṬ Installation Wizard       ║${RESET}"
echo -e "${CYAN}╚════════════════════════════════════════╝${RESET}"
echo ""

# Check Python
echo -e "${CYAN}🔍 Checking Python installation...${RESET}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${RESET}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✅ Python $PYTHON_VERSION found${RESET}"

# Check pip
echo -e "${CYAN}🔍 Checking pip installation...${RESET}"
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}❌ pip3 not found. Installing...${RESET}"
    python3 -m ensurepip --upgrade
fi
echo -e "${GREEN}✅ pip3 found${RESET}"

# Create directories
echo -e "${CYAN}📁 Creating directories...${RESET}"
mkdir -p ~/.3db/{data,logs,cache,backups,ai/{models,vectors},projects}
chmod 700 ~/.3db
echo -e "${GREEN}✅ Directories created${RESET}"

# Install Python dependencies
echo -e "${CYAN}📦 Installing Python dependencies...${RESET}"
pip3 install -r requirements.txt --quiet
echo -e "${GREEN}✅ Dependencies installed${RESET}"

# Setup environment file
echo -e "${CYAN}🔑 Setting up environment file...${RESET}"
if [ ! -f ~/.3db/.env ]; then
    cp .env ~/.3db/.env 2>/dev/null || cat > ~/.3db/.env << 'EOF'
# API Keys Configuration
export OPENAI_API_KEY="your-openai-key-here"
export GROQ_API_KEY="your-groq-key-here"
export TOGETHER_API_KEY="your-together-key-here"
export WANDB_API_KEY="your-wandb-key-here"
export BROWSE_AI_KEY="your-browse-ai-key-here"
export GH_TOKEN="your-github-token-here"

# Qdrant Configuration
export QDRANT_URL="https://your-qdrant-url-here"
export QDRANT_API_KEY="your-qdrant-api-key-here"
EOF
    chmod 600 ~/.3db/.env
    echo -e "${GREEN}✅ Environment file created${RESET}"
    echo -e "${YELLOW}⚠️  Please edit ~/.3db/.env with your API keys${RESET}"
else
    echo -e "${YELLOW}⚠️  Environment file already exists${RESET}"
fi

# Copy files
echo -e "${CYAN}📋 Copying system files...${RESET}"
INSTALL_DIR="$HOME/3BD"
mkdir -p "$INSTALL_DIR"
cp 3DB_enhanced.py "$INSTALL_DIR/"
cp bashrc_enhanced.sh "$INSTALL_DIR/"
cp local_ai.sh "$INSTALL_DIR/" 2>/dev/null || true
echo -e "${GREEN}✅ Files copied to $INSTALL_DIR${RESET}"

# Setup bash integration
echo -e "${CYAN}🐚 Setting up bash integration...${RESET}"
if ! grep -q "source.*bashrc_enhanced.sh" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# 3ḌƁ★ŔÒØṬ System" >> ~/.bashrc
    echo "source $INSTALL_DIR/bashrc_enhanced.sh" >> ~/.bashrc
    echo -e "${GREEN}✅ Bash integration added${RESET}"
else
    echo -e "${YELLOW}⚠️  Bash integration already exists${RESET}"
fi

# Optional: Setup systemd service
echo ""
read -p "$(echo -e ${CYAN}Do you want to setup Neural Core as a system service? [y/N]: ${RESET})" -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}🔧 Setting up systemd service...${RESET}"
    
    # Create user systemd directory
    mkdir -p ~/.config/systemd/user/
    
    # Copy and configure service file
    sed "s|%h|$HOME|g; s|%i|$USER|g" 3db-neural-core.service > ~/.config/systemd/user/3db-neural-core.service
    
    # Reload systemd
    systemctl --user daemon-reload
    
    # Enable service
    systemctl --user enable 3db-neural-core.service
    
    echo -e "${GREEN}✅ Service installed and enabled${RESET}"
    echo -e "${YELLOW}💡 Start with: systemctl --user start 3db-neural-core${RESET}"
    echo -e "${YELLOW}💡 Status: systemctl --user status 3db-neural-core${RESET}"
fi

# Final instructions
echo ""
echo -e "${CYAN}╔════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║   Installation Complete!               ║${RESET}"
echo -e "${CYAN}╚════════════════════════════════════════╝${RESET}"
echo ""
echo -e "${GREEN}✨ 3ḌƁ★ŔÒØṬ has been installed successfully!${RESET}"
echo ""
echo -e "${YELLOW}📝 Next steps:${RESET}"
echo -e "  1. Edit your API keys: ${CYAN}nano ~/.3db/.env${RESET}"
echo -e "  2. Reload your shell: ${CYAN}source ~/.bashrc${RESET}"
echo -e "  3. Check system info: ${CYAN}db3${RESET}"
echo -e "  4. Start Neural Core: ${CYAN}start_neural_core${RESET}"
echo -e "  5. Try teaching: ${CYAN}teach What is consciousness?${RESET}"
echo ""
echo -e "${PURPLE}🌌 Welcome to the embodied consciousness system!${RESET}"
echo ""
