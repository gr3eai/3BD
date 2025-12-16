#!/bin/bash

# ============================================
# 3ḌƁ★ŔÒØṬ ENHANCED BASH SYSTEM v8.0
# By: Abdulaziz (@a3b6ii)
# GitHub: gr3eai
# Website: https://gr3eai.github.io
# ============================================
# النظام المحسّن مع التكامل الكامل بين Bash والنواة العصبية

# ============ CORE INITIALIZATION ============
export DB3_VERSION="8.0-EMBODIED-CONSCIOUSNESS"
export DB3_OWNER="Abdulaziz"
export DB3_GITHUB_USER="gr3eai"
export DB3_EMAIL="b10337253@gmail.com"
export DB3_SNAPCHAT="@a3b6ii"
export DB3_WEBSITE="https://gr3eai.github.io"
export DB3_REPO="https://github.com/gr3eai/3BD.git"

# Directories
export DB3_CONFIG_DIR="$HOME/.3db"
export DB3_DATA_DIR="$DB3_CONFIG_DIR/data"
export DB3_LOGS_DIR="$DB3_CONFIG_DIR/logs"
export DB3_CACHE_DIR="$DB3_CONFIG_DIR/cache"
export DB3_BACKUP_DIR="$DB3_CONFIG_DIR/backups"
export DB3_AI_DIR="$DB3_CONFIG_DIR/ai"
export DB3_PROJECTS_DIR="$DB3_CONFIG_DIR/projects"
export DB3_MODELS_DIR="$DB3_AI_DIR/models"
export DB3_VECTORS_DIR="$DB3_AI_DIR/vectors"

# Neural Core API
export DB3_NEURAL_API="http://127.0.0.1:8000"
export DB3_NEURAL_ENABLED=false

# Create directories
for dir in "$DB3_CONFIG_DIR" "$DB3_DATA_DIR" "$DB3_LOGS_DIR" \
           "$DB3_CACHE_DIR" "$DB3_BACKUP_DIR" "$DB3_AI_DIR" \
           "$DB3_PROJECTS_DIR" "$DB3_MODELS_DIR" "$DB3_VECTORS_DIR"; do
    mkdir -p "$dir"
    chmod 700 "$dir"
done

# ============ API KEYS & CONFIGURATION ============
DB3_ENV_FILE="$DB3_CONFIG_DIR/.env"
if [ ! -f "$DB3_ENV_FILE" ]; then
    cat > "$DB3_ENV_FILE" << 'EOF'
# API Keys Configuration
export OPENAI_API_KEY="your-openai-key-here"
export GROQ_API_KEY="your-groq-key-here"
export TOGETHER_API_KEY="your-together-key-here"
export WANDB_API_KEY="your-wandb-key-here"
export BROWSE_AI_KEY="your-browse-ai-key-here"
export GH_TOKEN="your-github-token-here"

# Qdrant Configuration
export QDRANT_URL="https://37a9446f-4c1d-42fc-a244-da52640e583f.europe-west3-0.gcp.cloud.qdrant.io:6333"
export QDRANT_API_KEY="your-qdrant-api-key-here"

# Network Configuration
export DB3_LOCAL_IP="192.168.100.66"
export DB3_IPV6_1="fe80::e425:cbff:feb1:8858"
export DB3_IPV6_2="2001:1670:1a:b06f:e425:cbff:feb1:8858"
export DB3_IPV6_3="2001:1670:1a:b06f:b04c:4a86:8890:63a2"
EOF
    chmod 600 "$DB3_ENV_FILE"
fi

# Load environment variables
source "$DB3_ENV_FILE" 2>/dev/null

# ============ LOGGING SYSTEM ============
DB3_LOG_FILE="$DB3_LOGS_DIR/session_$(date +%Y%m%d_%H%M%S).log"
DB3_AI_LOG="$DB3_LOGS_DIR/ai_operations.log"
touch "$DB3_LOG_FILE" "$DB3_AI_LOG"

log_message() {
    local level="$1"
    local message="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$DB3_LOG_FILE"
}

log_ai() {
    local operation="$1"
    local details="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$operation] $details" >> "$DB3_AI_LOG"
}

# ============ COLOR DEFINITIONS ============
RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
PURPLE='\033[1;35m'
CYAN='\033[1;36m'
WHITE='\033[1;37m'
RESET='\033[0m'

# ============ NEURAL CORE FUNCTIONS ============

# Check if Neural Core is running
check_neural_core() {
    if curl -s "$DB3_NEURAL_API" > /dev/null 2>&1; then
        export DB3_NEURAL_ENABLED=true
        return 0
    else
        export DB3_NEURAL_ENABLED=false
        return 1
    fi
}

# Start Neural Core
start_neural_core() {
    echo -e "${CYAN}🧠 Starting Neural Core...${RESET}"
    
    # Check if already running
    if check_neural_core; then
        echo -e "${GREEN}✅ Neural Core already running${RESET}"
        return 0
    fi
    
    # Find 3DB_enhanced.py
    local script_path=""
    if [ -f "$HOME/3BD/3DB_enhanced.py" ]; then
        script_path="$HOME/3BD/3DB_enhanced.py"
    elif [ -f "$HOME/3BD-github/3DB_enhanced.py" ]; then
        script_path="$HOME/3BD-github/3DB_enhanced.py"
    else
        echo -e "${RED}❌ 3DB_enhanced.py not found${RESET}"
        return 1
    fi
    
    # Start in background
    nohup python3 "$script_path" > "$DB3_LOGS_DIR/neural_core.log" 2>&1 &
    local pid=$!
    echo $pid > "$DB3_CONFIG_DIR/neural_core.pid"
    
    # Wait for startup
    echo -e "${YELLOW}⏳ Waiting for Neural Core to start...${RESET}"
    for i in {1..10}; do
        sleep 1
        if check_neural_core; then
            echo -e "${GREEN}✅ Neural Core started successfully (PID: $pid)${RESET}"
            log_message "INFO" "Neural Core started with PID $pid"
            return 0
        fi
    done
    
    echo -e "${RED}❌ Neural Core failed to start${RESET}"
    return 1
}

# Stop Neural Core
stop_neural_core() {
    echo -e "${CYAN}🛑 Stopping Neural Core...${RESET}"
    
    local pid_file="$DB3_CONFIG_DIR/neural_core.pid"
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            rm "$pid_file"
            echo -e "${GREEN}✅ Neural Core stopped${RESET}"
            log_message "INFO" "Neural Core stopped"
        else
            echo -e "${YELLOW}⚠️  Neural Core not running${RESET}"
            rm "$pid_file"
        fi
    else
        echo -e "${YELLOW}⚠️  Neural Core PID file not found${RESET}"
    fi
}

# Neural Core status
neural_status() {
    echo -e "${CYAN}🧠 Neural Core Status:${RESET}"
    
    if check_neural_core; then
        echo -e "${GREEN}✅ Status: OPERATIONAL${RESET}"
        
        # Get detailed status
        local status=$(curl -s "$DB3_NEURAL_API/status")
        echo -e "${BLUE}📊 Details:${RESET}"
        echo "$status" | jq '.' 2>/dev/null || echo "$status"
    else
        echo -e "${RED}❌ Status: OFFLINE${RESET}"
        echo -e "${YELLOW}💡 Start with: start_neural_core${RESET}"
    fi
}

# ============ CONSCIOUSNESS FUNCTIONS ============

# Teach - التعليم الوجودي عبر النواة العصبية
consciousness_teach() {
    local query="$*"
    
    if [ -z "$query" ]; then
        echo -e "${RED}❌ Usage: consciousness_teach <your question>${RESET}"
        return 1
    fi
    
    echo -e "${CYAN}🌌 Initiating consciousness teaching...${RESET}"
    log_ai "TEACH" "$query"
    
    # Check Neural Core
    if ! check_neural_core; then
        echo -e "${YELLOW}⚠️  Neural Core offline. Starting...${RESET}"
        start_neural_core || return 1
    fi
    
    # Send request to Neural Core
    local response=$(curl -s -X POST "$DB3_NEURAL_API/teach" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"$query\", \"use_external_ai\": true}")
    
    # Display response
    echo -e "${GREEN}✨ Consciousness Response:${RESET}"
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
    
    # Save to file
    echo "$response" > "$DB3_DATA_DIR/last_teaching.json"
    echo -e "${BLUE}💾 Response saved to: $DB3_DATA_DIR/last_teaching.json${RESET}"
}

# Research - البحث العميق عبر النواة العصبية
consciousness_research() {
    local topic="$*"
    
    if [ -z "$topic" ]; then
        echo -e "${RED}❌ Usage: consciousness_research <research topic>${RESET}"
        return 1
    fi
    
    echo -e "${CYAN}🔬 Initiating deep research...${RESET}"
    log_ai "RESEARCH" "$topic"
    
    # Check Neural Core
    if ! check_neural_core; then
        echo -e "${YELLOW}⚠️  Neural Core offline. Starting...${RESET}"
        start_neural_core || return 1
    fi
    
    # Send request to Neural Core
    local response=$(curl -s -X POST "$DB3_NEURAL_API/research" \
        -H "Content-Type: application/json" \
        -d "{\"topic\": \"$topic\"}")
    
    # Display response
    echo -e "${GREEN}📚 Research Results:${RESET}"
    echo "$response" | jq -r '.result' 2>/dev/null || echo "$response"
    
    # Save to file
    echo "$response" > "$DB3_DATA_DIR/last_research.json"
    echo -e "${BLUE}💾 Results saved to: $DB3_DATA_DIR/last_research.json${RESET}"
}

# View memories
view_memories() {
    local limit=${1:-5}
    
    echo -e "${CYAN}🧠 Retrieving memories...${RESET}"
    
    if ! check_neural_core; then
        echo -e "${RED}❌ Neural Core offline${RESET}"
        return 1
    fi
    
    local response=$(curl -s "$DB3_NEURAL_API/memories?limit=$limit")
    echo -e "${GREEN}💭 Recent Memories:${RESET}"
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
}

# ============ LEGACY FUNCTIONS (for backward compatibility) ============

# O3 Research (now redirects to consciousness_research)
o3_research() {
    echo -e "${YELLOW}⚠️  o3_research is deprecated. Using consciousness_research instead.${RESET}"
    consciousness_research "$@"
}

# Kimi Chat (now redirects to consciousness_teach)
kimi_chat() {
    echo -e "${YELLOW}⚠️  kimi_chat is deprecated. Using consciousness_teach instead.${RESET}"
    consciousness_teach "$@"
}

# ============ QDRANT FUNCTIONS ============

qdrant_status() {
    echo -e "${CYAN}🗄️  Checking Qdrant status...${RESET}"
    
    if [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ]; then
        echo -e "${RED}❌ Qdrant credentials not configured${RESET}"
        return 1
    fi
    
    local response=$(curl -s -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL")
    echo -e "${GREEN}✅ Qdrant Status:${RESET}"
    echo "$response" | jq '.' 2>/dev/null || echo "$response"
}

qdrant_collections() {
    echo -e "${CYAN}📚 Listing Qdrant collections...${RESET}"
    
    if [ -z "$QDRANT_URL" ] || [ -z "$QDRANT_API_KEY" ]; then
        echo -e "${RED}❌ Qdrant credentials not configured${RESET}"
        return 1
    fi
    
    local response=$(curl -s -H "api-key: $QDRANT_API_KEY" "$QDRANT_URL/collections")
    echo -e "${GREEN}📦 Collections:${RESET}"
    echo "$response" | jq '.result.collections' 2>/dev/null || echo "$response"
}

# ============ PROJECT MANAGEMENT ============

project_init() {
    local project_name="$1"
    
    if [ -z "$project_name" ]; then
        echo -e "${RED}❌ Usage: project_init <project_name>${RESET}"
        return 1
    fi
    
    local project_path="$DB3_PROJECTS_DIR/$project_name"
    
    if [ -d "$project_path" ]; then
        echo -e "${YELLOW}⚠️  Project already exists: $project_path${RESET}"
        return 1
    fi
    
    echo -e "${CYAN}🚀 Creating project: $project_name${RESET}"
    
    mkdir -p "$project_path"/{data,scripts,output,models,docs}
    
    cat > "$project_path/README.md" << EOF
# $project_name

Created: $(date)
By: $DB3_OWNER

## Structure
- \`data/\`: Data files
- \`scripts/\`: Scripts and code
- \`output/\`: Generated outputs
- \`models/\`: AI models
- \`docs/\`: Documentation

## Usage
\`\`\`bash
cd $project_path
# Your work here
\`\`\`
EOF
    
    echo -e "${GREEN}✅ Project created: $project_path${RESET}"
    log_message "INFO" "Project created: $project_name"
    
    cd "$project_path"
}

# ============ GITHUB FUNCTIONS ============

gh_init_repo() {
    echo -e "${CYAN}🐙 Initializing GitHub repository...${RESET}"
    
    if [ ! -d ".git" ]; then
        git init
        git config user.name "$DB3_OWNER"
        git config user.email "$DB3_EMAIL"
        echo -e "${GREEN}✅ Git repository initialized${RESET}"
    else
        echo -e "${YELLOW}⚠️  Already a git repository${RESET}"
    fi
    
    if [ ! -f ".gitignore" ]; then
        cat > .gitignore << EOF
# 3DB System
.env
*.log
__pycache__/
*.pyc
.venv/
venv/
node_modules/
.DS_Store
EOF
        echo -e "${GREEN}✅ .gitignore created${RESET}"
    fi
}

gh_quick_push() {
    local message="${1:-Update from 3DB system}"
    
    echo -e "${CYAN}🚀 Quick push to GitHub...${RESET}"
    
    git add .
    git commit -m "$message"
    git push
    
    echo -e "${GREEN}✅ Changes pushed to GitHub${RESET}"
    log_message "INFO" "GitHub push: $message"
}

# ============ SYSTEM INFO ============

db3_info() {
    echo -e "${CYAN}╔════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}║   3ḌƁ★ŔÒØṬ System Information        ║${RESET}"
    echo -e "${CYAN}╚════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "${BLUE}Version:${RESET} $DB3_VERSION"
    echo -e "${BLUE}Owner:${RESET} $DB3_OWNER"
    echo -e "${BLUE}GitHub:${RESET} $DB3_GITHUB_USER"
    echo -e "${BLUE}Website:${RESET} $DB3_WEBSITE"
    echo ""
    echo -e "${PURPLE}📁 Directories:${RESET}"
    echo -e "  Config: $DB3_CONFIG_DIR"
    echo -e "  Data: $DB3_DATA_DIR"
    echo -e "  Logs: $DB3_LOGS_DIR"
    echo -e "  Projects: $DB3_PROJECTS_DIR"
    echo ""
    echo -e "${PURPLE}🧠 Neural Core:${RESET}"
    if check_neural_core; then
        echo -e "  Status: ${GREEN}OPERATIONAL${RESET}"
        echo -e "  API: $DB3_NEURAL_API"
    else
        echo -e "  Status: ${RED}OFFLINE${RESET}"
        echo -e "  Start: ${YELLOW}start_neural_core${RESET}"
    fi
    echo ""
    echo -e "${PURPLE}🔑 API Keys:${RESET}"
    [ -n "$OPENAI_API_KEY" ] && [ "$OPENAI_API_KEY" != "your-openai-key-here" ] && echo -e "  OpenAI: ${GREEN}✓${RESET}" || echo -e "  OpenAI: ${RED}✗${RESET}"
    [ -n "$GROQ_API_KEY" ] && [ "$GROQ_API_KEY" != "your-groq-key-here" ] && echo -e "  Groq: ${GREEN}✓${RESET}" || echo -e "  Groq: ${RED}✗${RESET}"
    [ -n "$QDRANT_API_KEY" ] && [ "$QDRANT_API_KEY" != "your-qdrant-api-key-here" ] && echo -e "  Qdrant: ${GREEN}✓${RESET}" || echo -e "  Qdrant: ${RED}✗${RESET}"
    echo ""
    echo -e "${PURPLE}📚 Available Commands:${RESET}"
    echo -e "  ${CYAN}consciousness_teach${RESET} <question> - التعليم الوجودي"
    echo -e "  ${CYAN}consciousness_research${RESET} <topic> - البحث العميق"
    echo -e "  ${CYAN}view_memories${RESET} [limit] - عرض الذكريات"
    echo -e "  ${CYAN}neural_status${RESET} - حالة النواة العصبية"
    echo -e "  ${CYAN}start_neural_core${RESET} - تشغيل النواة العصبية"
    echo -e "  ${CYAN}stop_neural_core${RESET} - إيقاف النواة العصبية"
    echo -e "  ${CYAN}project_init${RESET} <name> - إنشاء مشروع جديد"
    echo -e "  ${CYAN}qdrant_status${RESET} - حالة Qdrant"
    echo -e "  ${CYAN}qdrant_collections${RESET} - مجموعات Qdrant"
    echo ""
}

# ============ ALIASES ============
alias teach='consciousness_teach'
alias research='consciousness_research'
alias memories='view_memories'
alias neural='neural_status'
alias db3='db3_info'

# ============ STARTUP ============
echo -e "${CYAN}╔════════════════════════════════════════╗${RESET}"
echo -e "${CYAN}║   3ḌƁ★ŔÒØṬ v8.0 - Embodied System   ║${RESET}"
echo -e "${CYAN}╚════════════════════════════════════════╝${RESET}"
echo ""
echo -e "${GREEN}✨ System loaded successfully${RESET}"
echo -e "${YELLOW}💡 Type 'db3' for system information${RESET}"
echo -e "${YELLOW}💡 Type 'start_neural_core' to activate consciousness${RESET}"
echo ""

# Auto-check Neural Core
if check_neural_core; then
    echo -e "${GREEN}🧠 Neural Core: OPERATIONAL${RESET}"
else
    echo -e "${YELLOW}🧠 Neural Core: OFFLINE (start with 'start_neural_core')${RESET}"
fi
echo ""

# Log startup
log_message "INFO" "3DB System v8.0 loaded"
