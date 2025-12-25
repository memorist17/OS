#!/bin/bash
# Start dashboard and ngrok with health checks and auto-restart

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DASHBOARD_LOG="/tmp/dashboard.log"
NGROK_LOG="/tmp/ngrok.log"
DASHBOARD_PORT=8050
NGROK_WEB_PORT=4040

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if port is listening
check_port() {
    local port=$1
    if command -v netstat >/dev/null 2>&1; then
        netstat -tlnp 2>/dev/null | grep -q ":$port " || ss -tlnp 2>/dev/null | grep -q ":$port "
    elif command -v ss >/dev/null 2>&1; then
        ss -tlnp 2>/dev/null | grep -q ":$port "
    else
        # Fallback: try to connect
        timeout 1 bash -c "echo >/dev/tcp/127.0.0.1/$port" 2>/dev/null
    fi
}

# Function to check if dashboard is responding
check_dashboard() {
    # Use timeout to avoid hanging
    timeout 2 curl -s -f http://127.0.0.1:$DASHBOARD_PORT >/dev/null 2>&1
}

# Function to kill existing processes
cleanup() {
    echo -e "${YELLOW}Cleaning up existing processes...${NC}"
    pkill -f "run_unified_dashboard" 2>/dev/null || true
    pkill -f "ngrok" 2>/dev/null || true
    sleep 2
}

# Function to start dashboard
start_dashboard() {
    echo -e "${GREEN}Starting dashboard...${NC}"
    cd "$PROJECT_DIR"
    
    # Check if dashboard is already running
    if check_port $DASHBOARD_PORT && check_dashboard; then
        echo -e "${GREEN}Dashboard is already running on port $DASHBOARD_PORT${NC}"
        return 0
    fi
    
    # Start dashboard in background
    nohup python scripts/run_unified_dashboard.py \
        --outputs-dir outputs \
        --port $DASHBOARD_PORT \
        --host 127.0.0.1 \
        > "$DASHBOARD_LOG" 2>&1 &
    
    local dashboard_pid=$!
    echo "Dashboard PID: $dashboard_pid"
    
    # Wait for dashboard to start (max 60 seconds - increased for large datasets)
    local max_wait=60
    local waited=0
    local last_log_size=0
    
    while [ $waited -lt $max_wait ]; do
        # Check if process is still running
        if ! kill -0 $dashboard_pid 2>/dev/null; then
            echo -e "\n${RED}Dashboard process died!${NC}"
            echo "Last 30 lines of dashboard log:"
            tail -30 "$DASHBOARD_LOG"
            return 1
        fi
        
        # Check if port is listening and dashboard is responding
        if check_port $DASHBOARD_PORT && check_dashboard; then
            echo -e "\n${GREEN}Dashboard started successfully!${NC}"
            return 0
        fi
        
        # Show progress every 5 seconds
        if [ $((waited % 5)) -eq 0 ] && [ $waited -gt 0 ]; then
            current_log_size=$(wc -l < "$DASHBOARD_LOG" 2>/dev/null || echo 0)
            if [ $current_log_size -gt $last_log_size ]; then
                echo -n " [loading...]"
                last_log_size=$current_log_size
            fi
        fi
        
        sleep 1
        waited=$((waited + 1))
        echo -n "."
    done
    
    # Check if process is still running (might just be slow)
    if kill -0 $dashboard_pid 2>/dev/null; then
        echo -e "\n${YELLOW}Dashboard is still starting (process running, but not responding yet)${NC}"
        echo "This may be normal for large datasets. Check logs: tail -f $DASHBOARD_LOG"
        echo "Dashboard PID: $dashboard_pid"
        return 0  # Don't fail, let it continue
    else
        echo -e "\n${RED}Dashboard failed to start within $max_wait seconds${NC}"
        echo "Last 30 lines of dashboard log:"
        tail -30 "$DASHBOARD_LOG"
        return 1
    fi
}

# Function to start ngrok
start_ngrok() {
    echo -e "${GREEN}Starting ngrok...${NC}"
    
    # Check if ngrok is already running
    if check_port $NGROK_WEB_PORT; then
        echo -e "${GREEN}ngrok is already running${NC}"
        # Get public URL
        local public_url=$(curl -s http://127.0.0.1:$NGROK_WEB_PORT/api/tunnels 2>/dev/null | \
            python3 -c "import sys, json; data = json.load(sys.stdin); \
            print(data['tunnels'][0]['public_url'] if data.get('tunnels') else '')" 2>/dev/null || echo "")
        if [ -n "$public_url" ]; then
            echo -e "${GREEN}ngrok URL: $public_url${NC}"
        fi
        return 0
    fi
    
    # Check if ngrok is installed
    if ! command -v ngrok >/dev/null 2>&1; then
        echo -e "${RED}ngrok is not installed${NC}"
        return 1
    fi
    
    # Start ngrok in background
    cd "$PROJECT_DIR"
    nohup ngrok http $DASHBOARD_PORT --log=stdout > "$NGROK_LOG" 2>&1 &
    
    local ngrok_pid=$!
    echo "ngrok PID: $ngrok_pid"
    
    # Wait for ngrok to start (max 10 seconds)
    local max_wait=10
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if check_port $NGROK_WEB_PORT; then
            sleep 2  # Give ngrok time to establish tunnel
            local public_url=$(curl -s http://127.0.0.1:$NGROK_WEB_PORT/api/tunnels 2>/dev/null | \
                python3 -c "import sys, json; data = json.load(sys.stdin); \
                print(data['tunnels'][0]['public_url'] if data.get('tunnels') else '')" 2>/dev/null || echo "")
            if [ -n "$public_url" ]; then
                echo -e "${GREEN}ngrok started successfully!${NC}"
                echo -e "${GREEN}Public URL: $public_url${NC}"
                return 0
            fi
        fi
        sleep 1
        waited=$((waited + 1))
        echo -n "."
    done
    
    echo -e "\n${YELLOW}ngrok may still be starting. Check logs: tail -f $NGROK_LOG${NC}"
    return 0
}

# Function to monitor and restart if needed
monitor() {
    echo -e "${GREEN}Monitoring dashboard and ngrok...${NC}"
    echo "Press Ctrl+C to stop"
    
    while true; do
        sleep 30  # Check every 30 seconds
        
        # Check dashboard
        if ! check_dashboard; then
            echo -e "${RED}[$(date)] Dashboard is not responding, restarting...${NC}"
            cleanup
            start_dashboard
        fi
        
        # Check ngrok (only if dashboard is running)
        if check_dashboard && ! check_port $NGROK_WEB_PORT; then
            echo -e "${YELLOW}[$(date)] ngrok is not running, restarting...${NC}"
            start_ngrok
        fi
    done
}

# Main execution
main() {
    echo "=========================================="
    echo "Dashboard & ngrok Startup Script"
    echo "=========================================="
    echo ""
    
    # Cleanup existing processes
    cleanup
    
    # Start dashboard
    if ! start_dashboard; then
        echo -e "${RED}Failed to start dashboard. Exiting.${NC}"
        exit 1
    fi
    
    # Start ngrok
    start_ngrok
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Setup complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Dashboard: http://127.0.0.1:$DASHBOARD_PORT"
    if check_port $NGROK_WEB_PORT; then
        local public_url=$(curl -s http://127.0.0.1:$NGROK_WEB_PORT/api/tunnels 2>/dev/null | \
            python3 -c "import sys, json; data = json.load(sys.stdin); \
            print(data['tunnels'][0]['public_url'] if data.get('tunnels') else '')" 2>/dev/null || echo "")
        if [ -n "$public_url" ]; then
            echo "Public URL: $public_url"
        fi
    fi
    echo ""
    echo "Logs:"
    echo "  Dashboard: tail -f $DASHBOARD_LOG"
    echo "  ngrok: tail -f $NGROK_LOG"
    echo ""
    
    # Start monitoring if requested
    if [ "$1" == "--monitor" ]; then
        monitor
    else
        echo "To start monitoring (auto-restart), run with --monitor flag"
        echo "Example: $0 --monitor"
    fi
}

# Run main function
main "$@"

