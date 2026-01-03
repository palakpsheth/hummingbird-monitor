#!/usr/bin/env bash
#
# run_detection_test.sh
#
# End-to-end detection pipeline test:
# 1. Finds a test video from recent observations
# 2. Starts an RTSP test server with that video
# 3. Restarts the worker with the test RTSP URL
# 4. Monitors for bird detections
# 5. Reports results and cleans up
#
# Usage:
#   ./scripts/run_detection_test.sh              # Auto-select latest video
#   ./scripts/run_detection_test.sh --video /path/to/video.mp4
#   ./scripts/run_detection_test.sh --observation-id 28
#   ./scripts/run_detection_test.sh --duration 60   # Monitor for 60 seconds
#
# The script will:
#   - Start a test RTSP server on port 8555
#   - Temporarily override HBMON_RTSP_URL
#   - Restart hbmon-worker to use the test stream
#   - Monitor logs for bird detections
#   - Report success/failure
#   - Restore original RTSP URL and restart worker
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_PORT=8555
MONITOR_DURATION=30
OBSERVATION_ID=""
VIDEO_PATH=""
KEEP_RUNNING=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --video|-v)
            VIDEO_PATH="$2"
            shift 2
            ;;
        --observation-id|-o)
            OBSERVATION_ID="$2"
            shift 2
            ;;
        --duration|-d)
            MONITOR_DURATION="$2"
            shift 2
            ;;
        --keep-running|-k)
            KEEP_RUNNING=true
            shift
            ;;
        --direct)
            DIRECT_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --video, -v PATH          Use specific video file"
            echo "  --observation-id, -o ID   Use video from observation ID"
            echo "  --duration, -d SECONDS    Monitor duration (default: 30)"
            echo "  --keep-running, -k        Don't restore original URL after test"
            echo "  --help, -h                Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  HBMON Detection Pipeline Test${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Find test video
echo -e "${YELLOW}[1/6] Finding test video...${NC}"

if [[ -n "$VIDEO_PATH" ]]; then
    if [[ ! -f "$VIDEO_PATH" ]]; then
        echo -e "${RED}ERROR: Video not found: $VIDEO_PATH${NC}"
        exit 1
    fi
    echo "  Using specified video: $VIDEO_PATH"
elif [[ -n "$OBSERVATION_ID" ]]; then
    VIDEO_PATH=$(uv run python scripts/extract_test_videos.py --observation-id "$OBSERVATION_ID" --paths-only -f 2>/dev/null | head -1)
    if [[ -z "$VIDEO_PATH" ]]; then
        echo -e "${RED}ERROR: No video found for observation ID $OBSERVATION_ID${NC}"
        exit 1
    fi
    echo "  Found video for observation #$OBSERVATION_ID: $VIDEO_PATH"
else
    # Try true positive observations first (reviewed and confirmed)
    VIDEO_PATH=$(uv run python scripts/extract_test_videos.py --true-positive --paths-only -f --limit 1 2>/dev/null | head -1)
    if [[ -n "$VIDEO_PATH" ]]; then
        echo "  Using true positive observation: $VIDEO_PATH"
    else
        # Fallback to latest observation (may not be reviewed)
        VIDEO_PATH=$(uv run python scripts/extract_test_videos.py --paths-only -f --limit 1 2>/dev/null | head -1)
        if [[ -z "$VIDEO_PATH" ]]; then
            echo -e "${RED}ERROR: No observation videos found in database${NC}"
            echo "  Make sure containers are running: make docker-up-intel"
            exit 1
        fi
        echo -e "  ${YELLOW}Warning: No true positive observations found${NC}"
        echo "  Using unreviewed observation: $VIDEO_PATH"
        echo "  Consider labeling observations in the web UI for reliable testing."
    fi
fi

# Step 2: Get current RTSP URL (to restore later)
echo -e "${YELLOW}[2/6] Saving current configuration...${NC}"

ORIGINAL_RTSP_URL=$(grep "^HBMON_RTSP_URL=" .env 2>/dev/null | cut -d= -f2- || echo "")
if [[ -z "$ORIGINAL_RTSP_URL" ]]; then
    # Check docker-compose.yml for default
    ORIGINAL_RTSP_URL="rtsp://192.168.1.52:8554/hummingbirdcam"
fi
echo "  Original RTSP URL: $ORIGINAL_RTSP_URL"

if [[ "$DIRECT_MODE" == "true" ]]; then
    # In direct mode, we use the file path directly to bypass RTSP server entirely
    # Assuming video is in data/media/clips/... and mapped to /data/media/clips/...
    # Host: /media/palak/hbmon2/hummingbird-monitor/data/media/clips/...
    # Container structure: /data maps to ./data
    
    # Heuristic: strip everything up to /data/
    REL_PATH="${VIDEO_PATH#*/data/}"
    TEST_RTSP_URL="/data/$REL_PATH"
    
    echo "  Direct File Mode: $TEST_RTSP_URL"
else
    TEST_RTSP_URL="rtsp://172.17.0.1:${TEST_PORT}/test"
    echo "  Test RTSP URL: $TEST_RTSP_URL"
fi

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Cleaning up...${NC}"
    
    # Kill RTSP server if running
    if [[ -n "${RTSP_PID:-}" ]] && kill -0 "$RTSP_PID" 2>/dev/null; then
        echo "  Stopping RTSP test server (PID $RTSP_PID)..."
        kill "$RTSP_PID" 2>/dev/null || true
        wait "$RTSP_PID" 2>/dev/null || true
    fi
    
    # Stop mediamtx container if running
    if docker ps -q --filter "name=hbmon-rtsp-test" 2>/dev/null | grep -q .; then
        echo "  Stopping mediamtx container..."
        docker rm -f hbmon-rtsp-test >/dev/null 2>&1 || true
    fi
    
    # Restore original RTSP URL if not keeping
    if [[ "$KEEP_RUNNING" != "true" && -n "$ORIGINAL_RTSP_URL" ]]; then
        echo "  Restoring original RTSP URL..."
        if grep -q "^HBMON_RTSP_URL=" .env 2>/dev/null; then
            sed -i "s|^HBMON_RTSP_URL=.*|HBMON_RTSP_URL=$ORIGINAL_RTSP_URL|" .env
        fi
        echo "  Restarting worker with original stream..."
        docker compose up -d hbmon-worker >/dev/null 2>&1 || true
    fi
    
    echo -e "${GREEN}Cleanup complete.${NC}"
}

trap cleanup EXIT

# Step 3: Start RTSP test server (Only in non-direct mode)
if [[ "$DIRECT_MODE" != "true" ]]; then
    echo -e "${YELLOW}[3/6] Starting RTSP test server...${NC}"
    echo "  Video: $(basename "$VIDEO_PATH")"
    echo "  Port: $TEST_PORT"

    # Start RTSP server in background
    uv run python scripts/rtsp_test_server.py --port "$TEST_PORT" "$VIDEO_PATH" &
    RTSP_PID=$!

    # Wait for server to start
    sleep 3

    if ! kill -0 "$RTSP_PID" 2>/dev/null; then
        echo -e "${RED}ERROR: RTSP server failed to start${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}RTSP server started (PID $RTSP_PID)${NC}"
else
    echo -e "${YELLOW}[3/6] Skipping RTSP server (Direct Mode)${NC}"
fi

# Step 4: Update .env and restart worker
echo -e "${YELLOW}[4/6] Configuring worker for test stream...${NC}"

# Update .env with test URL
if grep -q "^HBMON_RTSP_URL=" .env 2>/dev/null; then
    sed -i "s|^HBMON_RTSP_URL=.*|HBMON_RTSP_URL=$TEST_RTSP_URL|" .env
else
    echo "HBMON_RTSP_URL=$TEST_RTSP_URL" >> .env
fi

echo "  Restarting hbmon-worker..."
docker compose up -d hbmon-worker

# Wait for worker to start
sleep 5
echo -e "  ${GREEN}Worker restarted${NC}"

# Step 5: Monitor for detections
echo -e "${YELLOW}[5/6] Monitoring for detections (${MONITOR_DURATION}s)...${NC}"
echo ""
echo "  Watching for bird detections in worker logs..."
echo "  (Press Ctrl+C to stop early)"
echo ""

DETECTION_COUNT=0
VISIT_COUNT=0
START_TIME=$(date +%s)

# Monitor logs with timeout
set +e
while true; do
    ELAPSED=$(($(date +%s) - START_TIME))
    if [[ $ELAPSED -ge $MONITOR_DURATION ]]; then
        break
    fi
    
    # Check recent logs (last 5 seconds)
    RECENT=$(docker compose logs --since 5s hbmon-worker 2>&1)
    
    # Count detections
    NEW_BIRDS=$(echo "$RECENT" | grep -c "1 bird\|2 bird" || true)
    NEW_VISITS=$(echo "$RECENT" | grep -c "Visit STARTED" || true)
    
    if [[ $NEW_BIRDS -gt 0 ]]; then
        DETECTION_COUNT=$((DETECTION_COUNT + NEW_BIRDS))
        echo -e "  ${GREEN}🐦 Bird detected! (total: $DETECTION_COUNT)${NC}"
    fi
    
    if [[ $NEW_VISITS -gt 0 ]]; then
        VISIT_COUNT=$((VISIT_COUNT + NEW_VISITS))
        echo -e "  ${GREEN}📹 Visit started! (total: $VISIT_COUNT)${NC}"
    fi
    
    # Show progress
    REMAINING=$((MONITOR_DURATION - ELAPSED))
    printf "\r  Time remaining: %ds | Detections: %d | Visits: %d    " "$REMAINING" "$DETECTION_COUNT" "$VISIT_COUNT"
    
    sleep 2
done
set -e

echo ""
echo ""

# Step 6: Report results
echo -e "${YELLOW}[6/6] Test Results${NC}"
echo -e "${BLUE}========================================${NC}"

if [[ $DETECTION_COUNT -gt 0 ]]; then
    echo -e "  ${GREEN}✓ DETECTION WORKING${NC}"
    echo ""
    echo "  Bird frame detections: $DETECTION_COUNT"
    echo "  Visit events recorded: $VISIT_COUNT"
    echo ""
    echo "  The detection pipeline is operational!"
    echo "  If live stream still shows no detections,"
    echo "  the issue may be with RTSP stream quality."
    EXIT_CODE=0
else
    echo -e "  ${RED}✗ NO DETECTIONS${NC}"
    echo ""
    echo "  No birds were detected in ${MONITOR_DURATION}s of testing."
    echo ""
    echo "  Possible causes:"
    echo "  1. YOLO model issue - try clearing OpenVINO cache:"
    echo "     make docker-down && make clean-openvino-cache && make docker-up-intel"
    echo ""
    echo "  2. Detection threshold too high - check Config UI:"
    echo "     - detect_conf: try 0.05 (currently may be 0.10)"
    echo "     - min_box_area: try 400 (currently may be 600)"
    echo ""
    echo "  3. Test with direct snapshot analysis:"
    echo "     uv run python scripts/test_detection.py --sweep-conf"
    echo ""
    EXIT_CODE=1
fi

echo -e "${BLUE}========================================${NC}"

# Cleanup happens via trap
exit $EXIT_CODE
