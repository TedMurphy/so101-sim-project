#!/bin/bash
# Retries the teleop launch if it stalls. stdbuf -oL is applied to EVERY
# stage of the pipe (not just isaaclab.sh) since grep/tee buffer their own
# output when writing to a non-terminal, which can hide a real success from
# this script's monitoring loop for a long time.

TIMEOUT=180
MAX_ATTEMPTS=200
CMD='cd ~/IsaacLab && stdbuf -oL ./isaaclab.sh -p ~/so101-sim-project/source/sim_to_real_so101/scripts/lerobot_agent.py --task Lerobot-So101-Teleop-Cylinder-To-Target --device cpu --robot_id SO101_leader_arm --enable_cameras --livestream 2 --kit_args "--ext-folder /home/ted/isaac-sim/extsDeprecated --enable isaacsim.core.utils --/exts/omni.kit.livestream.app/primaryStream/allowDynamicResize=true" 2>&1 | stdbuf -oL grep -v "getMaterialFromInternalFaceIndex"'

for attempt in $(seq 1 $MAX_ATTEMPTS); do
    echo "=== Attempt $attempt/$MAX_ATTEMPTS ==="
    LOGFILE=$(mktemp)
    eval "$CMD" | stdbuf -oL tee "$LOGFILE" &
    PID=$!

    SUCCESS=0
    for i in $(seq 1 $TIMEOUT); do
        if grep -q "Connected to the Arm" "$LOGFILE" 2>/dev/null; then
            SUCCESS=1
            break
        fi
        if ! kill -0 $PID 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if [ $SUCCESS -eq 1 ]; then
        echo "=== SUCCESS on attempt $attempt ==="
        wait $PID
        rm -f "$LOGFILE"
        exit 0
    else
        echo "=== Attempt $attempt stalled after ${TIMEOUT}s, killing and retrying ==="
        pkill -9 -f lerobot_agent.py
        sleep 5
        rm -f "$LOGFILE"
    fi
done

echo "=== Failed after $MAX_ATTEMPTS attempts ==="
exit 1
