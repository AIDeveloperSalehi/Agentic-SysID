#!/usr/bin/env bash
# Run the agentic system identification pipeline on the pendulum plant.
# Usage:
#   ./run_pendulum.sh                    # default settings
#   ./run_pendulum.sh --budget 150       # override budget
#   ./run_pendulum.sh --seed 7           # override random seed
#   ./run_pendulum.sh --log-level DEBUG  # verbose console output

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load .env ─────────────────────────────────────────────────────────────────
if [[ ! -f ".env" ]]; then
    echo "ERROR: .env file not found"
    exit 1
fi

set -o allexport
source ".env"
set +o allexport

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "ERROR: ANTHROPIC_API_KEY not found in .env"
    exit 1
fi

# ── Run ───────────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║      Agentic System Identification — Pendulum Run        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

python main.py --config configs/pendulum_config.yaml "$@"

echo ""
echo "Done. Latest log: $(ls -t logs/run_*.log 2>/dev/null | head -1)"
