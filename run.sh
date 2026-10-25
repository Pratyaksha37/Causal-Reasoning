#!/usr/bin/env bash
# ==============================================================================
# One-Command Runner for Phase 1 Causal Inference Pipeline
#
# Usage:
#   ./run.sh          -> Runs the complete end-to-end pipeline
#   ./run.sh test     -> Runs all test suites (pytest)
#   ./run.sh <args>   -> Passes custom args to run_pipeline.py (e.g. --max-eval-pairs 20)
# ==============================================================================

set -e

# Resolve script directory (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "🚀 Causal Relation Extraction & Evidence Graph Pipeline"
echo "   Working Directory: $PROJECT_ROOT"
echo "======================================================================"

# 1. Check or setup virtual environment
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
VENV_PIP="$PROJECT_ROOT/.venv/bin/pip"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "⚠️ Virtual environment not detected at .venv. Creating one now..."
    if command -v python3.11 &> /dev/null; then
        python3.11 -m venv .venv
    else
        python3 -m venv .venv
    fi
    echo "📦 Installing project dependencies..."
    "$VENV_PIP" install --upgrade pip
    "$VENV_PIP" install -e ".[dev]"
    "$VENV_PYTHON" -m spacy download en_core_web_sm
    echo "✅ Virtual environment created and initialized."
fi

# 2. Check for .env file
if [ ! -f "$PROJECT_ROOT/.env" ]; then
    if [ -f "$PROJECT_ROOT/.env.example" ]; then
        echo "ℹ️ No .env found. Copying .env.example to .env..."
        cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
        echo "⚠️ Please add your GROQ_API_KEY inside .env if not already set."
    fi
fi

# 3. Handle 'test' argument
if [ "$1" = "test" ] || [ "$1" = "tests" ]; then
    echo "🧪 Running pytest test suite..."
    shift
    exec "$PROJECT_ROOT/.venv/bin/pytest" -v "$@"
fi

# 4. Run the end-to-end pipeline
echo "▶️ Running pipeline via .venv/bin/python run_pipeline.py..."
"$VENV_PYTHON" run_pipeline.py "$@"

echo ""
echo "======================================================================"
echo "🎉 SUCCESS! Pipeline execution complete."
echo "   📊 Evaluation Table:        results/tables/intrinsic_evaluation.csv"
echo "   🖼️ Evidence Graph figure:   results/figures/evidence_graph_after_repair.png"
echo "   📦 GraphML Export:          results/graph/evidence_graph_after_repair.graphml"
echo "   📑 Provenance Metadata:     results/graph/provenance.json"
echo "======================================================================"
