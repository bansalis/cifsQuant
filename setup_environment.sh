#!/usr/bin/env bash
# ============================================================================
# cifsQuant Environment Setup
# ============================================================================
# Sets up everything needed to run cifsQuant on a fresh workstation.
# Run once after cloning the repository:
#
#   bash setup_environment.sh
#
# Prerequisites (install these before running this script):
#   - conda or mamba  (https://docs.conda.io/en/latest/miniconda.html)
#   - Docker          (https://docs.docker.com/engine/install/)
#   - Nextflow        (https://nextflow.io — for Stage 1 segmentation only)
#   - NVIDIA drivers  (recommended for GPU-accelerated segmentation)
#
# What this script does:
#   1. Checks prerequisites
#   2. Creates the 'cifsquant' conda environment from environment.yaml
#   3. Installs SpatialCells from source
#   4. Verifies all imports work
# ============================================================================

set -e
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

ok()   { echo -e "${GREEN}  ✓${NC} $1"; }
warn() { echo -e "${YELLOW}  ⚠${NC} $1"; }
err()  { echo -e "${RED}  ✗${NC} $1"; }

echo ""
echo "=============================================="
echo " cifsQuant Environment Setup"
echo "=============================================="
echo ""


# ─────────────────────────────────────────────────────────
# 1. CHECK PREREQUISITES
# ─────────────────────────────────────────────────────────
echo "[1/4] Checking prerequisites..."

if command -v mamba &>/dev/null; then
    CONDA_CMD=mamba; ok "mamba found (faster installs)"
elif command -v conda &>/dev/null; then
    CONDA_CMD=conda; ok "conda found"
else
    err "conda not found. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if command -v docker &>/dev/null; then
    ok "Docker: $(docker --version | head -1)"
else
    warn "Docker not found — Stage 1 segmentation will not work."
    warn "Install: https://docs.docker.com/engine/install/"
fi

if command -v nextflow &>/dev/null; then
    ok "Nextflow: $(nextflow -version 2>&1 | grep -o 'version .*' | head -1)"
else
    warn "Nextflow not found — Stage 1 segmentation will not work."
    warn "Install: curl -s https://get.nextflow.io | bash && sudo mv nextflow /usr/local/bin/"
fi

if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: ${GPU_INFO}"
else
    warn "No NVIDIA GPU — CPU fallback will be used for normalization."
fi
echo ""


# ─────────────────────────────────────────────────────────
# 2. CREATE CONDA ENVIRONMENT
# ─────────────────────────────────────────────────────────
echo "[2/4] Setting up 'cifsquant' conda environment..."

if $CONDA_CMD env list 2>/dev/null | grep -q "^cifsquant "; then
    warn "Environment 'cifsquant' already exists — updating..."
    $CONDA_CMD env update -n cifsquant -f "${REPO_DIR}/environment.yaml" --prune
else
    echo "  Creating environment from environment.yaml (takes 5–10 min)..."
    $CONDA_CMD env create -f "${REPO_DIR}/environment.yaml"
fi

ok "Conda environment 'cifsquant' ready"

# Activate for remaining steps
CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cifsquant
echo ""


# ─────────────────────────────────────────────────────────
# 3. INSTALL SPATIALCELLS
# ─────────────────────────────────────────────────────────
echo "[3/4] Installing SpatialCells..."

# SpatialCells is not on PyPI.
# Auto-detect source directory in common locations:
SPATIALCELLS_DIR=""
SEARCH_PATHS=(
    "${REPO_DIR}/../SpatialCells"
    "${HOME}/SpatialCells"
    "/opt/SpatialCells"
    "${REPO_DIR}/vendor/SpatialCells"
)

for p in "${SEARCH_PATHS[@]}"; do
    if [[ -d "$p" ]] && ([[ -f "$p/setup.py" ]] || [[ -f "$p/pyproject.toml" ]]); then
        SPATIALCELLS_DIR="$(cd "$p" && pwd)"
        break
    fi
done

if python -c "import spatialcells" &>/dev/null 2>&1; then
    SC_VER=$(python -c "import spatialcells; print(getattr(spatialcells,'__version__','?'))" 2>/dev/null)
    ok "SpatialCells already installed (v${SC_VER})"
elif [[ -n "$SPATIALCELLS_DIR" ]]; then
    ok "Found SpatialCells source: $SPATIALCELLS_DIR"
    pip install -e "$SPATIALCELLS_DIR"
    ok "SpatialCells installed"
else
    warn "SpatialCells not found. Spatial structure analyses (per_tumor_analysis,"
    warn "immune_infiltration, marker_region_analysis) require this library."
    warn ""
    warn "To install, get the SpatialCells package from the MGH Systems Biology lab"
    warn "or your collaborator, then run:"
    warn "   conda activate cifsquant && pip install -e /path/to/SpatialCells"
    warn ""
    warn "Place the SpatialCells directory next to this repo for auto-detection:"
    warn "   /path/to/SpatialCells/      ← SpatialCells source"
    warn "   /path/to/cifsQuant/         ← this repo"
fi
echo ""


# ─────────────────────────────────────────────────────────
# 4. VERIFY INSTALLATION
# ─────────────────────────────────────────────────────────
echo "[4/4] Verifying package imports..."

python - <<'PYEOF'
import sys

failures = []

def check(label, name, optional=False):
    try:
        m = __import__(name)
        ver = getattr(m, '__version__', '?')
        print(f"  ✓  {label:<25} {ver}")
    except ImportError:
        if optional:
            print(f"  ⚠  {label:<25} not installed (optional)")
        else:
            print(f"  ✗  {label:<25} MISSING")
            failures.append(label)

print("\n  Core:")
check("numpy",         "numpy")
check("pandas",        "pandas")
check("scipy",         "scipy")
check("matplotlib",    "matplotlib")
check("seaborn",       "seaborn")
check("scikit-learn",  "sklearn")
check("scikit-image",  "skimage")
check("statsmodels",   "statsmodels")
check("pyyaml",        "yaml")
check("joblib",        "joblib")
check("tifffile",      "tifffile")

print("\n  Single-cell / spatial:")
check("anndata",       "anndata")
check("scanpy",        "scanpy")
check("scimap",        "scimap")
check("umap-learn",    "umap")
check("shapely",       "shapely")
check("igraph",        "igraph")
check("leidenalg",     "leidenalg")
check("networkx",      "networkx")

print("\n  Optional:")
check("spatialcells",  "spatialcells", optional=True)
check("cupy (GPU)",    "cupy",         optional=True)
check("torch",         "torch",        optional=True)

if failures:
    print(f"\n  FAILED: {failures}")
    print("  Try: pip install " + " ".join(failures))
    sys.exit(1)
else:
    print("\n  All required packages OK.")
PYEOF

echo ""
echo "=============================================="
echo " Setup complete!"
echo "=============================================="
echo ""
echo "  Activate environment:"
echo "    conda activate cifsquant"
echo ""
echo "  Copy an example config and start a project:"
echo "    cp configs/examples/batch25_tumor_kp/project.yaml project.yaml"
echo "    # edit project.yaml for your study"
echo ""
echo "  Validate config:"
echo "    python run_cifsquant.py --project project.yaml --dry-run"
echo ""
echo "  Run pipeline:"
echo "    python run_cifsquant.py --project project.yaml"
echo ""
