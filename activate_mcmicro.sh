#!/usr/bin/env bash
# Activate the cifsQuant conda environment.
# Run: source activate_mcmicro.sh

CONDA_BASE="$(conda info --base 2>/dev/null)"
if [[ -z "$CONDA_BASE" ]]; then
    echo "conda not found. Install Miniconda and run setup_environment.sh first."
    exit 1
fi

source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate cifsquant

echo "cifsquant environment activated."
echo "Run the pipeline:  python run_cifsquant.py --project project.yaml"
echo "Deactivate:        conda deactivate"
