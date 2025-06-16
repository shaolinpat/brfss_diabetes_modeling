#!/bin/bash
echo "Registering Jupyter kernel for brfss..."

# Ensure conda is initialized
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate the brfss environment
conda activate brfss_env

# Register the kernel
python -m ipykernel install --user --name=brfss_env --display-name="brfss_env"

