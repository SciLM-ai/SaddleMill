# tsearch

## Installation on Vista

### 1. Base Environment Setup

Load necessary modules and create the base Conda environment.

```bash
module reset
module unload xalt

# Create base environment
conda create -n executorlib -c conda-forge python=3.12 flux-core flux-sched "openmpi=5.0.5=external_*" executorlib
conda activate executorlib

# Install hardware locality
conda install "libhwloc=*=cuda*" -c conda-forge
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

```

**(Optional) MPI Support:**
If `mpi4py` is required, use the following (experimental):

```bash
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py

```

### 2. Verify Flux Resources

Check if flux detects all resources correctly:

```bash
srun -n 1 flux start flux resource list

```

### 3. Application Specifics

Clone the environment and install specific machine learning libraries.

```bash
conda create --prefix /work/08405/ilgar/vista/conda_libraries/tsearch --clone executorlib

pip install fairchem-core
pip uninstall torch
# Install PyTorch with CUDA 12.8 support
pip install "torch==2.9.0+cu128" --index-url https://download.pytorch.org/whl/cu128

```

## Configuration

### Hugging Face Login

The CLI login may encounter issues. Use the Python interface instead:

```bash
python -c "import huggingface_hub; huggingface_hub.login()"

```

Alternatively, export the token directly:

```bash
export HF_TOKEN="***"

```

### Environment Variables

Add these to your `.bashrc` or run before execution to manage cache and paths:

```bash
# Move cache to scratch to save home directory space
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"

```

## Usage

### Interactive Node

Request an interactive node on the `gh-dev` partition:

```bash
idev -p gh-dev -m 120 -A CHE23004

```

### Runtime Requirements

Ensure you run these commands (or add to `.bashrc`) every time you log in or start a job:

```bash
module unload xalt
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

```