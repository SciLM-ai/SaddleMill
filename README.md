# tsearch

# Installation
module reset
module unload xalt

conda create -n executorlib -c conda-forge python=3.12 flux-core flux-sched "openmpi=5.0.5=external_*" executorlib
conda activate executorlib
conda install "libhwloc=*=cuda*" -c conda-forge
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

# If you need mpi4py use this line but this is not tested, just copy paste straight from gemini:
MPICC=$(which mpicc) pip install --no-binary=mpi4py mpi4py

# To verify if flux sees all resources
srun -n 1 flux start flux resource list



conda create --prefix /work/08405/ilgar/vista/conda_libraries/tsearch --clone executorlib

pip install fairchem-core
pip uninstall torch
pip install "torch==2.9.0+cu128" --index-url https://download.pytorch.org/whl/cu128

python -c "import huggingface_hub; huggingface_hub.login()"
or add this to .bashrc
export HF_TOKEN="***"
also recommend adding this to not fill up home directory:
export FAIRCHEM_CACHE_DIR="$SCRATCH/.cache/fairchem"
# instead of this: huggingface-cli login





idev -p gh-dev -m 120 -A CHE23004

# DO this everytim when using the environment
module unload xalt
export LD_LIBRARY_PATH=/opt/apps/cuda/12.4/targets/sbsa-linux/lib/:$LD_LIBRARY_PATH

python
```
import numpy as np
from ase.build import bulk, make_supercell
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-m-1p1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omat")

atoms = bulk('C', 'diamond', a=3.57)
atoms = make_supercell(atoms, np.diag([2,2,2]))
atoms.positions += np.random.random((16,3))
atoms.calc = calc

opt = FIRE(FrechetCellFilter(atoms))
opt.run(0.0001, 1000)
```





# To use catsunami:
pip install fairchem-data-oc