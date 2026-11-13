# Source code for "Physically-motivated primitive path analysis of entangled polymer networks"

DOI: https://doi.org/10.1103/2bvb-fnrb

This repository contains the network-generation, topology-distillation and simulation scripts, together with the single-chain and two-chain study.

## Folder contents

| Folder | Contents | Starting point |
| --- | --- | --- |
| `Complete_Workflow` | Connected Python workflow: (1) generate a CGMD network, (2) run CGMD, (3) identify entanglements and primitive paths, (4) generate and run the DNM, and (5) export results. | `RunCompleteWorkflow.py` (see its own README.md) |
| `Generate_CGMD` |  Python network generator and tension simulation. | `RunSAWNetwork.py` |
| `Generate_CGMD_Batch` | MATLAB network generator and Slurm batch workflow, with volume-fraction and loading-mode sweeps. Supports tension, compression and pure shear. | `RunSAWNetwork.m`, `run_Gen_CGMD_batch.sh` |
| `Process_DNM/Gen_DNM` | Reads a selected CGMD dump frame, detects entanglements, and writes primitive-path vertices and edges. | `RunProcessDump2DNM.py` |
| `Process_DNM/Run_DNM` | Converts primitive-path vertices and edges to a LAMMPS DNM topology and runs mechanical loading. | `RunDNM.py` |
| `Single_Chain` | MATLAB single-chain simulations, extraction, and force–extension plots. | `ParallelRunScript.m` |
| `TwoChains` | MATLAB two-chain generation, simulations, linking-number/COE calculations, and dump extraction. | `RunTwoChains.m` |

## Running the complete workflow

1. For the complete workflow, follow `Complete_Workflow/README.md`.

2. For standalone distillation, create `training_data` under a working directory and place matching `atoms_<tag>.dump` and `bonds_<tag>.dump` files there. Then run:

```bash
python Process_DNM/Gen_DNM/RunProcessDump2DNM.py /path/to/working_directory 113000000
```

The script writes `train_dump` (the selected CGMD frame with the entanglement field) and `target_dump` (primitive-path vertices and edges). 
The optional timestep argument can be `last`; the default is 113000000. Choose a frame appropriate to the actual loading schedule.

For standalone DNM runs, place the generated `target_dump` folder inside `Process_DNM/Run_DNM`, edit the controls in `RunDNM.py`, and run that driver. 
Loading mode is read from the tag: `tension`, `compression`, `pure_shear`, or the batch alias `shear`.

For MATLAB calculations, open MATLAB in the corresponding folder and edit the main driver's controls and executable paths. The Slurm files are cluster-specific examples, not portable job submissions.
