# Polymer network workflow

This code builds a polymer network and simulates it at two levels of detail. 
First, LAMMPS runs a coarse-grained molecular dynamics (CGMD) simulation of the polymer chains. 
The Python code then reads one saved frame, finds the entanglements, and construct primitive paths. 
This gives the discrete network model (DNM), which LAMMPS runs under the same loading mode.

**Run `runCompleteWorkflow.py`.** It calls the other scripts in order and passes the results from each step to the next.

## Set up your run

1. Open `workflow.json`. Set `cgmd_command` and `dnm_command` to the LAMMPS executables on your machine. 
2. Under `batch`, choose, adjust or add the total bead count, chain counts, and initial volume fractions. 
3. Under `batch.sweep_plan`, choose the target volume fractions and how many simulations to run.

```json
{"phi_target": 0.05, "tension": 3, "compression": 1, "pure_shear": 0}
```

The driver calculates the box size from the initial and target volume fractions. Both models use the strain levels specified by `strain_grid`. Here, the supplied settings give 20 evenly spaced levels from 0 to 0.3. You may change it as necessary. 

## Run the workflow

Open a terminal in the extracted `Workflow` folder. Install the Python packages once:

```bash
python -m pip install -r requirements.txt
```

Preview the cases, then run them:

```bash
python runCompleteWorkflow.py --list-cases
python runCompleteWorkflow.py
```

The supplied batch runs one tension case at a target volume fraction of 0.02. By default, `distillation.timestep` is `"undeformed"`. 
The driver reads the CGMD box dimensions and selects the last saved frame before loading changes the cubic box shape. An integer selects a particular saved timestep, and `null` selects the final frame of the trajectory.

## Where to find the results

Each case has its own folder under `Results/`. Inside that folder:

| Output | Location |
| --- | --- |
| CGMD atom and bond trajectories | `CGMD/Output/Dump/` |
| Distilled vertices and edges | `DNM/target_dump/` |
| DNM topology and LAMMPS input | `DNM/Input/` |
| DNM atom and bond trajectories | `DNM/Output/Dump/` |
| Tables and plots for each model | `CGMD/Output/ProcessedData/` and `DNM/Output/ProcessedData/` |

`Results/batch_summary.csv` lists which cases completed and any errors.

To run the Python checks without starting LAMMPS, use `python -m unittest testWorkflow`.

