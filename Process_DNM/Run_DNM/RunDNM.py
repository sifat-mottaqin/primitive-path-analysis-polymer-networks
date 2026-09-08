# type: ignore

import os
import glob
import time
from pathlib import Path

# Change to script directory so relative paths work
script_dir = Path(__file__).parent
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

from generateTopologyForDNM import generateTopologyForDNM
from generateInputForDNM import generateInputForDNM
from runLammpsForDNM import runLammpsForDNM


def _detect_pairs(folder):
    """
    Return list of (vertices_path, edges_path, tag)
    from:
        vertices_<tag>.dump
        edges_<tag>.dump
    """
    v_files = sorted(glob.glob(os.path.join(folder, "vertices_*.dump")))
    pairs = []

    for vf in v_files:
        base = os.path.basename(vf)
        tag = base.replace("vertices_", "", 1).replace(".dump", "")
        ef = os.path.join(folder, f"edges_{tag}.dump")
        if os.path.isfile(ef):
            pairs.append((vf, ef, tag))
        else:
            print(f"[WARN] No matching edges file for {base}, skipping.")

    return pairs


def GenerateDump2DNM():
    # Control switches
    create_folders     = True
    generate_topology  = True
    generate_input     = True
    run_lammps         = True

    override_topology  = True
    override_input     = True
    override_lammps    = True

    # Topology controls
    merge_cutoff = 0.5
    bond_merge_mode = "keep_all"   # "keep_all", "smallest_Np", or "avg_Np"

    # LAMMPS configuration
    useGPU        = False
    parallelRun   = False
    ranks         = 4
    gpusPerTask   = 1
    srunGPUFlags  = '--gpus-per-task=1'
    ldPath        = os.environ.get('LMP_LD_PATH', '')
    lmpExtra      = ''

    if useGPU:
        lammps_exe = '/mnt/c/Users/bsifatmottaq/Documents/Sifat/gpu_build/lammps/build/lmp'
    else:
        lammps_exe = '/mnt/c/Users/bsifatmottaq/Documents/lammps_build_pade/lammps/build_mpi/lmp' #pade version

    mainFolder = os.getcwd()

    if create_folders:
        subDirs = ['Input', 'Output', 'Output/Dump', 'target_dump']
        for d in subDirs:
            path = os.path.join(mainFolder, d)
            if not os.path.isdir(path):
                os.makedirs(path)

    targetDumpDir = os.path.join(mainFolder, 'target_dump')
    inpDir        = os.path.join(mainFolder, 'Input')
    outDump       = os.path.join(mainFolder, 'Output', 'Dump')

    if not os.path.isdir(targetDumpDir):
        raise RuntimeError(f"Missing folder: {targetDumpDir}")

    pairs = _detect_pairs(targetDumpDir)
    if not pairs:
        raise RuntimeError(
            f"No vertices_*.dump / edges_*.dump pairs found in {targetDumpDir}"
        )

    total_cases = len(pairs)
    start_all = time.time()

    print('=' * 80)
    print('DNM PIPELINE: TARGET_DUMP -> TOPOLOGY -> INPUT -> LAMMPS')
    print('=' * 80)

    for idx, (vertices_path, edges_path, fileTag) in enumerate(pairs, start=1):
        print('-' * 80)
        print(f'Case {idx} of {total_cases}: {fileTag}')
        print(f'  Vertices: {vertices_path}')
        print(f'  Edges   : {edges_path}')

        topoOut = os.path.join(inpDir, f'topology_{fileTag}_DNM.txt')
        inputOut = os.path.join(inpDir, f'input_{fileTag}_DNM.in')
        atomDumpOut = os.path.join(outDump, f'atoms_{fileTag}_DNM.dump')
        bondDumpOut = os.path.join(outDump, f'bonds_{fileTag}_DNM.dump')

        # Topology
        if generate_topology:
            if os.path.isfile(topoOut):
                if override_topology:
                    print('Topology exists; override=1 -> regenerating ...')
                    generateTopologyForDNM(
                        fileTag,
                        mainFolder,
                        merge_cutoff=merge_cutoff,
                        bond_merge_mode=bond_merge_mode
                    )
                else:
                    print('Topology exists; override=0 -> skipping ...')
            else:
                print('Topology missing -> generating ...')
                generateTopologyForDNM(
                        fileTag,
                        mainFolder,
                        merge_cutoff=merge_cutoff,
                        bond_merge_mode=bond_merge_mode
                    )

        # Input
        if generate_input:
            if os.path.isfile(inputOut):
                if override_input:
                    print('Input exists; override=1 -> regenerating ...')
                    generateInputForDNM(fileTag, mainFolder)
                else:
                    print('Input exists; override=0 -> skipping ...')
            else:
                print('Input missing -> generating ...')
                generateInputForDNM(fileTag, mainFolder)

        # LAMMPS
        if run_lammps:
            atom_exists = os.path.isfile(atomDumpOut)
            bond_exists = os.path.isfile(bondDumpOut)

            if atom_exists and bond_exists:
                if override_lammps:
                    print('DNM dumps exist; override=1 -> re-running ...')
                    runLammpsForDNM(
                        lammps_exe, fileTag, mainFolder,
                        Parallel=parallelRun, Ranks=ranks,
                        UseGPU=useGPU, GPUs=gpusPerTask,
                        SrunGPU=srunGPUFlags,
                        LDPath=ldPath, Extra=lmpExtra
                    )
                else:
                    print('DNM dumps exist; override=0 -> skipping ...')
            else:
                print('DNM dumps missing -> running LAMMPS ...')
                runLammpsForDNM(
                    lammps_exe, fileTag, mainFolder,
                    Parallel=parallelRun, Ranks=ranks,
                    UseGPU=useGPU, GPUs=gpusPerTask,
                    SrunGPU=srunGPUFlags,
                    LDPath=ldPath, Extra=lmpExtra
                )

    elapsed = time.time() - start_all
    print('=' * 80)
    print('DNM PIPELINE COMPLETED')
    print('=' * 80)
    print(f'Total cases: {total_cases}')
    print(f'Total duration: {elapsed:.2f} sec')


if __name__ == "__main__":
    GenerateDump2DNM()