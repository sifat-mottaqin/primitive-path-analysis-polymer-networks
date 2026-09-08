# type: ignore

import os
import sys
import numpy as np
import time
import random
import pickle
import subprocess
from pathlib import Path

# Change to script directory so relative paths work
script_dir = Path(__file__).parent
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}")

from generateSAWCordinates import generateSAWCordinates
from generateTopologyFromSAW import generateTopologyFromSAW
from generateInputFromSAW import generateInputFromSAW
from runLammpsFromSAW import runLammpsFromSAW
#from extractDumpData import extractDumpData
#from runPostProcess import runPostProcess

def runSAWNetwork():
    # Parameter Sweep Configuration
    total_beads_list = [50000]  # Total number of beads in the system
    phi = 0.5
    chains_list = [50]
    num_runs = 1

    # Control switches
    create_folders        = True
    gen_SAW_network       = True
    generate_topology     = True
    generate_input        = True
    run_lammps            = True
    extract_data          = False
    run_postprocess       = False

    override_SAW_network  = False
    override_topology     = False
    override_input        = False
    override_lammps       = False
    override_extract      = False
    override_postprocess  = False

    # LAMMPS Configuration
    useGPU        = False
    parallelRun   = True
    ranks         = 16
    gpusPerTask   = 1
    srunGPUFlags  = '--gpus-per-task=1'
    ldPath        = os.environ.get('LMP_LD_PATH', '')
    lmpExtra      = ''

    if useGPU:
        lammps_exe = '/mnt/c/Users/bsifatmottaq/Documents/Sifat/gpu_build/lammps/build/lmp'
    else:
        lammps_exe = '/mnt/c/Users/bsifatmottaq/Documents/LAMMPS/CPU_PARA_BUILD/lammps/build_mpi/lmp'

    mainFolder = os.getcwd()

    if create_folders:
        subDirs = ['Output','Input','Coordinates','Figures','Output/Dump','Output/RawData', 'Output/ProcessedData']
        for d in subDirs:
            path = os.path.join(mainFolder, d)
            if not os.path.isdir(path):
                os.makedirs(path)

    inpDir   = os.path.join(mainFolder, 'Input')
    outDump  = os.path.join(mainFolder, 'Output','Dump')
    outRaw   = os.path.join(mainFolder, 'Output','RawData')
    outProc  = os.path.join(mainFolder, 'Output','ProcessedData')
    coordDir = os.path.join(mainFolder, 'Coordinates')
    figDir   = os.path.join(mainFolder, 'Figures')

    # Initialize Sweep Tracking
    total_combinations = len(total_beads_list) * len(chains_list) * num_runs
    sweep_log = []
    sweep_counter = 0
    overall_start = time.time()
    
    # Track which cases need post-processing
    cases_for_postprocess = []
    
    # PHASE 1 : NETWORK GENERATION
    print('='*80)
    print('PHASE 1: GENERATING SAW NETWORKS, TOPOLOGY, INPUT, AND RUNNING LAMMPS')
    print('='*80)

    for total_beads in total_beads_list:
        for c_idx, no_chains in enumerate(chains_list):
            for run in range(1, num_runs+1):
                no_beads = int(np.ceil(total_beads / no_chains))
                random_seed = 1710025 + (c_idx * 100) + run
                np.random.seed(random_seed)
                random.seed(random_seed)

                sweep_counter += 1
                fileTag  = f'C{no_chains}_B{no_beads}_phi{int(phi*100)}_{run}'
                coordOut = os.path.join(coordDir, f'coords_{fileTag}.pkl')
                topoOut  = os.path.join(inpDir, f'topology_{fileTag}.txt')
                inputOut = os.path.join(inpDir, f'input_{fileTag}.in')
                atomDumpOut = os.path.join(outDump, f'atoms_{fileTag}.dump')
                bondDumpOut = os.path.join(outDump, f'bonds_{fileTag}.dump')
                extractOut = os.path.join(outRaw, f'raw_{fileTag}_ID.pkl')

                run_start = time.time()
                print('-'*80)
                print(f'Run {sweep_counter} of {total_combinations}: C={no_chains}, B={no_beads} (per chain), phi={phi:.1f}, iteration={run}')
                print(f'Total beads: {no_chains * no_beads} | Random Seed: {random_seed}')

                try:
                    # SAW
                    if gen_SAW_network:
                        if os.path.isfile(coordOut):
                            if override_SAW_network:
                                print('SAW coordinates exist; override=1 -> regenerating ...')
                                generateSAWCordinates(no_chains, no_beads, phi, fileTag, mainFolder)
                            else:
                                print('SAW coordinates exist; override=0 -> skipping ...')
                        else:
                            print('SAW coordinates missing -> generating ...')
                            generateSAWCordinates(no_chains, no_beads, phi, fileTag, mainFolder)

                    # Load coords
                    coordsOK = os.path.isfile(coordOut)
                    if coordsOK:
                        with open(coordOut, 'rb') as f:
                            S = pickle.load(f)
                    else:
                        print('Coordinates missing -> nothing else can run.')
                        coordsOK = False

                    # Topology
                    if coordsOK and generate_topology:
                        if os.path.isfile(topoOut):
                            if override_topology:
                                print('Topology exists; override=1 -> regenerating ...')
                                generateTopologyFromSAW(S, fileTag, mainFolder)
                            else:
                                print('Topology exists; override=0 -> skipping ...')
                        else:
                            print('Topology missing -> generating ...')
                            generateTopologyFromSAW(S, fileTag, mainFolder)

                    # Input
                    if coordsOK and generate_input:
                        if os.path.isfile(inputOut):
                            if override_input:
                                print('Input exists; override=1 -> regenerating ...')
                                generateInputFromSAW(S, fileTag, mainFolder)
                            else:
                                print('Input exists; override=0 -> skipping ...')
                        else:
                            print('Input missing -> generating ...')
                            generateInputFromSAW(S, fileTag, mainFolder)

                    # LAMMPS
                    atom_file_check = os.path.isfile(atomDumpOut)
                    bond_file_check = os.path.isfile(bondDumpOut)
                    ok = True
                    
                    if run_lammps:
                        if atom_file_check and bond_file_check:
                            if override_lammps:
                                print('Dumps exist; override=1 -> re-running ...')
                                ok = runLammpsFromSAW(
                                    lammps_exe, fileTag, mainFolder,
                                    Parallel=parallelRun, Ranks=ranks,
                                    UseGPU=useGPU, GPUs=gpusPerTask,
                                    SrunGPU=srunGPUFlags,
                                    LDPath=ldPath, Extra=lmpExtra)
                            else:
                                print('Dumps exist; override=0 -> skipping ...')
                                ok = True
                        else:
                            print('Dumps missing -> running LAMMPS ...')
                            ok = runLammpsFromSAW(
                                lammps_exe, fileTag, mainFolder,
                                Parallel=parallelRun, Ranks=ranks,
                                UseGPU=useGPU, GPUs=gpusPerTask,
                                SrunGPU=srunGPUFlags,
                                LDPath=ldPath, Extra=lmpExtra)
                    else:
                        ok = atom_file_check and bond_file_check

                    # Extract data
                    extract_ok = True
                    if extract_data and ok:
                        if os.path.isfile(extractOut):
                            if override_extract:
                                print('Extract file exists; override=1 -> extracting again ...')
                                extractDumpData(atomDumpOut, bondDumpOut, extractOut)
                            else:
                                print('Extract file exists; override=0 -> skipping extract ...')
                        else:
                            print('Extract file missing -> running extraction ...')
                            extractDumpData(atomDumpOut, bondDumpOut, extractOut)
                        extract_ok = os.path.isfile(extractOut)

                    run_duration = time.time() - run_start
                    status = ok and coordsOK and extract_ok

                    sweep_log.append({
                        'run': sweep_counter,
                        'no_chains': no_chains,
                        'no_beads_per_chain': no_beads,
                        'total_beads': no_chains * no_beads,
                        'phi': phi,
                        'iteration': run,
                        'random_seed': random_seed,
                        'fileTag': fileTag,
                        'status': status,
                        'duration_sec': run_duration,
                        'postprocess_status': None
                    })

                    # Queue for post-processing if successful - INCLUDE fileTag
                    if extract_ok and run_postprocess:
                        cases_for_postprocess.append({
                            'extractOut': extractOut,
                            'outProc': outProc,
                            'fileTag': fileTag
                        })

                    print(f'Run completed: Status={status}, Duration={run_duration:.2f} sec')

                except Exception as e:
                    print(f'ERROR in run {sweep_counter}: {str(e)}')
                    import traceback
                    traceback.print_exc()
                    sweep_log.append({
                        'run': sweep_counter,
                        'no_chains': no_chains,
                        'no_beads_per_chain': no_beads,
                        'total_beads': no_chains * no_beads,
                        'phi': phi,
                        'iteration': run,
                        'random_seed': random_seed,
                        'fileTag': fileTag,
                        'status': False,
                        'duration_sec': time.time() - run_start,
                        'postprocess_status': None
                    })

    # PHASE 2: POST-PROCESSING 
    if run_postprocess and cases_for_postprocess:
        print('\n' + '='*80)
        print(f'PHASE 2: POST-PROCESSING ({len(cases_for_postprocess)} CASES)')
        print('='*80)
        
        postprocess_start = time.time()
        
        for idx, case in enumerate(cases_for_postprocess, 1):
            try:
                extractOut = case['extractOut']
                outProc = case['outProc']
                fileTag = case['fileTag']
                
                print(f'[{idx}/{len(cases_for_postprocess)}] Processing {fileTag}...')
                runPostProcess(extractOut, outProc, fileTag)
                
                # Update sweep_log
                for entry in sweep_log:
                    if entry['fileTag'] == fileTag:
                        entry['postprocess_status'] = True
                        break
                
            except Exception as e:
                print(f'ERROR in post-processing: {str(e)}')
                import traceback
                traceback.print_exc()
                for entry in sweep_log:
                    if entry['fileTag'] == fileTag:
                        entry['postprocess_status'] = False
                        break
        
        postprocess_duration = time.time() - postprocess_start
        print(f'\nAll post-processing completed in {postprocess_duration:.2f} sec')
    
    # SUMMARY 
    overall_duration = time.time() - overall_start
    print('\n' + '='*80)
    print('SWEEP COMPLETED')
    print('='*80)
    print(f'Phase 1 (SAW+LAMMPS+Extract): Completed')
    if run_postprocess and cases_for_postprocess:
        print(f'Phase 2 (Post-Processing): Completed ({len(cases_for_postprocess)} cases)')
    print(f'\nTotal runs: {sweep_counter}')
    print(f'Total duration: {overall_duration:.2f} sec ({overall_duration/60:.2f} minutes)')
    print(f'Average time per run: {overall_duration / sweep_counter:.2f} sec')
    print('\nSummary:')
    print('-'*80)
    
    passed = 0
    failed = 0
    postprocess_done = 0
    for entry in sweep_log:
        status_str = 'PASS' if entry['status'] else 'FAIL'
        postprocess_str = ''
        if entry['postprocess_status'] is True:
            postprocess_str = ' [POST OK]'
            postprocess_done += 1
        elif entry['postprocess_status'] is False:
            postprocess_str = ' [POST FAIL]'
        
        if entry['status']:
            passed += 1
        else:
            failed += 1
        print(f"Run {entry['run']:2d}: C={entry['no_chains']:2d}, B={entry['no_beads_per_chain']:6d}, "
              f"iteration={entry['iteration']}, seed={entry['random_seed']:4d} - {status_str} ({entry['duration_sec']:7.2f} sec){postprocess_str}")
    
    print('-'*80)
    print(f'Phase 1 - Passed: {passed}/{sweep_counter}')
    print(f'Phase 1 - Failed: {failed}/{sweep_counter}')
    if run_postprocess:
        print(f'Phase 2 - Post-Processing: {postprocess_done}/{len(cases_for_postprocess)} completed')
    print('='*80)

if __name__ == "__main__":
    runSAWNetwork()