import os
import subprocess
from shutil import move
import platform

def runLammpsFromSAW(lammps_exe, fileTag, mainFolder, Parallel=True, Ranks=8, UseGPU=False, GPUs=1, SrunGPU='', Extra='', LDPath=''):
    inSlurm = 'SLURM_JOB_ID' in os.environ
    ok = False

    gpuFlags = ''
    if UseGPU:
        gpuFlags = f'-sf gpu -pk gpu {GPUs} '

    inpDir = os.path.join(mainFolder, 'Input')
    outDir = os.path.join(mainFolder, 'Output', 'Dump')
    os.makedirs(outDir, exist_ok=True)
    inpName = f'input_{fileTag}.in'
    inpPath = os.path.join(inpDir, inpName)
    atomDumpInRun = os.path.join(inpDir, f'atoms_{fileTag}.dump')
    bondDumpInRun = os.path.join(inpDir, f'bonds_{fileTag}.dump')
    logFile = os.path.join(inpDir, 'log.lammps')

    def cleanup_partials():
        safeDelete(atomDumpInRun)
        safeDelete(bondDumpInRun)
        safeDelete(logFile)

    isWindows = platform.system() == 'Windows'

    if not os.path.isfile(inpPath):
        print(f'Skipping – input file not found: {inpPath}')
        return False

    if isWindows:
        linuxDir = win2lin(inpDir)
        core = f'{lammps_exe} {gpuFlags}-in {inpName} {Extra}'
        if Parallel and Ranks > 1:
            core = f'mpirun -np {Ranks} {core}'
        cmd = f'wsl bash -c "cd {linuxDir} && {core}"'
    else:
        if inSlurm:
            if Parallel and Ranks>1:
                if not LDPath:
                    cmd = f'cd "{inpDir}" && srun --ntasks={Ranks} {SrunGPU} {lammps_exe} {gpuFlags}-in {inpName} {Extra}'
                else:
                    cmd = f'cd "{inpDir}" && srun --ntasks={Ranks} {SrunGPU} env LD_LIBRARY_PATH="{LDPath}" {lammps_exe} {gpuFlags}-in {inpName} {Extra}'
            else:
                if not LDPath:
                    cmd = f'cd "{inpDir}" && {lammps_exe} {gpuFlags}-in {inpName} {Extra}'
                else:
                    cmd = f'cd "{inpDir}" && env LD_LIBRARY_PATH="{LDPath}" {lammps_exe} {gpuFlags}-in {inpName} {Extra}'
        else:
            core = f'{lammps_exe} {gpuFlags}-in {inpName} {Extra}'
            if Parallel and Ranks > 1:
                core = f'mpirun -np {Ranks} {core}'
            if LDPath:
                cmd = f'cd "{inpDir}" && env LD_LIBRARY_PATH="{LDPath}" {core}'
            else:
                cmd = f'cd "{inpDir}" && {core}'

    # Suppress LAMMPS output - redirect to DEVNULL
    with open(os.devnull, 'w') as devnull:
        status = subprocess.call(cmd, shell=True, stdout=devnull, stderr=devnull)
    
    if status != 0:
        print(f'LAMMPS failed for {fileTag} (status={status})')
        cleanup_partials()
        return False

    if not(os.path.isfile(atomDumpInRun)) or not(os.path.isfile(bondDumpInRun)):
        print(f'LAMMPS ended without required dump files for {fileTag} (atoms={os.path.isfile(atomDumpInRun)}, bonds={os.path.isfile(bondDumpInRun)})')
        cleanup_partials()
        return False

    move(atomDumpInRun, outDir)
    move(bondDumpInRun, outDir)
    if os.path.isfile(logFile):
        os.remove(logFile)
    print(f'Simulation {fileTag} finished (dumps moved).')
    ok = True
    return ok

def win2lin(win_path):
    r"""
    Convert Windows path to WSL path.
    """
    linux_path = win_path.replace('\\', '/')
    if len(linux_path) >= 2 and linux_path[1] == ':':
        drive = linux_path[0].lower()
        rest = linux_path[2:]
        if rest.startswith('/'):
            rest = rest[1:]
        linux_path = f'/mnt/{drive}/{rest}'
    return linux_path

def safeDelete(p):
    if os.path.isfile(p):
        os.remove(p)