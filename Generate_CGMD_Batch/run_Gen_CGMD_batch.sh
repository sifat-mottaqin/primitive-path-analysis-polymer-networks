#!/bin/bash
#SBATCH -J ent_study
#SBATCH -p wagnerresearch
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH -t 72:00:00
#SBATCH --output=/data/home/bsifatmottaq/2026_Ent_Study/sweep_%j.out
#SBATCH --error=/data/home/bsifatmottaq/2026_Ent_Study/sweep_%j.err

WORK_DIR="/data/home/bsifatmottaq/2026_Ent_Study"
LAMMPS_EXE="/data/home/bsifatmottaq/lammps_build/build_cpu/lmp"

mkdir -p "$WORK_DIR/Output/Logs"
cd "$WORK_DIR" || exit 1

echo "========================================================================"
echo "2026 Entanglement Study - MPI LAMMPS job"
echo "========================================================================"
echo "Job ID          : ${SLURM_JOB_ID:-unknown}"
echo "Host            : $(hostname)"
echo "Start time      : $(date)"
echo "Working dir     : $(pwd)"
echo "SLURM tasks     : ${SLURM_NTASKS:-unknown}"
echo "LAMMPS exe      : $LAMMPS_EXE"
echo "========================================================================"

# 1. Load the compiler + OpenMPI environment used by the LAMMPS build.
#    IMPORTANT: this happens BEFORE MATLAB is loaded.
module purge
module load gnu13 openmpi5

MPI_LAUNCHER="$(command -v mpirun)"
if [ -z "$MPI_LAUNCHER" ]; then
    echo "ERROR: mpirun was not found after loading gnu13/openmpi5."
    exit 10
fi

if [ ! -x "$LAMMPS_EXE" ]; then
    echo "ERROR: LAMMPS executable does not exist or is not executable:"
    echo "       $LAMMPS_EXE"
    exit 11
fi

# 2. Verify all shared libraries resolve NOW, before MATLAB can alter the
#    environment. This catches the old libucp.so.0 problem immediately.
echo
echo "[PRECHECK] MPI launcher: $MPI_LAUNCHER"
"$MPI_LAUNCHER" --version | head -n 4

echo
echo "[PRECHECK] Checking LAMMPS shared libraries ..."
LDD_OUT="$(ldd "$LAMMPS_EXE" 2>&1)"
echo "$LDD_OUT" | grep -E 'libmpi|libucp|libpmix|not found' || true

if echo "$LDD_OUT" | grep -q 'not found'; then
    echo "ERROR: LAMMPS has unresolved shared libraries BEFORE MATLAB starts."
    echo "$LDD_OUT" | grep 'not found'
    exit 12
fi

# This package is intentionally MPI-only. Refuse to continue if the LAMMPS
# binary is not dynamically linked against libmpi.
if ! echo "$LDD_OUT" | grep -q 'libmpi'; then
    echo "ERROR: This LAMMPS executable does not appear to be MPI-linked."
    echo "       Expected libmpi in: ldd $LAMMPS_EXE"
    exit 13
fi

echo "[PRECHECK] LAMMPS binary is MPI-linked and all libraries resolve."

# 3. Run an ACTUAL 2-rank LAMMPS MPI smoke test inside this SLURM job.
#    If this fails, MATLAB is never started.
SMOKE_SCREEN="$WORK_DIR/Output/Logs/mpi_smoke_${SLURM_JOB_ID}.txt"
echo
echo "[PRECHECK] Running 2-rank LAMMPS MPI smoke test ..."
"$MPI_LAUNCHER" -np 2 "$LAMMPS_EXE" \
    -in "$WORK_DIR/mpi_smoke.in" \
    -log none \
    -screen "$SMOKE_SCREEN"
SMOKE_STATUS=$?

if [ $SMOKE_STATUS -ne 0 ]; then
    echo "ERROR: 2-rank LAMMPS MPI smoke test failed with status $SMOKE_STATUS."
    echo "       See $SMOKE_SCREEN"
    exit 14
fi

echo "[PRECHECK] MPI smoke test PASSED."
grep -E 'MPI processor|MPI task|OpenMP thread' "$SMOKE_SCREEN" || true

# 4. Save the CLEAN MPI environment for MATLAB -> LAMMPS launches.
#    MATLAB changes library search paths internally, so the MATLAB wrapper
#    restores these exact values before every mpirun call.
export LAMMPS_EXE
export LAMMPS_MPIRUN="$MPI_LAUNCHER"
export LAMMPS_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PATH_BEFORE_MATLAB="$PATH"
export LAMMPS_RANKS="${SLURM_NTASKS:-32}"

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export MKL_NUM_THREADS=1

echo
echo "[MPI] Production ranks: $LAMMPS_RANKS"
echo "[MPI] Production command will use: $LAMMPS_MPIRUN -np $LAMMPS_RANKS ..."

# 5. Load MATLAB only after the MPI environment has been saved.
module load MATLAB

echo
echo "Starting MATLAB workflow ..."
matlab -batch "RunSAWNetwork" -logfile "sweep_matlab_${SLURM_JOB_ID}.log"
MATLAB_STATUS=$?

echo "========================================================================"
echo "MATLAB status : $MATLAB_STATUS"
echo "End time      : $(date)"
echo "========================================================================"

if [ $MATLAB_STATUS -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "ERROR: MATLAB/workflow failed with status $MATLAB_STATUS."
fi

exit $MATLAB_STATUS
