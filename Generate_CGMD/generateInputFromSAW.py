import os
import numpy as np


def generateInputFromSAW(S, fileTag, mainFolder):
    """
    Writes:
        Input/input_<fileTag>.in
    Uses topology:
        Input/topology_<fileTag>.txt
    Dumps:
        Output/Dump/atoms_<fileTag>.dump
        Output/Dump/bonds_<fileTag>.dump
    """

    b      = 1
    tau0   = 1e-8
    dt     = 1e-5
    iout   = 1e6
    npull  = 1e6
    nhold  = 1e7
    nhold0 = 1e8
    eps    = 1.0
    gamma  = 1
    fact   = 5.0

    engStrains = np.linspace(0.0, 0.3, 20)

    if 'meta' not in S:
        raise RuntimeError("S['meta'] missing.")

    meta = S['meta']
    L    = meta['L_grid']

    xlo = 0.0; xhi = float(L)
    ylo = 0.0; yhi = float(L)
    zlo = 0.0; zhi = float(L)
 
    # Symmetric isotropic expansion about the original box center
    xmid = 0.5 * (xlo + xhi)
    ymid = 0.5 * (ylo + yhi)
    zmid = 0.5 * (zlo + zhi)

    Lx_final = fact * (xhi - xlo)
    Ly_final = fact * (yhi - ylo)
    Lz_final = fact * (zhi - zlo)

    xlo_final = xmid - 0.5 * Lx_final
    xhi_final = xmid + 0.5 * Lx_final

    ylo_final = ymid - 0.5 * Ly_final
    yhi_final = ymid + 0.5 * Ly_final

    zlo_final = zmid - 0.5 * Lz_final
    zhi_final = zmid + 0.5 * Lz_final

    inpDir  = os.path.join(mainFolder, 'Input')
    outDump = os.path.join(mainFolder, 'Output', 'Dump')
    os.makedirs(inpDir, exist_ok=True)
    os.makedirs(outDump, exist_ok=True)

    topoFileFull = os.path.join(inpDir,  f'topology_{fileTag}.txt')
    inputFile    = os.path.join(inpDir,  f'input_{fileTag}.in')
    atomDumpOut  = os.path.join(outDump, f'atoms_{fileTag}.dump')
    bondDumpOut  = os.path.join(outDump, f'bonds_{fileTag}.dump')

    if not os.path.isfile(topoFileFull):
        raise RuntimeError(f"Topology file missing: {topoFileFull}")

    topFile  = os.path.basename(topoFileFull)
    atomDump = os.path.basename(atomDumpOut)
    bondDump = os.path.basename(bondDumpOut)

    # Reference lengths after symmetric expansion
    Lx0 = xhi_final - xlo_final
    Ly0 = yhi_final - ylo_final
    Lz0 = zhi_final - zlo_final

    # Centers after expansion
    y_mid_exp = 0.5 * (ylo_final + yhi_final)
    z_mid_exp = 0.5 * (zlo_final + zhi_final)

    with open(inputFile, 'w') as fid:
        fid.write(f"######### Input_{fileTag} #########\n\n")

        fid.write("units        lj\n")
        fid.write("boundary     p p p\n\n")

        fid.write("variable    kb      equal 1.380e-23\n")
        fid.write("variable    T       equal 1.000\n")
        fid.write("variable    kbT     equal 1.000\n\n")

        fid.write(f"variable    b       equal {b:.6g}\n\n")

        fid.write(f"variable    eps_11  equal {eps:.6g}\n")
        fid.write(f"variable    eps_22  equal {eps:.6g}\n")
        fid.write(f"variable    eps_12  equal {eps:.6g}\n")
        fid.write(f"variable    sig_LJ  equal {b:.6g}\n")
        fid.write(f"variable    sig_cut equal {1.122*b:.6g}\n\n")

        fid.write(f"variable    tau0    equal {tau0:.6g}\n")
        fid.write(f"variable    dt      equal {dt:.6g}\n")
        fid.write("variable    T0      equal 1.00\n")
        fid.write(f"variable    gamma   equal {gamma:.6g}\n\n")

        fid.write(f"variable    ithermo equal {iout*10:.6g}\n")
        fid.write(f"variable    iout    equal {iout:.6g}\n")
        fid.write(f"variable    npull   equal {npull:.6g}\n")
        fid.write(f"variable    nhold   equal {nhold:.6g}\n")
        fid.write(f"variable    nhold0  equal {nhold0:.6g}\n\n")

        fid.write(f"neighbor    {1.122*b:.6g} bin\n")
        fid.write(f"comm_modify cutoff  {5*b:.6g}\n")
        fid.write("neigh_modify delay 0 every 5 check yes\n\n")

        fid.write("atom_style   molecular\n")
        fid.write("bond_style   harmonic\n\n")

        fid.write(f"read_data {topFile} &\n")
        fid.write(" extra/bond/per/atom 1 extra/special/per/atom 10\n")
        fid.write("neigh_modify one 2000\n\n")

        fid.write("pair_style   lj/cut ${sig_cut}\n")
        fid.write("pair_coeff   1 1 ${eps_11} ${sig_LJ}\n")
        fid.write("pair_coeff   2 2 ${eps_22} ${sig_LJ}\n")
        fid.write("pair_coeff   1 2 ${eps_12} ${sig_LJ}\n")
        fid.write("special_bonds lj 0.0 1.0 1.0\n\n")

        fid.write("timestep ${dt}\n\n")

        fid.write("group  ends type 1\n")
        fid.write("group  ints type 2\n")
        fid.write("fix    free ints brownian ${T0} 6605 gamma_t ${gamma}\n\n")

        fid.write("compute  Pairs all property/local btype batom1 batom2\n")
        fid.write("compute  Bonds all bond/local dx dy dz dist force\n\n")

        fid.write(f"dump atomsdump all custom ${{iout}} {atomDump} id x y z vx vy vz type mol\n")
        fid.write(f"dump bondsdump all local  ${{iout}} {bondDump} index c_Pairs[*] c_Bonds[*]\n\n")

        fid.write("thermo_style  custom step time ebond epair vol press temp\n")
        fid.write("thermo        ${ithermo}\n")
        fid.write("thermo_modify flush yes\n")
        fid.write("thermo_modify lost warn\n\n")

        fid.write("timer full\n\n")

        fid.write("run ${npull}\n\n")
  
        # Symmetric hydrostatic expansion  
        fid.write("# === Symmetric expansion section ===\n")
        fid.write(
            f"fix EXPAND all deform 100 "
            f"x final {xlo_final:.6f} {xhi_final:.6f} "
            f"y final {ylo_final:.6f} {yhi_final:.6f} "
            f"z final {zlo_final:.6f} {zhi_final:.6f}\n"
        )
        fid.write("run ${npull}\n\n")
        fid.write("unfix EXPAND\n")
        fid.write("run ${nhold0}\n")

        
        # Uniaxial tension:
        # xlo fixed, xhi moves.
        # y and z contract symmetrically about their expanded centers.
        
        fid.write("\n# === Uniaxial tension in x after symmetric expansion ===\n")

        for k, e in enumerate(engStrains, start=1):
            lx = 1.0 + e
            ly = 1.0 / np.sqrt(lx)
            lz = 1.0 / np.sqrt(lx)

            # Keep one side of x fixed during tension
            xlo_k = xlo_final
            xhi_k = xlo_final + Lx0 * lx

            # Symmetric lateral contraction
            Ly_k = Ly0 * ly
            Lz_k = Lz0 * lz

            ylo_k = y_mid_exp - 0.5 * Ly_k
            yhi_k = y_mid_exp + 0.5 * Ly_k

            zlo_k = z_mid_exp - 0.5 * Lz_k
            zhi_k = z_mid_exp + 0.5 * Lz_k

            fid.write(f"\n# engineering strain = {e:.6f}, lambda = {lx:.6f} \n")
            fid.write(
                f"fix DEF{k} all deform 100 "
                f"x final {xlo_k:.6f} {xhi_k:.6f} "
                f"y final {ylo_k:.6f} {yhi_k:.6f} "
                f"z final {zlo_k:.6f} {zhi_k:.6f}\n"
            )
            fid.write("run ${npull}\n")
            fid.write(f"unfix DEF{k}\n")
            fid.write("run ${nhold}\n")