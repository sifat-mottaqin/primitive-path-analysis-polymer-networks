import os
import numpy as np


def _get_deformation_mode(fileTag):
    """
    Infer the loading mode from the CGMD/DNM file tag.

    Expected examples:
        C50_B1000_phiI50_phiT01_tension_1
        C50_B1000_phiI50_phiT01_compression_2
        C50_B1000_phiI50_phiT01_pure_shear_1
    """
    tag = str(fileTag).strip().lower()

    # Check pure_shear first because it contains an underscore.
    if "_pure_shear_" in f"_{tag}_":
        return "pure_shear"
    if "_compression_" in f"_{tag}_":
        return "compression"
    if "_tension_" in f"_{tag}_":
        return "tension"

    raise ValueError(
        f"Could not determine deformation mode from fileTag='{fileTag}'. "
        "Expected the tag to contain 'tension', 'compression', or 'pure_shear'."
    )


def _read_box_from_topology(topoFileFull):
    """
    Read x/y/z box bounds from a LAMMPS data/topology file.
    Expects lines like:
        0.000000 140.000000 xlo xhi
        0.000000 140.000000 ylo yhi
        0.000000 140.000000 zlo zhi
    """
    xlo = xhi = ylo = yhi = zlo = zhi = None

    with open(topoFileFull, "r") as f:
        for line in f:
            s = line.strip()
            if s.endswith("xlo xhi"):
                vals = s.split()
                xlo, xhi = float(vals[0]), float(vals[1])
            elif s.endswith("ylo yhi"):
                vals = s.split()
                ylo, yhi = float(vals[0]), float(vals[1])
            elif s.endswith("zlo zhi"):
                vals = s.split()
                zlo, zhi = float(vals[0]), float(vals[1])

    if None in (xlo, xhi, ylo, yhi, zlo, zhi):
        raise RuntimeError(f"Could not read box bounds from topology file: {topoFileFull}")

    return xlo, xhi, ylo, yhi, zlo, zhi


def generateInputForDNM(fileTag, mainFolder):
    """
    Writes:
        Input/input_<fileTag>_DNM.in
    Uses topology:
        Input/topology_<fileTag>_DNM.txt
    Dumps:
        Output/Dump/atoms_<fileTag>_DNM.dump
        Output/Dump/bonds_<fileTag>_DNM.dump
    """

    deformation_mode = _get_deformation_mode(fileTag)

    b      = 1
    tau0   = 1e-8
    dt     = 1e-4
    iout   = 1e4
    npull  = 1e5
    nhold  = 1e5
    nhold0 = 1e5
    eps    = 1.0
    gamma  = 1

    inpDir  = os.path.join(mainFolder, 'Input')
    outDump = os.path.join(mainFolder, 'Output', 'Dump')
    os.makedirs(inpDir, exist_ok=True)
    os.makedirs(outDump, exist_ok=True)

    topoFileFull = os.path.join(inpDir,  f'topology_{fileTag}_DNM.txt')
    inputFile    = os.path.join(inpDir,  f'input_{fileTag}_DNM.in')
    atomDumpOut  = os.path.join(outDump, f'atoms_{fileTag}_DNM.dump')
    bondDumpOut  = os.path.join(outDump, f'bonds_{fileTag}_DNM.dump')

    if not os.path.isfile(topoFileFull):
        raise RuntimeError(f"Topology file missing: {topoFileFull}")

    topFile  = os.path.basename(topoFileFull)
    atomDump = os.path.basename(atomDumpOut)
    bondDump = os.path.basename(bondDumpOut)

    xlo, xhi, ylo, yhi, zlo, zhi = _read_box_from_topology(topoFileFull)

    # Same engineering-strain magnitudes used by the corresponding CGMD.
    # tension:     lambda_x = 1 + e
    # compression: lambda_x = 1 - e
    # pure_shear:  lambda_x = 1 + e
    engStrains = np.linspace(0.0, 0.3, 20)

    Lx0 = xhi - xlo
    Ly0 = yhi - ylo
    Lz0 = zhi - zlo

    with open(inputFile, 'w') as fid:
        fid.write(f"######### Input_{fileTag}_DNM #########\n")
        fid.write(f"# deformation_mode = {deformation_mode}\n\n")

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
        fid.write("variable    T0      equal 1.00e-10\n")
        fid.write(f"variable    gamma   equal {gamma:.6g}\n\n")

        fid.write(f"variable    ithermo equal {iout:.6g}\n")
        fid.write(f"variable    iout    equal {iout:.6g}\n")
        fid.write(f"variable    npull   equal {npull:.6g}\n")
        fid.write(f"variable    nhold   equal {nhold:.6g}\n")
        fid.write(f"variable    nhold0   equal {nhold0:.6g}\n\n")

        fid.write(f"neighbor    {1.122*b:.6g} bin\n")
        fid.write(f"comm_modify cutoff  {5*b:.6g}\n")
        fid.write("neigh_modify delay 0 every 5 check yes\n\n")

        fid.write("atom_style   molecular\n")
        fid.write("bond_style   pade\n\n")

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
        fid.write("fix    free ints brownian ${T0} 6605 gamma_t ${gamma}\n\n") # rng none

        fid.write("compute  Pairs all property/local btype batom1 batom2\n")
        fid.write("compute  Bonds all bond/local dx dy dz dist force\n\n")

        fid.write(f"dump atomsdump all custom ${{iout}} {atomDump} id x y z vx vy vz type mol\n")
        fid.write(f"dump bondssdump all local  ${{iout}} {bondDump} index c_Pairs[*] c_Bonds[*]\n\n")

        # fid.write("# Virial-only global pressure/stress tensor\n")
        # fid.write("compute  pvir all pressure NULL virial\n\n")

        # fid.write("# Optional per-atom virial stress if you also want local data\n")
        # fid.write("compute  sigma all stress/atom NULL virial\n")
        # fid.write("compute  sig11 all reduce sum c_sigma[1]\n")
        # fid.write("compute  sig22 all reduce sum c_sigma[2]\n")
        # fid.write("compute  sig33 all reduce sum c_sigma[3]\n\n")

        #fid.write("# Convert summed per-atom stress*volume to stress\n")
        #fid.write("variable Sxx equal -c_sig11/vol\n")
        #fid.write("variable Syy equal -c_sig22/vol\n")
        #fid.write("variable Szz equal -c_sig33/vol\n\n")

        # fid.write("variable teq equal ${dt}*elapsed\n\n")

        # fid.write("thermo_style custom step time ebond epair vol c_pvir[1] c_pvir[2] c_pvir[3] v_Sxx v_Syy v_Szz\n")
        fid.write("thermo_style custom step time ebond epair vol press temp\n")
        fid.write("thermo          ${ithermo}\n")
        fid.write("thermo_modify   flush yes\n")
        fid.write("thermo_modify   lost warn\n\n")

        # fid.write("reset_timestep 0\n")
        # fid.write("min_style fire\n")
        # fid.write("minimize 1.0e-10 1.0e-10 10000 100000\n\n")

        fid.write("run ${npull}\n")

        # To match the CGMD
        # fid.write("run ${npull}\n")
        # fid.write("run ${nhold0}\n")

        # Mechanical loading
        # Mode is inherited directly from the CGMD file tag.   
        if deformation_mode == "tension":
            fid.write("\n# === Uniaxial tension in x ===\n")
            fid.write("# F = diag(lambda, lambda^(-1/2), lambda^(-1/2))\n")
            fid.write("# lambda = 1 + engineering_strain\n")

        elif deformation_mode == "compression":
            fid.write("\n# === Uniaxial compression in x ===\n")
            fid.write("# F = diag(lambda, lambda^(-1/2), lambda^(-1/2))\n")
            fid.write("# lambda = 1 - engineering_strain\n")

        elif deformation_mode == "pure_shear":
            fid.write("\n# === Pure shear / planar extension ===\n")
            fid.write("# F = diag(lambda, lambda^(-1), 1)\n")
            fid.write("# lambda = 1 + engineering_strain\n")

        for k, e in enumerate(engStrains, start=1):

            if deformation_mode == "tension":
                lx = 1.0 + e
                ly = 1.0 / np.sqrt(lx)
                lz = 1.0 / np.sqrt(lx)

            elif deformation_mode == "compression":
                lx = 1.0 - e
                if lx <= 0.0:
                    raise ValueError(
                        f"Compression produced lambda={lx} <= 0 "
                        f"at engineering strain {e}."
                    )
                ly = 1.0 / np.sqrt(lx)
                lz = 1.0 / np.sqrt(lx)

            else:  # pure_shear
                lx = 1.0 + e
                ly = 1.0 / lx
                lz = 1.0

            # Preserve the existing DNM box convention:
            # x lower boundary fixed; y/z changes centered.
            xlo_k = xlo
            xhi_k = xlo + Lx0 * lx

            Ly_k = Ly0 * ly
            ymid = 0.5 * (ylo + yhi)
            ylo_k = ymid - 0.5 * Ly_k
            yhi_k = ymid + 0.5 * Ly_k

            Lz_k = Lz0 * lz
            zmid = 0.5 * (zlo + zhi)
            zlo_k = zmid - 0.5 * Lz_k
            zhi_k = zmid + 0.5 * Lz_k

            J = lx * ly * lz

            fid.write(
                f"\n# ---- mode={deformation_mode}, "
                f"engineering strain={e:.6f}, "
                f"lambda_x={lx:.6f}, lambda_y={ly:.6f}, "
                f"lambda_z={lz:.6f}, J={J:.6f} ----\n"
            )
            fid.write(
                f"fix DEF{k} all deform 100 "
                f"x final {xlo_k:.6f} {xhi_k:.6f} "
                f"y final {ylo_k:.6f} {yhi_k:.6f} "
                f"z final {zlo_k:.6f} {zhi_k:.6f}\n"
            )
            fid.write("run ${npull}\n")
            fid.write(f"unfix DEF{k}\n")
            fid.write("run ${nhold}\n")

    # print(f"Input written: {inputFile}")