"""
LAMMPS dump file loader and writer.

Reads only the LAST timestep from multi-frame dump files (matching the MATLAB
``load_last_timestep_atoms`` / ``load_last_timestep_bonds`` behaviour).

Writers produce LAMMPS-compatible dump text files.
"""

import numpy as np


# LOADERS

def load_last_timestep_atoms(filename):
    """
    Load the last timestep from a LAMMPS atoms dump file.

    Returns
    -------
    atoms_data  : ndarray (num_atoms, n_cols)
    header_line : str   – the "ITEM: ATOMS …" line (a Python string)
    num_atoms   : int
    box_bounds  : ndarray (3, 2)
    timestep    : int
    """
    # First pass: find byte offset of the last "ITEM: TIMESTEP"
    last_pos = -1
    with open(filename, "r") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                last_pos = pos

    if last_pos < 0:
        raise RuntimeError(f"No ITEM: TIMESTEP found in {filename}")

    # Second pass: read that timestep
    with open(filename, "r") as f:
        f.seek(last_pos)
        f.readline()                              # ITEM: TIMESTEP
        timestep = int(f.readline().strip())
        f.readline()                              # ITEM: NUMBER OF ATOMS
        num_atoms = int(f.readline().strip())
        f.readline()                              # ITEM: BOX BOUNDS ...
        box_bounds = np.zeros((3, 2))
        for b in range(3):
            vals = f.readline().strip().split()
            box_bounds[b, 0] = float(vals[0])
            box_bounds[b, 1] = float(vals[1])

        # ---- THIS IS THE KEY FIX: keep header as a plain Python str ----
        header_line = str(f.readline().strip())   # "ITEM: ATOMS id type x y z ..."

        # Determine number of data columns from the header
        hdr_tokens = header_line.replace("ITEM: ATOMS", "").strip().split()
        n_cols = len(hdr_tokens)

        data = np.zeros((num_atoms, n_cols))
        for i in range(num_atoms):
            row_line = f.readline()
            if not row_line:
                break
            vals = row_line.strip().split()
            for j in range(min(len(vals), n_cols)):
                data[i, j] = float(vals[j])

    return data, header_line, num_atoms, box_bounds, timestep


def load_last_timestep_bonds(filename):
    """
    Load the last timestep from a LAMMPS bonds dump file.

    Returns
    -------
    bonds_data  : ndarray (num_entries, n_cols)
    header_line : str
    num_entries : int
    box_bounds  : ndarray (3, 2)
    timestep    : int
    """
    last_pos = -1
    with open(filename, "r") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if "ITEM: TIMESTEP" in line:
                last_pos = pos

    if last_pos < 0:
        raise RuntimeError(f"No ITEM: TIMESTEP found in {filename}")

    with open(filename, "r") as f:
        f.seek(last_pos)
        f.readline()                              # ITEM: TIMESTEP
        timestep = int(f.readline().strip())
        f.readline()                              # ITEM: NUMBER OF ENTRIES
        num_entries = int(f.readline().strip())
        f.readline()                              # ITEM: BOX BOUNDS ...
        box_bounds = np.zeros((3, 2))
        for b in range(3):
            vals = f.readline().strip().split()
            box_bounds[b, 0] = float(vals[0])
            box_bounds[b, 1] = float(vals[1])

        # ---- keep header as a plain Python str ----
        header_line = str(f.readline().strip())

        # Determine n_cols: try both ENTRIES and ATOMS prefixes
        hdr_body = header_line
        for prefix in ("ITEM: ENTRIES", "ITEM: ATOMS"):
            if prefix in hdr_body:
                hdr_body = hdr_body.replace(prefix, "")
                break
        hdr_tokens = hdr_body.strip().split()
        n_cols = max(len(hdr_tokens), 9)

        data = np.zeros((num_entries, n_cols))
        for i in range(num_entries):
            row_line = f.readline()
            if not row_line:
                break
            vals = row_line.strip().split()
            for j in range(min(len(vals), n_cols)):
                data[i, j] = float(vals[j])

    return data, header_line, num_entries, box_bounds, timestep

def load_specific_timestep_atoms(filename, target_timestep):
    """
    Load a specific timestep from a LAMMPS atoms dump file.

    Returns
    -------
    atoms_data  : ndarray (num_atoms, n_cols)
    header_line : str
    num_atoms   : int
    box_bounds  : ndarray (3, 2)
    timestep    : int
    """
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" not in line:
                continue

            timestep = int(f.readline().strip())

            f.readline()  # ITEM: NUMBER OF ATOMS
            num_atoms = int(f.readline().strip())

            f.readline()  # ITEM: BOX BOUNDS ...
            box_bounds = np.zeros((3, 2))
            for b in range(3):
                vals = f.readline().strip().split()
                box_bounds[b, 0] = float(vals[0])
                box_bounds[b, 1] = float(vals[1])

            header_line = str(f.readline().strip())  # ITEM: ATOMS ...

            hdr_tokens = header_line.replace("ITEM: ATOMS", "").strip().split()
            n_cols = len(hdr_tokens)

            if timestep == target_timestep:
                data = np.zeros((num_atoms, n_cols))
                for i in range(num_atoms):
                    vals = f.readline().strip().split()
                    for j in range(min(len(vals), n_cols)):
                        data[i, j] = float(vals[j])
                return data, header_line, num_atoms, box_bounds, timestep

            # skip this frame if not the one we want
            for _ in range(num_atoms):
                f.readline()

    raise RuntimeError(
        f"Timestep {target_timestep} not found in atoms dump: {filename}"
    )


def load_specific_timestep_bonds(filename, target_timestep):
    """
    Load a specific timestep from a LAMMPS bonds dump file.

    Returns
    -------
    bonds_data  : ndarray (num_entries, n_cols)
    header_line : str
    num_entries : int
    box_bounds  : ndarray (3, 2)
    timestep    : int
    """
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break

            if "ITEM: TIMESTEP" not in line:
                continue

            timestep = int(f.readline().strip())

            f.readline()  # ITEM: NUMBER OF ENTRIES
            num_entries = int(f.readline().strip())

            f.readline()  # ITEM: BOX BOUNDS ...
            box_bounds = np.zeros((3, 2))
            for b in range(3):
                vals = f.readline().strip().split()
                box_bounds[b, 0] = float(vals[0])
                box_bounds[b, 1] = float(vals[1])

            header_line = str(f.readline().strip())  # ITEM: ENTRIES ...

            hdr_body = header_line
            for prefix in ("ITEM: ENTRIES", "ITEM: ATOMS"):
                if prefix in hdr_body:
                    hdr_body = hdr_body.replace(prefix, "")
                    break
            hdr_tokens = hdr_body.strip().split()
            n_cols = max(len(hdr_tokens), 9)

            if timestep == target_timestep:
                data = np.zeros((num_entries, n_cols))
                for i in range(num_entries):
                    vals = f.readline().strip().split()
                    for j in range(min(len(vals), n_cols)):
                        data[i, j] = float(vals[j])
                return data, header_line, num_entries, box_bounds, timestep

            # skip this frame if not the one we want
            for _ in range(num_entries):
                f.readline()

    raise RuntimeError(
        f"Timestep {target_timestep} not found in bonds dump: {filename}"
    )


# WRITERS
def write_atoms_dump(filepath, atoms_data, header_line, num_atoms,
                     box_bounds, timestep, ent_degree):
    """
    Write a LAMMPS-style atoms dump for the last timestep,
    appending an ``ent_degree`` column.
    """
    new_header = str(header_line).rstrip()
    if "ent_degree" not in new_header:
        new_header += " ent_degree"

    # Figure out which columns are integer-valued from the header
    hdr_tokens = new_header.replace("ITEM: ATOMS", "").strip().split()
    int_names = {"id", "type", "mol", "ix", "iy", "iz"}

    with open(filepath, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{timestep}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{num_atoms}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for b in range(3):
            f.write(f"{box_bounds[b, 0]:.16e} {box_bounds[b, 1]:.16e}\n")
        f.write(new_header + "\n")

        n_orig_cols = atoms_data.shape[1]
        for i in range(num_atoms):
            parts = []
            for j in range(n_orig_cols):
                v = atoms_data[i, j]
                # Use the header token name to decide int vs float
                if j < len(hdr_tokens) - 1 and hdr_tokens[j] in int_names:
                    parts.append(str(int(v)))
                else:
                    parts.append(f"{v:g}")
            # Append ent_degree
            parts.append(f"{ent_degree[i]:g}")
            f.write(" ".join(parts) + "\n")


def write_bonds_dump(filepath, bonds_data, header_line, num_entries,
                     box_bounds, timestep):
    """
    Write a LAMMPS-style bonds dump for the last timestep (no modification).
    """
    hdr = str(header_line).rstrip()

    # Determine which columns are integer from header
    hdr_body = hdr
    for prefix in ("ITEM: ENTRIES", "ITEM: ATOMS"):
        if prefix in hdr_body:
            hdr_body = hdr_body.replace(prefix, "")
            break
    hdr_tokens = hdr_body.strip().split()
    # Typically first 4 cols are int for bonds: index, type, atom1, atom2
    int_cols = set(range(min(4, len(hdr_tokens))))

    with open(filepath, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{timestep}\n")
        f.write("ITEM: NUMBER OF ENTRIES\n")
        f.write(f"{num_entries}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for b in range(3):
            f.write(f"{box_bounds[b, 0]:.16e} {box_bounds[b, 1]:.16e}\n")
        f.write(hdr + "\n")

        n_cols = bonds_data.shape[1]
        for i in range(num_entries):
            parts = []
            for j in range(n_cols):
                v = bonds_data[i, j]
                if j in int_cols:
                    parts.append(str(int(v)))
                else:
                    parts.append(f"{v:g}")
            f.write(" ".join(parts) + "\n")


def write_vertices_dump(filepath, vertices, box_bounds, timestep):
    """
    Write vertices dump (MATLAB-compatible format).

    vertices : ndarray (N, 7) — [id, vtype, x, y, z, mol, LN]

    Output format matches MATLAB:
        ITEM: ATOMS i vtype x y z mol LN
    """
    n_verts = len(vertices)

    with open(filepath, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{timestep}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{n_verts}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for b in range(3):
            f.write(f"{box_bounds[b, 0]:.16e} {box_bounds[b, 1]:.16e}\n")
        f.write("ITEM: ATOMS i vtype x y z mol LN\n")

        for r in range(n_verts):
            v = vertices[r]
            f.write(f"{int(v[0])} {int(v[1])} {v[2]:g} {v[3]:g} {v[4]:g} "
                    f"{int(v[5])} {v[6]:g}\n")


def write_edges_dump(filepath, edges, box_bounds, timestep):
    """
    Write edges dump (MATLAB-compatible format).

    edges : ndarray (N, 9) — [edge_i, etype, i, j, rx, ry, rz, mol, N_bonds]

    Output format matches MATLAB:
        ITEM: ENTRIES edge_i etype i j rx ry rz mol N
    """
    n_edges = len(edges)

    with open(filepath, "w") as f:
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{timestep}\n")
        f.write("ITEM: NUMBER OF ENTRIES\n")
        f.write(f"{n_edges}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        for b in range(3):
            f.write(f"{box_bounds[b, 0]:.16e} {box_bounds[b, 1]:.16e}\n")
        f.write("ITEM: ENTRIES edge_i etype i j rx ry rz mol N\n")

        for r in range(n_edges):
            e = edges[r]
            f.write(f"{int(e[0])} {int(e[1])} {int(e[2])} {int(e[3])} "
                    f"{e[4]:g} {e[5]:g} {e[6]:g} {int(e[7])} {int(e[8])}\n")