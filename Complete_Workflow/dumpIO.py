""" LAMMPS dump readers and processed dump writers."""
import numpy as np

def iter_frames(filename):
    with open(filename) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if not line.strip():
                continue
            if line.strip() != "ITEM: TIMESTEP":
                raise ValueError(f"Expected timestep header in {filename}: {line!r}")
            timestep = int(f.readline())
            count_header = f.readline().strip()
            if count_header not in {"ITEM: NUMBER OF ATOMS", "ITEM: NUMBER OF ENTRIES"}:
                raise ValueError(f"Invalid count header: {count_header}")
            count = int(f.readline())
            box_header = f.readline().split()
            if box_header[:3] != ["ITEM:", "BOX", "BOUNDS"] or any(x in box_header for x in ("xy", "xz", "yz", "abc")):
                raise ValueError("Only orthorhombic BOX BOUNDS dumps are supported")
            bounds = np.array([[float(x) for x in f.readline().split()] for _ in range(3)])
            if bounds.shape != (3, 2) or np.any(bounds[:, 1] <= bounds[:, 0]):
                raise ValueError("Invalid box bounds")
            header = f.readline().strip()
            if not header.startswith(("ITEM: ATOMS ", "ITEM: ENTRIES ")):
                raise ValueError(f"Invalid data header: {header}")
            ncols = len(header.split()) - 2
            data = np.empty((count, ncols))
            for i in range(count):
                values = f.readline().split()
                if len(values) != ncols:
                    raise ValueError(f"Truncated/malformed dump {filename}, timestep {timestep}, row {i+1}")
                data[i] = [float(x) for x in values]
            if not np.isfinite(data).all():
                raise ValueError(f"Nonfinite data in {filename}, timestep {timestep}")
            yield data, header, count, bounds, timestep


def _load(filename, timestep=None):
    last = None
    for frame in iter_frames(filename):
        if timestep is not None and frame[-1] == timestep:
            return frame
        last = frame
    if timestep is None and last is not None:
        return last
    raise ValueError(f"Timestep {timestep} not found in {filename}")


def load_last_timestep_atoms(filename):
    return _load(filename)


def load_last_timestep_bonds(filename):
    return _load(filename)


def load_specific_timestep_atoms(filename, target_timestep):
    return _load(filename, target_timestep)


def load_specific_timestep_bonds(filename, target_timestep):
    return _load(filename, target_timestep)


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