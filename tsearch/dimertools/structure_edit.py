import warnings
import numpy as np
import random
from ase import Atom
from ase.constraints import FixAtoms
from ase.neighborlist import NeighborList, natural_cutoffs, neighbor_list
from ase.build import make_supercell
from ase.data import covalent_radii, atomic_numbers


def turn_into_supercell(atoms):
    n_atoms = len(atoms)
    M = [1, 1, 1]
    if n_atoms == 1: M = [3, 3, 3]
    elif n_atoms <= 4: M = [2, 2, 2]
    elif 5 <= n_atoms <= 8:
        lengths = atoms.cell.lengths()
        M[np.argsort(lengths)[0]] = 2
        M[np.argsort(lengths)[1]] = 2
    elif 9 <= n_atoms <= 16:
        M[np.argmin(atoms.cell.lengths())] = 2
    if M != [1, 1, 1]:
        atoms = make_supercell(atoms, np.diag(M))
    return atoms


def find_interstitial_sites(atoms, min_dist_frac=0.4):
    """Find interstitial sites using Voronoi tessellation on periodic images.

    Parameters
    ----------
    atoms : ase.Atoms
        Structure to find interstitial sites in.
    min_dist_frac : float
        Minimum distance from any atom as a fraction of the nearest-neighbor
        distance. Sites closer than this are discarded.

    Returns
    -------
    sites : np.ndarray, shape (N_sites, 3)
        Cartesian coordinates of interstitial sites.
    """
    from scipy.spatial import Voronoi
    from scipy.cluster.hierarchy import fcluster, linkage

    cell = atoms.get_cell()
    positions = atoms.get_positions()

    # Build 3x3x3 periodic images
    shifts = np.array(
        [[i, j, k] for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)]
    )
    all_positions = []
    for s in shifts:
        all_positions.append(positions + s @ cell)
    all_positions = np.vstack(all_positions)

    if len(all_positions) < 4:
        return np.empty((0, 3))

    vor = Voronoi(all_positions)
    vertices = vor.vertices

    inv_cell = np.linalg.inv(cell)

    # Keep vertices inside the unit cell (fractional coords in [0, 1))
    frac_coords = vertices @ inv_cell
    inside = np.all((frac_coords >= -1e-6) & (frac_coords < 1.0 - 1e-6), axis=1)
    candidates = vertices[inside]

    if len(candidates) == 0:
        return np.empty((0, 3))

    # Compute nearest-neighbor distance in the original structure
    if len(atoms) >= 2:
        _, _, d = neighbor_list('ijd', atoms, 5.0)
        if len(d) > 0:
            nn_dist = d.min()
        else:
            nn_dist = 2.5  # fallback
    else:
        nn_dist = 2.5

    min_dist = min_dist_frac * nn_dist

    # Filter: keep sites far enough from any real atom (using MIC)
    keep = []
    for idx, site in enumerate(candidates):
        deltas = positions - site
        frac_deltas = deltas @ inv_cell
        frac_deltas -= np.round(frac_deltas)
        cart_deltas = frac_deltas @ cell
        dists_to_atoms = np.linalg.norm(cart_deltas, axis=1)
        if dists_to_atoms.min() > min_dist:
            keep.append(idx)

    if len(keep) == 0:
        return np.empty((0, 3))

    candidates = candidates[keep]

    # Cluster nearby sites within 0.5 Å
    if len(candidates) > 1:
        Z = linkage(candidates, method='single')
        labels = fcluster(Z, t=0.5, criterion='distance')
        unique_labels = np.unique(labels)
        clustered = []
        for lbl in unique_labels:
            mask = labels == lbl
            clustered.append(candidates[mask].mean(axis=0))
        candidates = np.array(clustered)

    return candidates


def _mic_vector(vec, cell, inv_cell):
    """Apply minimum image convention to a vector or array of vectors."""
    frac = vec @ inv_cell
    frac -= np.round(frac)
    return frac @ cell


def _safe_normalize(vec):
    """Normalize a vector, returning a random unit vector if norm is near zero."""
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        vec = np.random.randn(3)
        norm = np.linalg.norm(vec)
    return vec / norm


def _nearest_site(site_a, other_sites, cell, inv_cell):
    """Return the nearest site from other_sites to site_a under MIC."""
    deltas = _mic_vector(other_sites - site_a, cell, inv_cell)
    dists = np.linalg.norm(deltas, axis=1)
    idx = np.argmin(dists)
    return other_sites[idx], deltas[idx]


def _get_atom_selection_weights(atoms):
    """Return probability weights for atom selection based on inverse covalent radii."""
    numbers = atoms.get_atomic_numbers()
    weights = np.array([1.0 / covalent_radii[z] for z in numbers])
    weights /= weights.sum()
    return weights


# --- Element sampling pools ---

_HOP_INSERT_ELEMENTS = [1, 6, 7, 8, 5]  # H, C, N, O, B (small interstitial species)


def _sample_hop_insert_element():
    """Sample a common small interstitial element weighted by inverse covalent radius."""
    weights = np.array([1.0 / covalent_radii[z] for z in _HOP_INSERT_ELEMENTS])
    weights /= weights.sum()
    idx = np.random.choice(len(_HOP_INSERT_ELEMENTS), p=weights)
    return _HOP_INSERT_ELEMENTS[idx]


# Common metallic / bulk elements for kickout_insert (similar-sized substitutional candidates)
_KICKOUT_INSERT_POOL = [
    'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
    'Al', 'Si', 'Ge', 'Sn', 'Ga', 'In',
]
_KICKOUT_INSERT_POOL_Z = [atomic_numbers[s] for s in _KICKOUT_INSERT_POOL]
_KICKOUT_INSERT_POOL_RADII = np.array([covalent_radii[z] for z in _KICKOUT_INSERT_POOL_Z])


def _sample_kickout_insert_element(atoms, sigma=0.2):
    """Sample an element with covalent radius similar to the host atoms.

    Uses a Gaussian weight: exp(-(r_candidate - r_host_avg)^2 / (2*sigma^2)).
    """
    host_radii = np.array([covalent_radii[z] for z in atoms.get_atomic_numbers()])
    r_avg = host_radii.mean()

    weights = np.exp(-(_KICKOUT_INSERT_POOL_RADII - r_avg) ** 2 / (2 * sigma ** 2))
    weights /= weights.sum()

    idx = np.random.choice(len(_KICKOUT_INSERT_POOL_Z), p=weights)
    return _KICKOUT_INSERT_POOL_Z[idx]


# --- Vacancy attempts ---

def get_vacancy_attempts(atoms, config_dict, num_attempts):
    """Original bulk vacancy mechanism extracted from get_attempts."""
    atoms = turn_into_supercell(atoms)
    i_idx, j_idx = neighbor_list('ij', atoms, 3.5)

    num_attempts = min(num_attempts, len(atoms))
    remove_indices = random.sample(range(len(atoms)), num_attempts)

    images = []
    displacement_dicts = []
    selected_indices = []

    for rm_idx in remove_indices:
        neighbor_indices = j_idx[i_idx == rm_idx]
        if len(neighbor_indices) == 0:
            neighbor_indices = [x for x in range(len(atoms)) if x != rm_idx]

        chosen_neighbor = random.choice(neighbor_indices)

        atoms_new = atoms.copy()
        del atoms_new[rm_idx]
        atoms_new.info['reaction_type'] = 'vacancy'
        images.append(atoms_new)

        new_center_idx = chosen_neighbor if chosen_neighbor < rm_idx else chosen_neighbor - 1
        displacement_dicts.append({"displacement_center": int(new_center_idx)})
        selected_indices.append(rm_idx)

    return images, displacement_dicts, selected_indices


# --- Hop attempts (interstitial mechanism) ---

def get_hop_reuse_attempts(atoms, num_attempts):
    """Relocate an existing atom to an interstitial site, displace toward a neighbor site."""
    atoms = turn_into_supercell(atoms)
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping hop_reuse.")
        return [], [], []

    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    weights = _get_atom_selection_weights(atoms)

    images = []
    displacement_dicts = []
    selected_indices = []

    for _ in range(num_attempts):
        atom_idx = np.random.choice(len(atoms), p=weights)

        # Pick site A (random)
        site_a_idx = random.randrange(len(sites))
        site_a = sites[site_a_idx]

        # Pick site B (nearest other site to A)
        other_sites = np.delete(sites, site_a_idx, axis=0)
        _, delta_ab = _nearest_site(site_a, other_sites, cell, inv_cell)
        direction = _safe_normalize(delta_ab)

        # Build structure: move atom to site A
        atoms_new = atoms.copy()
        atoms_new.positions[atom_idx] = site_a
        atoms_new.info['reaction_type'] = 'hop_reuse'

        # Displacement vector: ~0.1 Å along direction, only for moved atom
        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[atom_idx] = 0.1 * direction

        images.append(atoms_new)
        displacement_dicts.append({
            "displacement_vector": disp_vector,
            "method": "vector",
        })
        selected_indices.append(int(atom_idx))

    return images, displacement_dicts, selected_indices


def get_hop_insert_attempts(atoms, num_attempts):
    """Insert a new small atom at an interstitial site, displace toward a neighbor site."""
    atoms = turn_into_supercell(atoms)
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping hop_insert.")
        return [], [], []

    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)

    images = []
    displacement_dicts = []
    selected_indices = []

    for _ in range(num_attempts):
        element_z = _sample_hop_insert_element()

        # Pick site A (random)
        site_a_idx = random.randrange(len(sites))
        site_a = sites[site_a_idx]

        # Pick site B (nearest other site to A)
        other_sites = np.delete(sites, site_a_idx, axis=0)
        _, delta_ab = _nearest_site(site_a, other_sites, cell, inv_cell)
        direction = _safe_normalize(delta_ab)

        # Build structure: add new atom at site A
        atoms_new = atoms.copy()
        atoms_new.append(Atom(element_z, position=site_a))
        new_atom_idx = len(atoms_new) - 1
        atoms_new.info['reaction_type'] = 'hop_insert'

        # Displacement vector: ~0.1 Å along direction, only for inserted atom
        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[new_atom_idx] = 0.1 * direction

        images.append(atoms_new)
        displacement_dicts.append({
            "displacement_vector": disp_vector,
            "method": "vector",
        })
        selected_indices.append(int(new_atom_idx))

    return images, displacement_dicts, selected_indices


# --- Kickout attempts (interstitialcy / kick-out mechanism) ---

def _find_nearest_lattice_atom(site, atoms, cell, inv_cell):
    """Return the index of the lattice atom nearest to an interstitial site (MIC)."""
    deltas = _mic_vector(atoms.get_positions() - site, cell, inv_cell)
    dists = np.linalg.norm(deltas, axis=1)
    return int(np.argmin(dists))


def get_kickout_reuse_attempts(atoms, num_attempts):
    """Existing atom placed at interstitial site kicks nearest lattice atom into another site.

    1. Pick an existing atom (weighted by inverse covalent radius), move it to site A.
    2. Find the nearest lattice atom to site A — this is the atom being kicked.
    3. The kicked atom is displaced toward site B (nearest interstitial site to it).
    4. The kicking atom is displaced toward the kicked atom's original position.
    """
    atoms = turn_into_supercell(atoms)
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping kickout_reuse.")
        return [], [], []

    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)
    weights = _get_atom_selection_weights(atoms)

    images = []
    displacement_dicts = []
    selected_indices = []

    for _ in range(num_attempts):
        # Pick the atom to become the interstitial (kicker)
        kicker_idx = np.random.choice(len(atoms), p=weights)

        # Pick site A for the kicker
        site_a_idx = random.randrange(len(sites))
        site_a = sites[site_a_idx]

        # Build structure: move kicker to site A
        atoms_new = atoms.copy()
        atoms_new.positions[kicker_idx] = site_a

        # Find the nearest lattice atom to site A (the one being kicked)
        # Exclude the kicker itself
        positions = atoms_new.get_positions()
        deltas = _mic_vector(positions - site_a, cell, inv_cell)
        dists = np.linalg.norm(deltas, axis=1)
        dists[kicker_idx] = np.inf  # exclude kicker
        kicked_idx = int(np.argmin(dists))
        kicked_original_pos = positions[kicked_idx].copy()

        # Find site B: nearest interstitial site to the kicked atom (exclude site A)
        other_sites = np.delete(sites, site_a_idx, axis=0)
        site_b, _ = _nearest_site(kicked_original_pos, other_sites, cell, inv_cell)

        # Direction for kicked atom: from its position toward site B
        delta_kicked = _mic_vector(site_b - kicked_original_pos, cell, inv_cell)
        dir_kicked = _safe_normalize(delta_kicked)

        # Direction for kicker: from site A toward kicked atom's original position
        delta_kicker = _mic_vector(kicked_original_pos - site_a, cell, inv_cell)
        dir_kicker = _safe_normalize(delta_kicker)

        atoms_new.info['reaction_type'] = 'kickout_reuse'

        # Two-atom displacement
        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[kicker_idx] = 0.1 * dir_kicker
        disp_vector[kicked_idx] = 0.1 * dir_kicked

        images.append(atoms_new)
        displacement_dicts.append({
            "displacement_vector": disp_vector,
            "method": "vector",
        })
        selected_indices.append(int(kicker_idx))

    return images, displacement_dicts, selected_indices


def get_kickout_insert_attempts(atoms, num_attempts):
    """Insert a new similar-sized atom at interstitial site; it kicks nearest lattice atom out.

    1. Sample a new element with covalent radius similar to host (Gaussian weight).
    2. Insert it at interstitial site A.
    3. Find the nearest lattice atom — this is the atom being kicked.
    4. The kicked atom is displaced toward site B.
    5. The inserted atom is displaced toward the kicked atom's original position.
    """
    atoms = turn_into_supercell(atoms)
    sites = find_interstitial_sites(atoms)

    if len(sites) < 2:
        warnings.warn("Found fewer than 2 interstitial sites; skipping kickout_insert.")
        return [], [], []

    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)

    images = []
    displacement_dicts = []
    selected_indices = []

    for _ in range(num_attempts):
        element_z = _sample_kickout_insert_element(atoms)

        # Pick site A
        site_a_idx = random.randrange(len(sites))
        site_a = sites[site_a_idx]

        # Find nearest lattice atom to site A (the one being kicked)
        kicked_idx = _find_nearest_lattice_atom(site_a, atoms, cell, inv_cell)
        kicked_original_pos = atoms.get_positions()[kicked_idx].copy()

        # Find site B: nearest interstitial site to kicked atom (exclude site A)
        other_sites = np.delete(sites, site_a_idx, axis=0)
        site_b, _ = _nearest_site(kicked_original_pos, other_sites, cell, inv_cell)

        # Direction for kicked atom: toward site B
        delta_kicked = _mic_vector(site_b - kicked_original_pos, cell, inv_cell)
        dir_kicked = _safe_normalize(delta_kicked)

        # Direction for inserted atom: toward kicked atom's original position
        delta_insert = _mic_vector(kicked_original_pos - site_a, cell, inv_cell)
        dir_insert = _safe_normalize(delta_insert)

        # Build structure: insert new atom at site A
        atoms_new = atoms.copy()
        atoms_new.append(Atom(element_z, position=site_a))
        inserted_idx = len(atoms_new) - 1
        atoms_new.info['reaction_type'] = 'kickout_insert'

        # Two-atom displacement
        disp_vector = np.zeros((len(atoms_new), 3))
        disp_vector[inserted_idx] = 0.1 * dir_insert
        disp_vector[kicked_idx] = 0.1 * dir_kicked

        images.append(atoms_new)
        displacement_dicts.append({
            "displacement_vector": disp_vector,
            "method": "vector",
        })
        selected_indices.append(int(inserted_idx))

    return images, displacement_dicts, selected_indices


# --- Exchange / Ring attempts (coordinated position swaps) ---

def _find_ring(neighbors_dict, seed, ring_size, max_retries=50):
    """Find a ring of connected atoms starting from seed.

    For ring_size=2, simply pick a random neighbor.
    For ring_size>=3, random walk through neighbors, where the last atom
    must be a neighbor of seed to close the ring.

    Returns list of ring_size unique atom indices forming the ring, or None.
    """
    if ring_size == 2:
        nbrs = neighbors_dict.get(seed, [])
        if len(nbrs) == 0:
            return None
        return [seed, random.choice(nbrs)]

    seed_neighbors = set(neighbors_dict.get(seed, []))

    for _ in range(max_retries):
        path = [seed]
        for step in range(ring_size - 1):
            current = path[-1]
            nbrs = neighbors_dict.get(current, [])
            if step == ring_size - 2:
                # Last step: pick an unvisited neighbor that is also a neighbor of seed
                candidates = [n for n in nbrs if n not in path and n in seed_neighbors]
            else:
                # Pick an unvisited neighbor (excluding seed to avoid short-circuit)
                candidates = [n for n in nbrs if n not in path]
            if not candidates:
                break
            path.append(random.choice(candidates))
        else:
            # path has ring_size entries, all unique, and last is a neighbor of seed
            return path

    return None


def _build_neighbor_dict(atoms, cutoff=3.5):
    """Build a dict mapping atom index -> list of neighbor indices."""
    i_idx, j_idx = neighbor_list('ij', atoms, cutoff)
    neighbors_dict = {}
    for i, j in zip(i_idx, j_idx):
        neighbors_dict.setdefault(int(i), []).append(int(j))
    return neighbors_dict


def _make_ring_attempt(atoms, neighbors_dict, cell, inv_cell, ring_size, reaction_type):
    """Create a single ring swap attempt. Returns (image, disp_dict, index) or None."""
    seed = random.randrange(len(atoms))
    ring = _find_ring(neighbors_dict, seed, ring_size)

    if ring is None:
        warnings.warn(f"Could not find ring of size {ring_size}; skipping one {reaction_type} attempt.")
        return None

    atoms_new = atoms.copy()
    positions = atoms_new.get_positions()
    disp_vector = np.zeros_like(positions)

    for k in range(len(ring)):
        src = ring[k]
        dst = ring[(k + 1) % len(ring)]
        delta = _mic_vector(positions[dst] - positions[src], cell, inv_cell)
        disp_vector[src] = 0.1 * _safe_normalize(delta)

    atoms_new.info['reaction_type'] = reaction_type

    return (atoms_new,
            {"displacement_vector": disp_vector, "method": "vector"},
            int(ring[0]))


def get_exchange_attempts(atoms, num_attempts):
    """Two neighboring atoms swap positions (ring_size=2)."""
    atoms = turn_into_supercell(atoms)
    neighbors_dict = _build_neighbor_dict(atoms)
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        result = _make_ring_attempt(atoms, neighbors_dict, cell, inv_cell, 2, 'exchange')
        if result:
            images.append(result[0])
            displacement_dicts.append(result[1])
            selected_indices.append(result[2])
    return images, displacement_dicts, selected_indices


def get_ring_attempts(atoms, config_dict, num_attempts):
    """Ring of 3+ atoms rotate cooperatively. Ring size randomly sampled from config."""
    ring_sizes = config_dict["ourDimer"].get("ring_sizes", [3, 4])
    if isinstance(ring_sizes, (int, float)):
        ring_sizes = [int(ring_sizes)]
    elif isinstance(ring_sizes, str):
        ring_sizes = [int(x) for x in ring_sizes.split()]
    else:
        ring_sizes = [int(x) for x in ring_sizes]

    atoms = turn_into_supercell(atoms)
    neighbors_dict = _build_neighbor_dict(atoms)
    cell = atoms.get_cell()
    inv_cell = np.linalg.inv(cell)

    images, displacement_dicts, selected_indices = [], [], []
    for _ in range(num_attempts):
        size = random.choice(ring_sizes)
        result = _make_ring_attempt(atoms, neighbors_dict, cell, inv_cell, size, 'ring')
        if result:
            images.append(result[0])
            displacement_dicts.append(result[1])
            selected_indices.append(result[2])
    return images, displacement_dicts, selected_indices


# --- Main dispatch ---

_REACTION_TYPE_DISPATCH = {
    "vacancy": lambda atoms, config_dict, n: get_vacancy_attempts(atoms, config_dict, n),
    "hop_reuse": lambda atoms, config_dict, n: get_hop_reuse_attempts(atoms, n),
    "hop_insert": lambda atoms, config_dict, n: get_hop_insert_attempts(atoms, n),
    "kickout_reuse": lambda atoms, config_dict, n: get_kickout_reuse_attempts(atoms, n),
    "kickout_insert": lambda atoms, config_dict, n: get_kickout_insert_attempts(atoms, n),
    "exchange": lambda atoms, config_dict, n: get_exchange_attempts(atoms, n),
    "ring": lambda atoms, config_dict, n: get_ring_attempts(atoms, config_dict, n),
}


def get_attempts(atoms, config_dict):

    atoms = atoms.copy()

    images = []
    displacement_dicts = []
    selected_indices = []

    if config_dict["ourDimer"]["dataset_type"] == "bulk":

        # Determine reaction types and per-type attempt count
        reaction_types = config_dict["ourDimer"].get("reaction_types")
        if reaction_types is None:
            raise ValueError("Configuration error: 'ourDimer' -> 'reaction_types' is not set. "
                             "Please specify reaction types (e.g., 'vacancy') in config.ini")
        if isinstance(reaction_types, str):
            reaction_types = [reaction_types]

        num_per_type = config_dict["ourDimer"].get("num_attempts_per_type", 1)

        for rtype in reaction_types:
            if rtype not in _REACTION_TYPE_DISPATCH:
                supported = ", ".join(_REACTION_TYPE_DISPATCH.keys())
                raise ValueError(f"Unknown reaction type: '{rtype}'. "
                                 f"Supported types: {supported}")
            imgs, dds, idxs = _REACTION_TYPE_DISPATCH[rtype](atoms, config_dict, num_per_type)
            images.extend(imgs)
            displacement_dicts.extend(dds)
            selected_indices.extend(idxs)

    elif config_dict["ourDimer"]["dataset_type"] == "oc":

        tags = atoms.get_tags()
        indices = np.where(tags == 0)[0]
        atoms.set_constraint(FixAtoms(indices=indices))

        cutoffs = natural_cutoffs(atoms, mult=1.25)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        adsorbate_indices = np.where(tags == 2)[0]
        if adsorbate_indices.shape[0] == 0:
            adsorbate_indices = np.where(tags == 1)[0]

        neighbor_indices = set()
        for idx in adsorbate_indices:
            indices, offsets = nl.get_neighbors(idx)
            neighbor_indices.update(indices)
        unique_neighbors = np.array(list(neighbor_indices))
        mask = np.zeros(len(atoms), dtype=bool)
        mask[unique_neighbors] = True
        mask = mask.tolist()

        num_needed = config_dict["ourDimer"]["num_attempts"]//3
        if len(adsorbate_indices) >= num_needed:
            chosen_indices = random.sample(list(adsorbate_indices), num_needed)
        else:
            chosen_indices = list(adsorbate_indices) * int(num_needed//len(adsorbate_indices))
            remainder = num_needed % len(adsorbate_indices)
            chosen_indices.extend(random.sample(list(adsorbate_indices), remainder))

        for i in range(config_dict["ourDimer"]["num_attempts"]):
            images.append(atoms.copy())

            if i < num_needed:
                displacement_dicts.append({"displacement_center": int(chosen_indices[i])})
                selected_indices.append(int(chosen_indices[i]))
            elif num_needed <= i < 2*num_needed:
                displacement_dicts.append({"displacement_center": int(chosen_indices[i-num_needed]), "gauss_std":0.2, "number_of_atoms":1})
                selected_indices.append(int(chosen_indices[i-num_needed]))
            else:
                displacement_dicts.append({"mask": mask})
                selected_indices.append(-1)

    else:
        raise Exception("dataset_type in ourDimer must be set")

    return images, displacement_dicts, selected_indices
