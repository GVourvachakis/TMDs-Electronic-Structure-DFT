import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # register the 3D projection (optional)

from ase.dft.kpoints import labels_from_kpts, BandPath
from ase.atoms import Atoms
from ase.parallel import parprint

"""
Tang, S., et al. (2017). Quantum spin Hall state in monolayer 1T'-WTe₂. Nature Physics, 13(7), 683-687.

The implementation is based on crystallographic data from the above source, 
particularly the distortion pattern observed in STM and TEM studies of 1T' phase TMDs. 
The distortion factor of 0.07 is a reasonable starting value based 
on observed metal-metal dimerization in materials like WTe₂, 
but can be adjusted as needed for different TMD compositions.
"""

def mx2_custom(formula: str = "MoS2", kind: str = "1H", a: float = 3.16, thickness: float = 3.19,
                size: tuple[int,int,int] = (1, 1, 1), vacuum: float | None = None,
                distortion_factor: float = 0.07):

    """Create three-layer 2D materials with hexagonal structure.

    This routine is used for metal dichalcogenides :mol:`MX_2` 2D structures
    such as :mol:`MoS_2`.

    https://en.wikipedia.org/wiki/Transition_metal_dichalcogenide_monolayers

    Parameters
    ----------
    formula (str): Chemical formula of the TMD (default: "MoS2")
    kind : {'1H', '1T', '1T\''}, default: '1H'
        - "1H": mirror-plane symmetry (trigonal prismatic) [semiconductor]
        - "1T": inversion symmetry (octahedral or trigonal antiprismatic) [metal]
        - "1T'": Distorted 1T - Metastable topological phase [semimetal or narrow-gap semiconductor]
    a (float): Lattice constant in Angstrom (default: 3.16)
    thickness (float): Thickness of the layer in Angstrom (default: 3.19)
    size (tuple[int,int,int]): Repetition of the unit cell (default: (1, 1, 1))
    vacuum (float | None): If not None, add vacuum padding along z-axis (default: None)
    distortion_factor (float): Controls the magnitude of distortion for 
                                1T' structure (dimerization) (default: 0.07)
    """
    if kind == '1H':
        basis = [(0, 0, 0),
                (2/3, 1/3, 0.5 * thickness),
                (2/3, 1/3, -0.5 * thickness)]
        cell = [[a, 0, 0], [-a/2, a * 3**0.5/2, 0], [0, 0, 0]]
    elif kind == '1T':
        basis = [(0, 0, 0),
                (2/3, 1/3, 0.5 * thickness),
                (1/3, 2/3, -0.5 * thickness)]
        cell = [[a, 0, 0], [-a/2, a * 3**0.5/2, 0], [0, 0, 0]]
    elif kind == '1T\'':
        # 1T' phase has a rectangular unit cell (2x1 supercell of 1T)
        # With distortion along one direction (metal-metal dimerization)
        rect_a = a * 2  # Doubled in x direction for the rectangular cell
        rect_b = a * 3**0.5  # Width in y direction
        
        # Metal atom positions with dimerization
        m1_pos = (0.0, 0.0, 0.0)
        m2_pos = (0.5 - distortion_factor, 0.0, 0.0)
        
        # Chalcogen positions (top and bottom layers)
        x1_top = (0.25, 0.5, 0.5 * thickness)
        x2_top = (0.75, 0.5, 0.5 * thickness)
        x1_bot = (0.25, 0.0, -0.5 * thickness)
        x2_bot = (0.75, 0.0, -0.5 * thickness)
        
        basis = [m1_pos, m2_pos, x1_top, x2_top, x1_bot, x2_bot]
        cell = [[rect_a, 0, 0], [0, rect_b, 0], [0, 0, 0]]
        
        # Need to adjust formula for the 2x1 supercell
        formula += formula
    else:
        raise ValueError('Structure not recognized:', kind)

    atoms = Atoms(formula, cell=cell, pbc=(1, 1, 0))
    atoms.set_scaled_positions(basis)

    if vacuum is not None: # inter-slab gap = c − t = 2 × vacuum, where c = t + 2 × vacuum
        atoms.center(vacuum, axis=2)
    
    atoms = atoms.repeat(size)
    return atoms

def plot_band_structure(bs, material, phase, ax=None):
    """
    Plot band structure with proper labels and formatting, and print band gap information.
    
    Args:
        bs: ASE BandStructure object
        material: Material name (string)
        phase: Phase designation (string)
        ax: Matplotlib axis (if None, a new figure and axis will be created)
        
    Returns:
        fig, ax: The figure and axis objects
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Calculate energy bounds
    emin = bs.energies.min() - 0.5
    emax = bs.energies.max() + 0.5
    
    # Get Fermi level
    efermi = getattr(bs, 'efermi', 0.0)

    # Plot main band structure
    bs.plot(ax=ax, emin=emin, emax=emax, filename=f'{material}_{phase}_bandstructure.png')
    
    # Add spin polarized bands if applicable
    if hasattr(bs, 'nspin') and bs.nspin == 2:
        bs.plot(ax=ax, emin=emin, emax=emax, spin=1, color='red')
    
    # Get the k-points and cell for label generation
    kpts = bs.path.kpts
    cell = bs.path.cell
    
    # Get special point labels using a compatible method
    x, xticks, xticklabels = labels_from_kpts(kpts, cell)
    
    # Format plot
    ax.set_xlabel('Wave vector')
    ax.set_ylabel('Energy (eV)')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.axhline(y=efermi, color='k', linestyle='dotted', alpha=0.5)
    plt.tight_layout()
    
    # Calculate VBM, CBM, and band gap
    energies = bs.energies
    nspin, nkpts, nbands = energies.shape

    # Find Valence Band Maximum (VBM)
    vbm = -np.inf
    vbm_kpts = []
    for s in range(nspin):
        for k in range(nkpts):
            for b in range(nbands):
                e = energies[s, k, b]
                if e <= efermi and e > vbm:
                    vbm = e
                    vbm_kpts = [(s, k, b)]
                elif e <= efermi and e == vbm:
                    vbm_kpts.append((s, k, b))

    # Find Conduction Band Minimum (CBM)
    cbm = np.inf
    cbm_kpts = []
    for s in range(nspin):
        for k in range(nkpts):
            for b in range(nbands):
                e = energies[s, k, b]
                if e >= efermi and e < cbm:
                    cbm = e
                    cbm_kpts = [(s, k, b)]
                elif e >= efermi and e == cbm:
                    cbm_kpts.append((s, k, b))

    # Print VBM/CBM locations regardless of gap type
    vbm_found = vbm != -np.inf
    cbm_found = cbm != np.inf

    if vbm_found:
        vbm_k_indices = list(set([k for (s, k, b) in vbm_kpts]))
        vbm_coords = [bs.path.kpts[k] for k in vbm_k_indices]
        vbm_coords_rounded = [tuple(np.round(coord, 3)) for coord in vbm_coords]
        parprint(f"VBM energy: {vbm:.3f} eV")
        parprint(f"VBM k-point(s) (reciprocal coords): {vbm_coords_rounded}")
    else:
        parprint("Warning: VBM not found (all energies above Fermi level)")
    
    if cbm_found:
        cbm_k_indices = list(set([k for (s, k, b) in cbm_kpts]))
        cbm_coords = [bs.path.kpts[k] for k in cbm_k_indices]
        cbm_coords_rounded = [tuple(np.round(coord, 3)) for coord in cbm_coords]
        parprint(f"CBM energy: {cbm:.3f} eV")
        parprint(f"CBM k-point(s) (reciprocal coords): {cbm_coords_rounded}")
    else:
        parprint("Warning: CBM not found (all energies below Fermi level)")

    # Analyze band gap if both found
    if vbm_found and cbm_found:
        band_gap = cbm - vbm
        if band_gap <= 0:
            parprint("Material is metallic (no band gap)")
        else:
            common_ks = set(vbm_k_indices).intersection(cbm_k_indices)
            gap_type = 'direct' if common_ks else 'indirect'
            parprint(f"Estimated band gap: {band_gap:.3f} eV")
            parprint(f"Band gap type: {gap_type}")
    
    return fig, ax

def labels_from_kpts(kpts, cell):
    # Placeholder for label generation logic
    x = np.arange(len(kpts))
    xticks = []
    xticklabels = []
    return x, xticks, xticklabels

def plot_band_path(atoms, phase, save_path=None):
    """
    Plot the Brillouin zone and high-symmetry path for the given crystal structure.
    
    Args:
        atoms (ase.Atoms): The atomic structure
        phase (str): Phase (1H, 1T, or 1T')
        save_path (str, optional): Path to save the figure. If None, the figure is not saved.

    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # --- Monkey-patch ax.text to bypass the extra argument issue ---
    orig_text = ax.text
    
    def fixed_text(*args, **kwargs):
        # The expected call signature is: text(x, y, s, fontdict=None, **kwargs)
        # If more than three positional arguments are given and the fourth is a string,
        # we drop that extra argument.
        if len(args) > 3 and isinstance(args[3], str):
            args = args[:3] + args[4:]
        return orig_text(*args, **kwargs)

    ax.text = fixed_text
    # --- End monkey-patch ---

    # Choose high symmetry path based on phase
    if phase in ['1H', '1T']:
        # 2D axis for hexagonal lattice
        fig, ax = plt.subplots(figsize=(6, 5))
        # Hexagonal lattice path
        bp = atoms.cell.bandpath('GMKG', npoints=100)
        # Plot the band path
        bp.plot(ax=ax)
        ax.set_title(f'Band Path for {phase} Phase (GMKG)')
    
    elif phase == "1T'":
        # For orthorhombic cell of 1T' structure
        # Define special points for rectangular Brillouin zone
        # ASE’s bz_plot calls ax.view_init, which exists only on 3D axes.
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        special_points = {
            'G': [0, 0, 0],
            'X': [0.5, 0, 0],
            'Y': [0, 0.5, 0],
            'S': [0.5, 0.5, 0]
        }
        path = 'GXSYG'  # Standard path for rectangular cell
        bp = BandPath(cell=atoms.cell, path=path, special_points=special_points)
        # Plot the band path
        bp.plot(ax=ax)
        ax.set_title(f'Band Path for {phase} Phase (GXSYG)')
    else:
        raise ValueError(f"Unknown phase: {phase}")
    
    # Add axis labels and grid
    # For 3D axes, you may also consider setting the z-label if needed.
    ax.set_xlabel('k-path')
    ax.set_ylabel('k-points')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    
    return fig, ax