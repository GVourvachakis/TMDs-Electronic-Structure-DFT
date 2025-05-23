#!/usr/bin/env python3
"""
TMD Band Structure Calculator

This module performs the calculation and plotting of the electronic band structure
for monolayer transition metal dichalcogenides (TMDs). It supports semiconducting 
and metallic phases including 1H, 1T, and the topologically interesting 1T' phase.

The script constructs the monolayer structure using a customized version of ASE's 
`mx2` builder that distinguishes between the commonly confused "2H" and "1H" labels 
and enables the inclusion of the distorted 1T' phase. After structure creation, the 
workflow proceeds through ionic relaxation (unless skipped), a self-consistent 
ground-state calculation using GPAW, and finally a non-self-consistent calculation 
to obtain the band structure along a high-symmetry path in the Brillouin zone.

Supported Materials:
  - MoS2
  - MoSe2
  - WS2
  - WSe2
  - MoTe2
  - WTe2

Supported Phases:
  - 1H (trigonal prismatic coordination; mirror-plane symmetry)
  - 1T (octahedral/trigonal antiprismatic coordination; inversion symmetry)
  - 1T' (distorted 1T; lower symmetry with metal-metal chains)

Module Structure:
  - parse_args(): Parses command-line arguments for material, phase, and parallelization settings.
  - build_structure(): Constructs the monolayer unit cell using ASE.
  - relax_structure(): Performs geometry optimization around a literature-based lattice constant.
  - run_ground_state(): Executes the self-consistent GPAW calculation.
  - calculate_band_structure(): Runs the non-self-consistent band calculation.
  - plot_band_structure(): Extracts band path and visualizes the electronic structure.

Prerequisites:
  - ASE and GPAW must be installed with MPI support.
  - Parameters for each phase and material are defined in the module.
  - A reliable estimate of the lattice constant is assumed for initial structure generation.

Usage:
  $ mpirun -np 8 python tmd_bs.py --nprocs 4 --material MoS2 --phase "1H" --domain-parallel --kpt-parallel
    (add: --skip-relax to skip the relaxation step)
    parallelization_params = {'kpt': 4, 'domain': 2}

  $ mpirun -np 8 python tmd_bs.py --nprocs 2 --material MoS2 --phase "1H" --domain-parallel --kpt-parallel --band-parallel
    parallelization_params = {'kpt': 2, 'band': 2, 'domain': 2}

  $ mpiexec -np 8 python tmd_bs.py --material MoS2 --phase "1H" --domain-parallel --kpt-parallel

Arguments:
  --material STRING         TMD material to simulate (e.g., MoS2, WSe2).
  --phase STRING            Monolayer phase (1H, 1T, 1T').
  --nprocs INT              Number of total MPI processes to use.
  --domain-parallel         Enable real-space domain decomposition.
  --kpt-parallel            Enable k-point parallelization.
  --band-parallel           Enable band-level parallelization.
  --skip-relax              Skip the geometry relaxation step if structure is already optimized.

Output:
  - Relaxed atomic structure file (if relaxation is not skipped).
  - GPAW ground-state density file (.gpw).
  - Band structure plot along high-symmetry path (e.g., Γ-K-M-Γ).
  - (Optional) Band energies saved to file for post-processing.

Phase-specific analysis:
  - 1H phases: Expected to be direct-gap semiconductors (Γ or K valley).
  - 1T phases: Often metallic or semi-metallic with d-orbital dominated bands.
  - 1T' phases: May exhibit SOC-driven topological transitions or band inversion.
"""

import argparse
import os
import matplotlib.pyplot as plt
import json
import numpy as np
import ase
from ase.optimize import BFGS  # For ionic relaxation
from ase.constraints import FixAtoms
from ase.io import write
from ase.dft.kpoints import BandPath
from ase.spectrum.band_structure import BandStructure as ASEBandStructure
from gpaw import GPAW, PW, setup_paths, FermiDirac
from gpaw.mixer import Mixer
from gpaw.spinorbit import soc_eigenstates

from ase.parallel import parprint # Print messages only once (from rank 0)
# from gpaw.bandstructure import BandStructurePlotter # if GPAW<=24.6.0 

from mx2_utils import mx2_custom, plot_band_structure, plot_band_path

def setup_gpaw_paths():
    # Set up GPAW dataset path
    setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
    setup_paths[:] = [setup_path]  # Replace all existing paths
    os.environ['GPAW_SETUP_PATH'] = setup_path

setup_gpaw_paths()

# Handle the expected SOC-induced band splitting in 1T' phase
kpoints_dist_1T: tuple[int, int, int] = (15,15,1)
pw_dist_1T: int = 400

TMD_parameters = {
    'MoS2': {
        '1H': {'a': 3.16, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T': {'a': 3.19, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T\'': {'a': 3.19, 'vacuum': 20.0, 'pw_cutoff': pw_dist_1T, 'kpts': kpoints_dist_1T}
    },
    'MoSe2': {
        '1H': {'a': 3.29, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T': {'a': 3.32, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T\'': {'a': 3.32, 'vacuum': 20.0, 'pw_cutoff': pw_dist_1T, 'kpts': kpoints_dist_1T}
    },
    'WS2': {
        '1H': {'a': 3.15, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T': {'a': 3.18, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T\'': {'a': 3.18, 'vacuum': 20.0, 'pw_cutoff': pw_dist_1T, 'kpts': kpoints_dist_1T}
    },
    'WSe2': {
        '1H': {'a': 3.28, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T': {'a': 3.30, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T\'': {'a': 3.30, 'vacuum': 20.0, 'pw_cutoff': pw_dist_1T, 'kpts': kpoints_dist_1T}
    },
    'MoTe2': {
        '1H': {'a': 3.52, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T': {'a': 3.54, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T\'': {'a': 3.54, 'vacuum': 20.0, 'pw_cutoff': pw_dist_1T, 'kpts': kpoints_dist_1T}
    },
    'WTe2': {
        '1H': {'a': 3.55, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        '1T': {'a': 3.56, 'vacuum': 20.0, 'pw_cutoff': 350, 'kpts': (12, 12, 1)},
        # For 1T' (the distorted phase of WTe2), several studies have reported similar 
        # in-plane lattice constants.
        # For instance, Qian et al., Science 346, 1344 (2014) predict a ~3.56 Å 
        # in-plane lattice constant, and Fei et al., Nat. Phys. 13, 677 (2017) 
        # report a distorted structure with similar values.
        '1T\'': {'a': 3.56, 'vacuum': 20.0, 'pw_cutoff': pw_dist_1T, 'kpts': kpoints_dist_1T}
    }
}

def parse_arguments():
    """
    Parse command-line arguments.
    
    '1H': Mirror-plane symmetry (trigonal prismatic coordination, '2H' in GPAW's source code).
    '1T': Inversion symmetry (octahedral coordination → standard 1T).
    
    Returns:
        args: Namespace with material and phase specified by the user.
    """
    parser = argparse.ArgumentParser(description="Calculate TMD band structure")
    parser.add_argument("--material", type=str, required=True,
                        choices=TMD_parameters.keys(),
                        help="TMD material: MoS2, MoSe2, WS2, WSe2, MoTe2, or WTe2")
    parser.add_argument("--phase", type=str, required=True,
                        choices=["1H", "1T", "1T'"],
                        help="Phase of the TMD material (e.g., '1H' or '1T' or \"1T'\")")
    # Add arguments for parallel processing
    parser.add_argument("--nprocs", type=int, default=1,
                        help="Number of processors to use (default: 1)")
    parser.add_argument("--kpt-parallel", action="store_true", default=False,
                        help="Use k-point parallelization")
    parser.add_argument("--band-parallel", action="store_true", default=False,
                        help="Use band parallelization (for band structure calculation)")
    parser.add_argument("--domain-parallel", action="store_true", default=True,
                        help="Use domain parallelization (default: True)")
    # Add arguments for controlling calculation parameters
    parser.add_argument("--nbands", type=int, default=40,
                        help="Number of bands for band structure calculation (default: 24)")
    parser.add_argument("--conv-bands", type=int, default=16,
                        help="Number of bands for convergence check (default: 8)")
    parser.add_argument("--skip-relax", action="store_true", default=False,
                        help="Skip structure relaxation (use for continuation runs)")
    return parser.parse_args()

def build_tmd_structure(material, phase):
    """
    Build the monolayer TMD structure using customed version of ASE's mx2 builder.
    
    Args:
        material (str): The material name (e.g., "MoS2").
        phase (str): The phase, either "1H" or "1T" or "1T'".
    
    Returns:
        atoms (ase.Atoms): The constructed TMD monolayer.
    """
    
    # Retrieve material-specific parameters.
    params = TMD_parameters[material][phase]
    a =  params["a"]         # in Angstroms
    vacuum = params["vacuum"]

    # Build the monolayer structure. The "kind" parameter accepts values '1H', '1T' or 1T''
    atoms = mx2_custom(formula=material, kind=phase, a=a, vacuum=vacuum)

    # Adjust the in-plane lattice parameter if necessary.
    # Don't manually adjust the cell for 1T' since it's already set in mx2_custom
    if phase != '1T\'':
        # For 1H and 1T phases only - adjust cell if needed
        cell = atoms.get_cell()
        # For hexagonal cells, maintain the proper a parameter
        cell[0, 0] = a
        cell[1, 1] = a * 3**0.5 / 2  # Correct for hexagonal cell
        # Set the c lattice vector to ensure the desired vacuum spacing.
        cell[2, 2] = vacuum
        atoms.set_cell(cell, scale_atoms=True)

    return atoms

def relax_structure(atoms, material, phase, pw_cutoff, kpts, parallelization_params) -> ase.Atoms:
    """
    Perform ionic relaxation of the structure with multiprocessing support.
    
    After relaxation, the updated atomic positions (stored in "atoms")
    represent the optimized geometry.
    Args:
        atoms (ase.Atoms): The atomic structure.
        material (str): Material name.
        phase (str): Phase (1H or 1T).
        pw_cutoff (int): Plane wave cutoff energy.
        kpts (tuple): k-point sampling grid.
        parallelization_params (dict): Parameters for parallel computation.
    """
    # Check if relaxed structure already exists
    relaxed_file = f'{material.lower()}_{phase}_relaxed.gpw'
    if os.path.exists(relaxed_file):
        parprint(f"Found existing relaxed structure {relaxed_file}, loading...")
        atoms = GPAW(relaxed_file, txt=None).get_atoms()
        return atoms

    # When setting up the calculator for 1T' phase (DFT+U)
    if phase == "1T'":
        # Use "PBE" while including Hubbard correction via setups
        xc = "PBE"  # in case one want to introduce hybrid functionals (e.g., 'SCAN', 'HSE06')
        if material.startswith("W"):  # For W-based TMDs
            u_value = 2.8  # Literature-supported value for WTe2
            setups = {"W": f":d,{u_value}"}
        else:  # For Mo-based TMDs in 1T' phase
            u_value = 2.0  # Typical value for Mo d-orbitals
            setups = {"Mo": f":d,{u_value}"}
    else:
        xc, setups = "PBE", {}

    # Set up the GPAW calculator for relaxation with parallelization
    relax_calc = GPAW(
        mode=PW(pw_cutoff),
        kpts={'size': kpts, 'gamma': True},
        xc=xc,
        spinpol=True,
        # Always use spinpol=True for relaxations to allow the system to find its most stable magnetic configuration
        occupations=FermiDirac(0.01),
        symmetry='off',
        txt=f'{material.lower()}_{phase}_relax.txt',
        # Parallel settings
        parallel=parallelization_params,
        setups=setups
    )

    atoms.calc = relax_calc
    energy = atoms.get_potential_energy()  # Initial energy
    parprint(f"Initial energy before relaxation: {energy:.4f} eV")
    
    # Apply constraints if needed (e.g., fix bottom layer atoms or vacuum padding)
    # Example: fix z-position of atoms near the edge of vacuum
    # z_coords = [atom.position[2] for atom in atoms]
    # fixed_indices = [i for i, z in enumerate(z_coords) if z < 3.0 or z > (max(z_coords) - 3.0)]
    # atoms.set_constraint(FixAtoms(indices=fixed_indices))

    # Run BFGS optimization until forces are below {fmax} eV/Å
    optimizer = BFGS(atoms, trajectory=f'{material.lower()}_{phase}_relax.traj')
    optimizer.run(fmax=0.05)
    
    # Save the relaxed structure for record and downstream calculations
    relax_calc.write(relaxed_file)
    # Save structure in a format ASE can directly read
    write(f'{material.lower()}_{phase}_relaxed.traj', atoms)
    parprint("Relaxation completed, relaxed structure saved.")
    return atoms  # "atoms" now holds the relaxed geometry


def ground_state_calculation(atoms, material, phase, pw_cutoff, kpts, parallelization_params) -> ase.Atoms:
    """
    Calculate and plot the ground state of the relaxed structure for the given TMD system.
    We invoke a two-step approach for the 1T'-phase TMDs.
    
    In the alternative route (used for the challenging 1T'-phase):
      Step 1: A coarse, non-magnetic SCF calculation (with reduced k-points) is used
              to obtain a converged (unpolarized) charge density.
      Step 2: Spin polarization is turned on (with initial magnetic moments set on the metal
              atoms), and a fixed-density (non-selfconsistent) calculation is performed on a 
              full (denser) k-point set to obtain the eigenvalue spectrum and ground state energy.

    Args:
        atoms (ase.Atoms): The atomic structure.
        material (str): Material name.
        phase (str): Phase (1H or 1T).
        pw_cutoff (int): Plane wave cutoff energy.
        kpts (tuple): k-point sampling grid.
        parallelization_params (dict): Parameters for parallel computation.
    """
    spinpol = True
    # 1H phases are generally non-magnetic
    # Spin polarization is important for 1T phases which may have magnetic instabilities and
    # critical for 1T' phases which often have magnetic character.

    # When setting up the calculator for 1T' phase (DFT+U)
    if phase == "1T'":
        # Use "PBE" while including Hubbard correction via setups
        xc = "PBE" # in case one want to introduce hybrid functionals (e.g., 'SCAN', 'HSE06')
        if material.startswith("W"):  # For W-based TMDs
            u_value = 2.8  # Literature-supported value for WTe2
            setups = {"W": f":d,{u_value}"}
        else:  # For Mo-based TMDs in 1T' phase
            u_value = 2.0  # Typical value for Mo d-orbitals
            setups = {"Mo": f":d,{u_value}"}
        
        # For challenging 1T' phase, use multi-step approach
        parprint(f"Starting multi-step calculation for {phase}-{material}")

        # -----------------------
        # Step 1: Coarse non-magnetic SCF run to converge the density.
        # -----------------------
        # Use a much coarser k-point mesh for initial density
        initial_kpts = {'size': (5, 5, 1), 'gamma': True}
        atoms_initial = atoms.copy()
        
        # Clear any initial magnetic moments (ensure non-magnetic run)
        atoms_initial.set_initial_magnetic_moments(None)

        parprint(f"Step 1: Initial non-magnetic preconditioning with reduced k-points")
        calc_initial = GPAW(
            mode=PW(pw_cutoff),
            kpts=initial_kpts,
            xc=xc,
            spinpol=False,
            symmetry='off',
            occupations=FermiDirac(0.1),  # Larger smearing for stability
            txt=f'{material.lower()}_{phase}_initial.txt',
            parallel=parallelization_params,
            setups=setups,
            # More aggressive mixing parameters for faster convergence
            mixer=Mixer(beta=0.05, nmaxold=3, weight=100.0),
            convergence={'energy': 1e-3, 'density': 1e-2},  # Much looser convergence
            maxiter=100  # Limit iterations for this preconditioning step
        )

        atoms_initial.calc = calc_initial
        initial_energy = atoms_initial.get_potential_energy()
        parprint(f"Step 1 completed, initial energy: {initial_energy:.4f} eV")
        calc_initial.write(f'{material.lower()}_{phase}_initial.gpw')

        # -----------------------
        # Step 2: Intermediate spin-polarized calculation with moderate k-points
        # -----------------------
        parprint("Step 2: Intermediate nonmagnetic PBE without U...")
        
        # Set moderate magnetic moments
        # for atom in atoms:
        #     if atom.symbol == 'W' or atom.symbol == 'Mo':
        #         atom.magmom = 2.0  # Reduced initial guess
                #1.0 μB (initial guess based on previous calculations)
        
        # Use intermediate k-point mesh
        intermediate_kpts = {'size': (10, 10, 1), 'gamma': True}
        
        # Calculate a reasonable number of bands - typically 1.5x the number of valence electrons
        # For WTe2, with W having 6 valence electrons and Te having 6 each, 
        # we have about 18 electrons per formula unit.

        # Calculate bands based on valence electrons
        n_electrons = 0
        for atom in atoms:
            if atom.symbol in ['W', 'Mo', 'Te', 'S', 'Se']:
                n_electrons += 6
        
        setups_int = {}            # turn off Hubbard U
        spinpol_int = False       # nonmagnetic run
        
        nbands_est = max(int(n_electrons * 0.6), 40)
        
        calc_intermediate = GPAW(
            f'{material.lower()}_{phase}_initial.gpw',  # Start from previous step
            mode=PW(pw_cutoff),
            kpts=intermediate_kpts,
            xc=xc,
            spinpol=spinpol_int,
            occupations=FermiDirac(0.2), # Higher temperature smearing
            txt=f'{material.lower()}_{phase}_intermediate.txt',
            #  Adjusted mixer parameters for this phase
            mixer=Mixer(beta=0.03, nmaxold=5),
            convergence={'energy': 5e-4, 'density': 5e-4},
            nbands=nbands_est,
            maxiter=50,  # Limit iterations
            parallel=parallelization_params,
            setups=setups_int
        )

        atoms.calc = calc_intermediate
        inter_energy = atoms.get_potential_energy()
        parprint(f"Step 2 completed, intermediate energy: {inter_energy:.4f} eV")
        calc_intermediate.write(f'{material.lower()}_{phase}_intermediate.gpw')

        # -----------------------
        # Step 3: Final calculation with target k-points
        # -----------------------
        parprint("Step 3: Final calculation with full k-point set...")
        setups_final = {}          # U remains off
        spinpol_final = False     # nonmagnetic run (magnetic runs didn't converge with the above settings)

        # Use smearing to help with metallic character
        calc = GPAW(
            f'{material.lower()}_{phase}_intermediate.gpw',  # Start from intermediate step
            mode=PW(pw_cutoff),
            kpts={'size': kpts, 'gamma': True},
            xc=xc,  # or 'MVS' if available in your libxc,
            spinpol=spinpol_final,
            # soc=True for post-processing  (previously converged charge density as an initial guess)
            #######################
            # Research on WTe₂ often points out that SOC plays a key role in explaining 
            # experimental measurements (e.g., angle-resolved photoemission spectroscopy (ARPES) 
            # data) and theoretical predictions for topological behavior. 
            # Including SOC can help capture these effects more accurately.
            #######################
            occupations=FermiDirac(0.02),
            txt=f'{material.lower()}_{phase}_gs.txt',
            # Final mixer strategy - lower beta, more history
            mixer=Mixer(beta=0.04, nmaxold=8, weight=50.0),
            convergence={'energy': 1e-4, 'density': 5e-4},
            nbands=nbands_est,
            maxiter=50,
            parallel=parallelization_params,
            setups=setups_final
        )

    else:
        xc, setups = "PBE", {}
        # Ground state calculation for 1H and 1T phases
        calc = GPAW(
                        mode=PW(pw_cutoff),
                        kpts={'size': kpts, 'gamma': True},
                        xc=xc,
                        spinpol=spinpol,
                        #soc=True,        # Spin-orbit coupling ON
                        symmetry='off',  # Turn off symmetry to cover the full BZ
                        occupations=FermiDirac(0.01),  # Small broadening for better convergence
                        txt=f'{material.lower()}_{phase}_gs.txt',
                        nbands='150%',    # Extra bands for spinors
                        parallel=parallelization_params,
                        setups=setups
                    )
    atoms.calc = calc
    # Run SCF to obtain the ground state
    gs_energy = atoms.get_potential_energy()
    parprint(f"Calculated ground state energy for {material} {phase}: {gs_energy:.4f} eV")
    # Save the ground state calculation
    calc.write(f'{material.lower()}_{phase}_gs.gpw')

    return atoms

def calculate_band_structure(atoms, material, phase, nbands_value, conv_bands, 
                             parallelization_params):
    """
    Calculate the band structure along a high-symmetry path using the relaxed structure.
    
    Args:
        atoms (ase.Atoms): The atomic structure.
        material (str): Material name.
        phase (str): Phase (1H or 1T or 1T').
        nbands_value (int): Number of bands to calculate.
        conv_bands (int): Number of bands for convergence check.
        parallelization_params (dict): Parameters for parallel computation.
    """
    # Check if band structure calculation already exists
    bs_file = f"{material.lower()}_{phase}_bs.json"
    if os.path.exists(bs_file):
        parprint(f"Found existing band structure calculation {bs_file}, loading...")
        # Assuming GPAW >= 25.1.0 to load band structure from file
        # Load the saved band structure data
        with open(bs_file, 'r') as f:
            bs_data = json.load(f)
        
        # Create path based on phase
        if phase in ['1H', '1T']:
            bp = atoms.cell.bandpath('GMKG', npoints=50)
        elif phase == '1T\'':
            special_points = {
                'G': [0, 0, 0],
                'X': [0.5, 0, 0],
                'Y': [0, 0.5, 0],
                'S': [0.5, 0.5, 0]
            }
            # Let ASE build and interpolate the kpts array for you:
            bp = atoms.cell.bandpath('GXSYG',
                                     special_points=special_points,
                                     npoints=50)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        # Reconstruct energies from the JSON file
        energies_info = bs_data['energies']['__ndarray__']
        shape = energies_info[0]  # Expected to be [2, 50, 40]
        dtype = energies_info[1]  # Expected to be 'float64'
        flat_data = energies_info[2]  # The flat list of energy values
        # Convert the flat list to a NumPy array and reshape it
        energies_array = np.array(flat_data, dtype=dtype).reshape(shape)

        # Create ASE BandStructure object
        # ASE will plot both spin‐components if shape[0] > 1
        bs = ASEBandStructure(path=bp, energies=energies_array)

        # kpts = bs.path.kpts  # Get k-points from the band path
        # cell = atoms.cell  # Unit cell of the atoms

        # Plot the band structure
        fig, ax = plot_band_structure(bs, material, phase)
        if os.getpid() == os.getppid(): plt.show()
        return

    # (Optional) plot the band path for visualization
    # band_path_fig, band_path_ax = plot_band_path(atoms, phase, save_path=f"{material.lower()}_{phase}_bandpath.png")
    
    # Choose high symmetry path based on phase
    if phase in ['1H', '1T']:
        # Hexagonal lattice path
        bp = atoms.cell.bandpath('GMKG', npoints=50)

    elif phase == '1T\'':
        # For orthorhombic cell of 1T' structure
        # Define special points for rectangular Brillouin zone
        special_points = {
            'G': [0, 0, 0],
            'X': [0.5, 0, 0],
            'Y': [0, 0.5, 0], 
            'S': [0.5, 0.5, 0]
        }
        # Standard path for rectangular cell
        bp = atoms.cell.bandpath('GXSYG',
                                 special_points=special_points,
                                 npoints=50)
    else:
        raise ValueError(f"Unknown phase: {phase}")

    # Calculate band structure along high symmetry path
    # For band structure calculations, we want to use band parallelization
    band_parallel_params = parallelization_params.copy()
    if 'nprocs' in band_parallel_params and band_parallel_params['nprocs'] > 1:
        # Enable band parallelization for band structure calculation
        band_parallel_params['band'] = min(band_parallel_params.get('band', 1), 
                                           band_parallel_params['nprocs'])
        
    # Load the ground-state calculator file.
    gs_calc = GPAW(f"{material.lower()}_{phase}_gs.gpw", txt=None)    

    # Once the ground state density is converged, 
    # perform a non-selfconsistent band structure calculation including SOC 
    # (using spinorbit=True or the dedicated SOC routines in GPAW).

    # Restart from ground state and fix density: (or atoms.calc.fixed_density(...) )
    # Note: using a localized basis ('dzp') may need to be reconsidered if including SOC.
    # Restart from relaxed ground state and fix density.
    # The band structure calculation will automatically use the same exchange-correlation 
    # functional and DFT/DFT+U parameters that were used in the ground state calculation 
    # without needing to specify them again.

    # Perform the fixed density (non-SCF) band structure (Harris functional style)
    bs_calc = gs_calc.fixed_density(
                    nbands=nbands_value,
                    basis='dzp',
                    symmetry='off',
                    kpts=bp,             # Use the defined band path k-points for interpolation
                    convergence={'bands': conv_bands}, # convergence check for additional bands
                    parallel=band_parallel_params
                    )

    # Non-self‐consistent SOC diagonalization
    # Diagonalize the SOC Hamiltonian on the fixed density
    soc = soc_eigenstates(bs_calc, n1=0, n2=nbands_value)
    e_km = soc.eigenvalues()   # (Nk, 2*Nb)

    # Reshape to (spin, kpt, band)
    Nk = len(bp.kpts)
    Nb = nbands_value
    e_km = e_km.reshape(Nk, 2, Nb).transpose(1, 0, 2)  # => (2, Nk, Nb)

    # Build ASE BandStructure (positional args): (path, eigenvalues, [fermi_level])
    ef = bs_calc.get_fermi_level()
    parprint(f"Fermi level: {ef:.4f} eV")
    bs = ASEBandStructure(bp, e_km, ef)

    # (Optional) Overwrite the JSON so future loads skip regeneration
    bs.write(bs_file)

    ##################### if GPAW <= 24.6.0 ########################
    # Plot the band structure using GPAW's BandStructurePlotter.
    # plotter = BandStructurePlotter(bs)
    # plotter.plot(filename=f'{material}_{phase}_bandstructure.png')
    ################################################################

    # Plot the band structure
    fig, ax = plot_band_structure(bs, material, phase)
    # Only show the plot if this is the main process 
    # (i.e., not a child process in a multiprocessing run)
    if os.getpid() == os.getppid(): plt.show()
 
def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Extract parameters for the chosen material and phase
    material = args.material
    phase = args.phase
    params = TMD_parameters[material][phase]
    pw_cutoff = params['pw_cutoff']
    kpts = params['kpts']

    # Set up parallelization parameters without "nprocs"
    parallelization_params = {}

    # Configure parallelization strategy based on command-line args
    if args.nprocs > 1:
        if args.kpt_parallel:
            # Distribute k-points across processes
            parallelization_params['kpt'] = min(args.nprocs, params['kpts'][0] * params['kpts'][1])
        if args.band_parallel:
            # Distribute bands; default to 4 otherwise
            parallelization_params['band'] = min(args.nprocs, 4)
        if args.domain_parallel:
            # Domain parallelization: typically the grid is divided among a few processes
            parallelization_params['domain'] = min(args.nprocs, 2)

    parprint(f"Running calculation for {material} in {phase} phase with parallelization: {parallelization_params}")

    # Build initial TMD structure
    atoms = build_tmd_structure(material, phase)
    parprint(f"Built {material} monolayer in {phase} phase with literature lattice constant.")
    parprint("Pristine cell:")
    parprint(atoms.get_cell())

    # --- Step 1: Ionic Relaxation ---
    # Perform structure relaxation if not skipped
    if not args.skip_relax:
        atoms = relax_structure(atoms, material, phase, pw_cutoff, kpts, parallelization_params)
        # The relaxed structure is now stored in "atoms" and saved to a file 
        # (e.g., "wte2_1t'_relaxed.gpw")
        parprint("Relaxed cell:")
        parprint(atoms.get_cell())

    # --- Step 2: Ground State Calculation ---
    atoms = ground_state_calculation(atoms, material, phase, pw_cutoff, 
                                     kpts, parallelization_params)
    
    # --- Step 3: Band Structure Calculation ---
    calculate_band_structure(atoms, material, phase, args.nbands, args.conv_bands, 
                            parallelization_params)

    parprint(f"Built a {args.material} monolayer in the {args.phase} phase.")

if __name__ == '__main__':
    main()