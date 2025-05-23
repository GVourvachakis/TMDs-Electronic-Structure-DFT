#!/usr/bin/env python3
"""
TMD DOS and PDOS Calculator

This module provides a modular framework to read a GPAW GPW file from band structure
calculations of transition metal dichalcogenide (TMD) monolayers, compute the total
and projected density of states (DOS/PDOS), and perform specialized analyses for different
TMD phases (e.g., 1H, 1T, 1T'). The script is organized into separate functions for 
argument parsing, calculator initialization, energy grid preparation, and plotting of 
both total DOS and orbital-projected DOS, with optional spin-orbit coupling (SOC)
analysis.

Module Structure:
  - parse_args(): Parses command-line arguments.
  - load_calculator(): Loads the GPAW calculator and ensures convergence.
  - setup_energy_grid(): Creates an energy grid relative to the Fermi level.
  - plot_total_dos(): Computes and plots the total DOS (handling spin channels if required).
  - plot_projected_dos(): Calculates and plots the orbital-projected DOS for metal and 
    chalcogen elements.
  - analyze_soc(): Performs additional SOC analysis for materials with heavy elements 
    and, when applicable, analyzes band inversion in 1T' phases.
  - main(): Coordinates the workflow based on the selected options.

Prerequisites:
  - A GPAW calculation file (*.gpw) from a completed ground state calculation for a TMD
    monolayer material (e.g., MoS2, MoSe2, WS2, WSe2, MoTe2, WTe2).
  - The calculation should be performed with the appropriate parameters for the specific TMD
    phase (1H, 1T, 1T').
  - For SOC analysis, the ground state calculation should include spin-orbit coupling.

Usage:
  $ python plot_dos_tmd.py <gpw_filename> [options]

Arguments:
  <gpw_filename>          Filename of the saved GPAW calculator state.

Options:
  --width FLOAT           Broadening width in eV (default: 0.1).
  --emin FLOAT            Minimum energy relative to Fermi level (default: -6 eV).
  --emax FLOAT            Maximum energy relative to Fermi level (default: 6 eV).
  --resolution FLOAT      Energy grid resolution in eV (default: 0.01).
  --pdos                  Calculate projected DOS (orbital- and element-resolved).
  --soc                   Perform additional spin-orbit coupling analysis.

Examples:
  $ python plot_dos_tmd.py mos2_1H_gs.gpw
  $ python plot_dos_tmd.py wte2_1T_gs.gpw --pdos --soc
  $ python plot_dos_tmd.py "mote2_1T'_gs.gpw" --pdos --soc --width 0.05 --emin -8 --emax 8

Output:
  The script generates several plots based on the chosen options:
  1. Total DOS plot:
     - Spin-polarized calculations: Separate DOS curves for spin up and spin down.
     - Non-spin-polarized calculations: A single total DOS curve.
  2. Orbital-Projected DOS plot (if --pdos is specified):
     - Contributions from metal d-orbitals and chalcogen p-orbitals.
  3. SOC analysis plots (if --soc is specified):
     - Focused analysis of DOS near the Fermi level to highlight SOC effects.
     - For 1T' phases, additional band inversion analysis around ±0.5 eV.

Phase-specific analysis:
  - 1H phases: Typically non-magnetic; focus on the band gap.
  - 1T phases: Potentially magnetic; analysis of d-orbital splitting.
  - 1T' phases: SOC-driven band inversion and topological characteristics are analyzed,
    particularly near the Fermi level.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from gpaw import GPAW
from gpaw.dos import DOSCalculator

from tmd_bs import setup_gpaw_paths

setup_gpaw_paths()

def parse_args():
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate and plot DOS and PDOS for TMD materials'
    )
    parser.add_argument('gpw_file', help='GPAW calculation file (.gpw)')
    parser.add_argument('--width', type=float, default=0.1, help='Broadening width in eV')
    parser.add_argument('--emin', type=float, default=-6, help='Minimum energy relative to Fermi level')
    parser.add_argument('--emax', type=float, default=6, help='Maximum energy relative to Fermi level')
    parser.add_argument('--resolution', type=float, default=0.01, help='Energy grid resolution in eV')
    parser.add_argument('--pdos', action='store_true', help='Calculate projected DOS')
    parser.add_argument('--soc', action='store_true', help='Analyze SOC effects (j-resolved PDOS)')
    return parser.parse_args()

def load_calculator(gpw_file):
    """Load the GPAW calculator from the specified file and force data loading."""
    calc = GPAW(gpw_file)
    calc.get_potential_energy()  # Ensures that the calculation is converged and data is loaded
    return calc

def setup_energy_grid(doscalc, fermi_level, args):
    """
    Create an energy grid for DOS calculations.
    
    Parameters:
        doscalc: A DOSCalculator object.
        fermi_level: The Fermi level from the calculator.
        args: Parsed command-line arguments.
    
    Returns:
        energies_abs: Absolute energies including the Fermi offset.
        energies_rel: Energies relative to the Fermi level.
    """
    npoints = int((args.emax - args.emin) / args.resolution)
    energies_abs = doscalc.get_energies(
        emin=args.emin + fermi_level,
        emax=args.emax + fermi_level,
        npoints=npoints
    )
    energies_rel = energies_abs - fermi_level
    return energies_abs, energies_rel

def plot_total_dos(doscalc, energies_abs, energies_rel, args, nspins, fermi_level):
    """Calculate, plot, and save the total DOS."""
    plt.figure(figsize=(10, 6))
    
    if nspins == 2:
        dos_up = doscalc.raw_dos(energies_abs, spin=0, width=args.width)
        dos_dn = doscalc.raw_dos(energies_abs, spin=1, width=args.width)
        plt.plot(energies_abs, dos_up, label='Spin Up', color='red')
        plt.plot(energies_abs, -dos_dn, label='Spin Down', color='blue')

        # Numerical integration for each spin channel
        mask = energies_rel <= 0
        N_up = np.trapezoid(dos_up[mask], energies_rel[mask])
        N_dn = np.trapezoid(dos_dn[mask], energies_rel[mask])
        delta_N = N_up - N_dn

        print("Integrated electron count up to Fermi level:")
        print(f"  Spin up:   {N_up:.3f} electrons")
        print(f"  Spin down: {N_dn:.3f} electrons")
        print(f"Net spin polarization: {delta_N:.3f} electrons")
        print(f"Estimated net magnetic moment: {delta_N:.3f} μB per unit cell")
    else:
        dos_total = doscalc.raw_dos(energies_abs, width=args.width)
        plt.plot(energies_rel, dos_total, label='Total DOS', color='black')
        print("Non-spin-polarized calculation; net magnetic moment is 0.")

    plt.xlabel(r"$E - E_F$ (eV)")
    plt.ylabel("DOS (states/eV)")
    plt.title(f"Total DOS - {args.gpw_file}")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    total_dos_filename = f"DOS_total_{args.gpw_file.replace('.gpw', '')}.png"
    plt.savefig(total_dos_filename)
    print(f"Saved total DOS to {total_dos_filename}")


def sum_pdos(doscalc, energies_abs, energies_rel, args, indices, orbital, nspins, spin=None):
    """
    Sum the projected DOS over selected atomic indices for a given orbital.
    
    Parameters:
        doscalc: DOSCalculator object.
        energies_abs: Array of absolute energy values.
        energies_rel: Array of energies relative to the Fermi level.
        args: Parsed arguments.
        indices: List of atom indices.
        orbital: Orbital index (e.g., 1 for p, 2 for d).
        nspins: Number of spins in the calculation.
        spin: Specific spin channel (if applicable).
    
    Returns:
        pdos_sum: The summed PDOS over the provided atom indices.
    """
    pdos_sum = np.zeros_like(energies_rel)
    for a in indices:
        if nspins == 2:
            pdos_sum += doscalc.raw_pdos(energies_abs, a, orbital, spin=spin, width=args.width)
        else:
            pdos_sum += doscalc.raw_pdos(energies_abs, a, orbital, width=args.width)
    return pdos_sum


def plot_projected_dos(doscalc, energies_abs, energies_rel, args, nspins, atoms):
    """
    Calculate, plot, and save the orbital-projected DOS.
    
    Returns:
        metal_indices, chalcogen_indices, metal_elements, chalcogen_elements for further use.
    """
    element_symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(element_symbols))
    # Identify metal and chalcogen elements (adjust as necessary)
    metal_elements = [el for el in unique_elements if el in ('Mo', 'W')]
    chalcogen_elements = [el for el in unique_elements if el in ('S', 'Se', 'Te')]
    metal_indices = [i for i, sym in enumerate(element_symbols) if sym in metal_elements]
    chalcogen_indices = [i for i, sym in enumerate(element_symbols) if sym in chalcogen_elements]

    plt.figure(figsize=(12, 8))
    
    if nspins == 2:
        metal_d_up = sum_pdos(doscalc, energies_abs, energies_rel, args, metal_indices, orbital=2, nspins=nspins, spin=0)
        metal_d_down = sum_pdos(doscalc, energies_abs, energies_rel, args, metal_indices, orbital=2, nspins=nspins, spin=1)
        chalcogen_p_up = sum_pdos(doscalc, energies_abs, energies_rel, args, chalcogen_indices, orbital=1, nspins=nspins, spin=0)
        chalcogen_p_down = sum_pdos(doscalc, energies_abs, energies_rel, args, chalcogen_indices, orbital=1, nspins=nspins, spin=1)
        
        plt.plot(energies_abs, metal_d_up, label=f'{metal_elements[0]} d (up)', color='darkgreen')
        plt.plot(energies_abs, -metal_d_down, label=f'{metal_elements[0]} d (down)', color='lightgreen')
        plt.plot(energies_abs, chalcogen_p_up, label=f'{chalcogen_elements[0]} p (up)', color='darkred')
        plt.plot(energies_abs, -chalcogen_p_down, label=f'{chalcogen_elements[0]} p (down)', color='salmon')
    else:
        metal_d = sum_pdos(doscalc, energies_abs, energies_rel, args, metal_indices, orbital=2, nspins=nspins)
        chalcogen_p = sum_pdos(doscalc, energies_abs, energies_rel, args, chalcogen_indices, orbital=1, nspins=nspins)
        plt.plot(energies_abs, metal_d, label=f'{metal_elements[0]} d', color='green')
        plt.plot(energies_abs, chalcogen_p, label=f'{chalcogen_elements[0]} p', color='red')

    plt.xlabel(r"$E - E_F$ (eV)")
    plt.ylabel("PDOS (states/eV)")
    plt.title(f"Orbital-Projected DOS - {args.gpw_file}")
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    pdos_filename = f"PDOS_orbital_{args.gpw_file.replace('.gpw', '')}.png"
    plt.savefig(pdos_filename)
    print(f"Saved orbital PDOS to {pdos_filename}")
    
    return metal_indices, chalcogen_indices, metal_elements, chalcogen_elements

def analyze_soc(doscalc, energies_abs, energies_rel, args, nspins,
                metal_indices, chalcogen_indices, metal_elements, chalcogen_elements, atoms):
    """
    Perform SOC analysis if heavy elements are present and plot the results.
    """
    element_symbols = atoms.get_chemical_symbols()
    unique_elements = sorted(set(element_symbols))
    has_heavy_elements = any(el in ('W', 'Te', 'Se') for el in unique_elements)
    
    if has_heavy_elements:
        print("Analyzing SOC effects for material with heavy elements...")
        # Define the Fermi window (relative energies in eV)
        fermi_window = (-1.0, 1.0)
        mask = (energies_abs >= fermi_window[0]) & (energies_abs <= fermi_window[1])

        plt.figure(figsize=(10, 6))
        if nspins == 2:
            metal_d_up = sum_pdos(doscalc, energies_abs, energies_rel, args, metal_indices, orbital=2, nspins=nspins, spin=0)
            metal_d_down = sum_pdos(doscalc, energies_abs, energies_rel, args, metal_indices, orbital=2, nspins=nspins, spin=1)
            chalcogen_p_up = sum_pdos(doscalc, energies_abs, energies_rel, args, chalcogen_indices, orbital=1, nspins=nspins, spin=0)
            chalcogen_p_down = sum_pdos(doscalc, energies_abs, energies_rel, args, chalcogen_indices, orbital=1, nspins=nspins, spin=1)
            
            plt.plot(energies_abs[mask], metal_d_up[mask], label=f'{metal_elements[0]} d (up)', color='darkgreen')
            plt.plot(energies_abs[mask], -metal_d_down[mask], label=f'{metal_elements[0]} d (down)', color='lightgreen')
            plt.plot(energies_abs[mask], chalcogen_p_up[mask], label=f'{chalcogen_elements[0]} p (up)', color='darkred')
            plt.plot(energies_abs[mask], -chalcogen_p_down[mask], label=f'{chalcogen_elements[0]} p (down)', color='salmon')
        else:
            metal_d = sum_pdos(doscalc, energies_abs, energies_rel, args, metal_indices, orbital=2, nspins=nspins)
            chalcogen_p = sum_pdos(doscalc, energies_abs, energies_rel, args, chalcogen_indices, orbital=1, nspins=nspins)
            plt.plot(energies_abs[mask], metal_d[mask], label=f'{metal_elements[0]} d', color='green')
            plt.plot(energies_abs[mask], chalcogen_p[mask], label=f'{chalcogen_elements[0]} p', color='red')
        
        plt.xlabel(r"$E - E_F$ (eV)")
        plt.ylabel("PDOS (states/eV)")
        plt.title(f"SOC Analysis Near Fermi Level - {args.gpw_file}")
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        soc_filename = f"PDOS_SOC_{args.gpw_file.replace('.gpw', '')}.png"
        plt.savefig(soc_filename)
        print(f"Saved SOC analysis to {soc_filename}")
        
        # Special analysis for 1T' phases
        if "1t" in args.gpw_file.lower() and "'" in args.gpw_file:
            print("Analyzing band inversion in 1T' phase...")
            tight_window = (-0.5, 1.1) # depends on the band inversion region
            tight_mask = (energies_abs >= tight_window[0]) & (energies_abs <= tight_window[1])
            plt.figure(figsize=(10, 6))
            
            if nspins == 2:
                plt.plot(energies_abs[tight_mask], metal_d_up[tight_mask],
                         label=f'{metal_elements[0]} d (up)', color='darkgreen')
                plt.plot(energies_abs[tight_mask], -metal_d_down[tight_mask],
                         label=f'{metal_elements[0]} d (down)', color='lightgreen')
                plt.plot(energies_abs[tight_mask], chalcogen_p_up[tight_mask],
                         label=f'{chalcogen_elements[0]} p (up)', color='darkred')
                plt.plot(energies_abs[tight_mask], -chalcogen_p_down[tight_mask],
                         label=f'{chalcogen_elements[0]} p (down)', color='salmon')
            else:
                plt.plot(energies_abs[tight_mask], metal_d[tight_mask],
                         label=f'{metal_elements[0]} d', color='green')
                plt.plot(energies_abs[tight_mask], chalcogen_p[tight_mask],
                         label=f'{chalcogen_elements[0]} p', color='red')
            
            plt.xlabel(r"$E - E_F$ (eV)")
            plt.ylabel("PDOS (states/eV)")
            plt.title(f"Band Inversion Analysis (1T' Phase) - {args.gpw_file}")
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='--', alpha=0.7)
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            inversion_filename = f"PDOS_inversion_{args.gpw_file.replace('.gpw', '')}.png"
            plt.savefig(inversion_filename)
            print(f"Saved band inversion analysis to {inversion_filename}")

def main():
    args = parse_args()
    
    # Set up calculator and retrieve fundamental properties
    calc = load_calculator(args.gpw_file)
    fermi_level = calc.get_fermi_level()
    nspins = calc.get_number_of_spins()
    atoms = calc.get_atoms()
    
    # Create DOSCalculator instance (SOC flag set to False in this example)
    doscalc = DOSCalculator.from_calculator(calc, soc=False)
    energies_abs, energies_rel = setup_energy_grid(doscalc, fermi_level, args)
    
    # Total DOS calculation and plot
    plot_total_dos(doscalc, energies_abs, energies_rel, args, nspins, fermi_level)
    
    # Orbital projected DOS (PDOS) calculation and plot
    if args.pdos:
        metal_indices, chalcogen_indices, metal_elements, chalcogen_elements = plot_projected_dos(
            doscalc, energies_abs, energies_rel, args, nspins, atoms
        )
        
        # SOC analysis if requested
        if args.soc:
            analyze_soc(doscalc, energies_abs, energies_rel, args, nspins,
                        metal_indices, chalcogen_indices, metal_elements, chalcogen_elements, atoms)
                        
    print("Analysis complete!")
 
if __name__ == "__main__":
    main()