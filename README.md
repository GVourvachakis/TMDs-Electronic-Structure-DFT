# TMD Electronic Structure Analysis using GPAW DFT

## Project Overview

This repository contains the computational framework and results for the systematic study of electronic properties of group-VI two-dimensional transition metal dichalcogenides (TMDs) using density functional theory (DFT). The project was developed as part of the [**MEMY-512 Computational Materials Science II**](https://mscs.uoc.gr/dmst/?courses=computational-materials-science-ii) course at the University of Crete.

> **📋 For Detailed Documentation**: For a comprehensive description of our development process, theoretical background, computational setup, and detailed analysis methodology, please refer to the complete project report: **[`tmd_report.pdf`](./tmd_report.pdf)**. The PDF contains extensive details on our thought process, validation procedures, and in-depth discussion of results that complement this markdown.

### Research Objectives

The primary goal is to calculate and analyze the electronic structure properties of TMD monolayers, specifically:

- **Band structure calculations** along high-symmetry k-point paths
- **Total density of states (DOS)** analysis
- **Orbital-projected DOS (PDOS)** to understand electronic contributions
- **Phase-dependent electronic properties** across different structural polymorphs

### Materials and Phases Studied

**1H Phase Materials (Semiconducting):**
- MoS₂ (Molybdenum disulfide)
- MoSe₂ (Molybdenum diselenide) 
- WS₂ (Tungsten disulfide)
- WSe₂ (Tungsten diselenide)
- MoTe₂ (Molybdenum ditelluride)
- WTe₂ (Tungsten ditelluride)

**Extended Phase Analysis:**
- **1T phase WTe₂** (Metallic octahedral coordination)
- **1T' phase WTe₂** (Distorted 1T with potential topological properties)

## Repository Structure

```
.
├── tmd_report.pdf           ← Project description
├── tmd_pres_slides.pdf.pdf  ← Presentation slides
├── requirements.yaml        ← Conda environment  
└── TMD_codes/               ← All scripts, data & workflows  
    ├── tmd_bs.py            ← Band structure calculator  
    ├── dos_tmd.py           ← DOS & PDOS plotting module  
    ├── mx2_utils.py         ← Custom ASE MX₂ builder + helpers  
    ├── tmd_prior_runs/      ← Reference `.gpw` runs & band‐path images  
    ├── fails/               ← Divergent or unconverged calculations  
    └── MX2_<metal><X>_<p>/  ← Material-phase subdirectories  
        ├── initial/         ← “Guess” structures & input scripts  
        ├── relax/           ← Relaxation outputs (`*.gpw`, `.traj`)  
        ├── gs/              ← Ground‐state densities (`*_gs.gpw`)  
        ├── bs/              ← Band‐structure JSON + plots  
        ├── dos/             ← DOS/PDOS data & figures  
        └── img/             ← All generated figures (band paths, DOS, SOC insets)
```

### Directory Organization

The project employs a **hierarchical directory structure** that mirrors the computational workflow:

- **Material-specific subdirectories** follow the naming convention `MX2_1p`, where:
  - `M` = transition metal (Mo, W)
  - `X` = chalcogen (S, Se, Te)  
  - `p` = structural phase (H, T, T')

- **Systematic file naming** encodes material identity and calculation type
- **Complete computational provenance** from initial structural guesses to converged geometries
- **Visualization consolidation** in `img/` subdirectories containing DOS plots, band diagrams, and orbital projections

## Computational Modules

### 1. Band Structure Calculator (`tmd_bs.py`)

Performs electronic band structure calculations for TMD monolayers with support for all three structural phases.

**Key Features:**
- Customized ASE MX2 builder distinguishing 1H from 2H phases
- Support for distorted 1T' phase construction
- Integrated workflow: structure building → relaxation → SCF → band calculation
- High-symmetry k-point path analysis (Γ-K-M-Γ)

**Usage:**
```bash
# Basic band structure calculation
mpirun -np 8 python tmd_bs.py --material MoS2 --phase "1H" --nprocs 4 --domain-parallel --kpt-parallel

# Skip relaxation for pre-optimized structures
mpirun -np 8 python tmd_bs.py --material WTe2 --phase "1T'" --skip-relax --band-parallel
```

**Supported Arguments:**
- `--material`: TMD compound (MoS2, MoSe2, WS2, WSe2, MoTe2, WTe2)
- `--phase`: Structural phase (1H, 1T, 1T')
- `--nprocs`: MPI process count
- `--domain-parallel`, `--kpt-parallel`, `--band-parallel`: Parallelization options
- `--skip-relax`: Skip geometry optimization

### 2. DOS Analysis Module (`dos_tmd.py`)

Computes total and projected density of states from completed GPAW calculations.

**Key Features:**
- Total DOS with spin-channel resolution
- Orbital-projected DOS (metal d-orbitals, chalcogen p-orbitals)
- Spin-orbit coupling (SOC) analysis for heavy elements
- Band inversion analysis for 1T' phases

**Usage:**
```bash
# Basic DOS calculation
python dos_tmd.py mos2_1H_gs.gpw

# Full analysis with PDOS and SOC
python dos_tmd.py wte2_1T_gs.gpw --pdos --soc --width 0.05

# Custom energy range
python dos_tmd.py "mote2_1T'_gs.gpw" --emin -8 --emax 8 --resolution 0.01
```

**Analysis Options:**
- `--pdos`: Enable orbital-projected DOS
- `--soc`: Spin-orbit coupling analysis
- `--width`: DOS broadening (default: 0.1 eV)
- `--emin/emax`: Energy window relative to Fermi level

## Phase-Specific Analysis

### 1H Phase (Semiconducting)
- **Electronic character**: Direct-gap semiconductors
- **Critical points**: Γ and K valley analysis
- **Orbital contributions**: Metal d-orbital and chalcogen p-orbital hybridization

### 1T Phase (Metallic)
- **Electronic character**: Metallic or semi-metallic
- **Band features**: d-orbital dominated conduction
- **Magnetic properties**: Potential spin-polarization

### 1T' Phase (Topological)
- **Electronic character**: Distorted 1T with metal-metal chains
- **SOC effects**: Spin-orbit driven band inversions
- **Topological analysis**: Band inversion around ±0.5 eV

## Environment Setup

### Prerequisites

- **GPAW** ≥ 25.1.0 with MPI support
- **ASE** (Atomic Simulation Environment)
- **Python** environment with scientific computing stack

### Installation

```bash
# Create conda environment
conda env create -f requirements.yaml
conda activate tmd-dft

# Verify GPAW installation
python -c "import gpaw; print(gpaw.__version__)"
```

### Computational Requirements

- **MPI-enabled** GPAW installation for parallel calculations
- **Memory**: Varies by system size and k-point sampling
- **Time**: Band structure calculations typically require 2-8 hours on modern clusters

## Reproducibility and Validation

The repository is designed for **complete reproducibility** of results presented in `tmd_report.pdf`. Users can:

1. **Verify existing calculations** using provided reference data in `tmd_prior_runs/`
2. **Reproduce specific results** by following the systematic workflow
3. **Extend to new materials** using the modular calculation framework
4. **Test alternative parameters** guided by the analysis in `fails/` directory

### Data Integrity

- **Complete provenance tracking** from initial guesses to final results
- **Systematic naming conventions** for easy data management
- **Benchmark comparisons** through established reference calculations
- **Parameter sensitivity analysis** documented in failed calculation logs

## Output and Visualization

The computational workflow generates:

1. **Band structure plots** along high-symmetry paths
2. **Total DOS** with spin-channel resolution
3. **Orbital-projected DOS** showing elemental and orbital contributions
4. **SOC analysis plots** for heavy-element compounds
5. **Phase comparison visualizations** highlighting electronic differences

### Specialized Analyses

- **1H phases**: Band gap characterization and valley physics
- **1T phases**: Metallic band analysis and d-orbital splitting
- **1T' phases**: Topological band inversion and SOC-driven transitions

## Contributing and Extension

The modular design enables straightforward extension to:
- **Additional TMD materials** (group-V TMDs, heterostructures)
- **Enhanced DFT methods** (hybrid functionals (e.g., B3LYP), GW-BSE corrections)
- **Advanced analysis techniques** (topological invariants, transport properties)

## References and Acknowledgments

This work was completed as part of the Computational Materials Science II course curriculum, utilizing the open-source GPAW DFT package and following established best practices for reproducible computational materials research.

## 📜 License & Citation

If you use these scripts or data, please cite:

Vourvachakis S. Georgios, “Band structure, total DOS, and orbital-projected DOS of group VI 2D TMDs…”, Department of Materials Science and Engineering, University of Crete, 2025.

