# Q-T Analysis with MPI Parallelization

## Overview
This project computes the **Q and T statistical measures** of masked full-sky density maps using **MPI-based parallelization**. It is designed for high-volume Monte Carlo simulations over randomized density fields, using smoothing over masks and antipodal pairing techniques for sky analysis.

The code uses `mpi4py` for distributed processing across multiple CPUs.

## Repository Structure
- `qstat_sim.py`: Main MPI-parallelized driver for performing simulations.
- `Tmax_helperfunctions.py`: Supporting functions for Q and T analysis (Q-measure, etc.).
- `mask_helperfunctions.py`: Functions for preparing and smoothing binary masks.
- `sim_data/`: Directory for saving output CSVs and log files (created during execution).
- `qstat_run_script.sh`: job script (.sh) for submition to a cluster

## Main Features
- **Real-space smoothing** of binary masks.
- **Randomization** of sky maps for each realization.
- **Parallel computation** of Qmax, Tmax, Qmin, Tmin statistics across multiple processes.
- **Periodic saving** of intermediate results to avoid data loss.
- **Logging** of configuration details.

## Installation

1. **Environment requirements**:
    - Python 3.8+
    - mpi4py
    - numpy
    - healpy
    - pandas
    - lmfit
    - scipy

2. **Installation**:
    ```bash
    pip install mpi4py numpy healpy pandas lmfit scipy
    ```

3. **MPI Setup**:
    You must have an MPI implementation installed, e.g., `OpenMPI` or `MPICH`.

## How to Run

Execute the script using `mpirun` or `mpiexec`:
```bash
mpirun -np <num_processes> python qstat_sim.py
```
Where `<num_processes>` is the number of parallel processes you want (e.g., 4, 8, 16).

### Example:
```bash
mpirun -np 8 python qstat_sim.py
```
On clusters

```bash
sbatch qstat_run_script.py
```

## Input Files

- **Density Map**: `densitymap_eclcor_nocmb_nomonopole.txt` map to be evaluated
- **Unmask Map**: `unmask.txt` pixel values with masked pixel = 0 otherwise 1

These files should be placed in the same directory or updated in `qstat_sim.py` accordingly.

## Output

Results are saved into the `sim_data/` directory, with files like:
- `Qmax_<map>_<smoothing>_nsim_<num_sim>_delta_piby_<delta1>_rank_<rank>.csv`
- `Tmax_<map>_<smoothing>_nsim_<num_sim>_delta_piby_<delta1>_rank_<rank>.csv`
- (and similarly for Qmin and Tmin)

Each MPI process writes its own results.

## Parameters
Editable directly inside `qstat_sim.py`:
- `NSIDE`: HEALPix resolution (`64` by default).
- `nsim`: Number of simulations (`10000` by default).
- `delta1`: Angular scale divisor (`3/2` by default, meaning delta = pi/(3/2)).
- `smoothing`: Type of smoothing applied (`gaussian`, `tophat`, or `gaussian-tophat`).

## Notes
- The `Q_measure` function (in `helperfunctions.py`) performs core analysis using pixel pairs.
- Mask smoothing is done in real-space, not harmonic-space.
- The full density map is shuffled randomly for each realization before computing Q and T.
- Make sure your density and mask maps have matching `NSIDE`.

## License
MIT License (or specify otherwise).

## Acknowledgments
- HEALPix developers for the pixelization scheme.
- mpi4py community for Python MPI bindings.

