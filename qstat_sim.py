#### The code calculates the Q and T analysis of the masked full sky data using MPI parallelization.

from mpi4py import MPI
import numpy as np
import healpy as hp
import pandas as pd
import time
from Tmax_helperfunctions import *
from mask_helperfunctions import *

# -----------------------------
# Function Definitions
# -----------------------------

def log_run_details(filename, densitymap_file, delta, nsim, smoothing, output_files):
    """
    Log run configuration details into a specified file.

    Parameters:
        filename (str): Path to the log file.
        densitymap_file (str): Input density map file.
        delta (float): Angular scale (in radians) parameter.
        nsim (int): Number of simulations.
        smoothing (str): Smoothing method applied.
        output_files (dict): Dictionary of output file names.
    """
    with open(filename, 'a') as log_file:
        log_file.write(f"Run Details\n")
        log_file.write(f"=============\n")
        log_file.write(f"Density Map File: {densitymap_file}\n")
        log_file.write(f"Delta (radians): pi/{delta}\n")
        log_file.write(f"Smoothing window: {smoothing}\n")
        log_file.write(f"Number of Simulations (nsim): {nsim}\n")
        log_file.write(f"Output Files:\n")
        for key, value in output_files.items():
            log_file.write(f"  {key}: {value}\n")
        log_file.write(f"=============\n")


def generate_smoothmask_realspace(mask_map, mask_nside, delta_mask, smoothing):
    """
    Generate a real-space smoothed version of the mask map.

    Parameters:
        mask_map (array): Binary mask array.
        mask_nside (int): HEALPix nside resolution of the mask.
        delta_mask (float): Angular window (radians) for smoothing.
        smoothing (str): Smoothing method (currently unused, placeholder for flexibility).

    Returns:
        DataFrame: Smoothed mask result containing Q-measure.
    """
    unmask = mask_map == 1

    m = np.arange(hp.nside2npix(mask_nside))
    theta, phi = hp.pix2ang(mask_nside, m)
    phi = phi[unmask]
    theta = theta[unmask]
    vec = hp.ang2vec(theta=theta, phi=phi)
    theta2_antipodal, phi2_antipodal = hp.vec2ang(-vec)
    pix = hp.ang2pix(nside=NSIDE, theta=theta2_antipodal, phi=phi2_antipodal)
    mask_patch = [unmask[x] for x in pix]

    phi = phi[mask_patch]
    theta = theta[mask_patch]

    map_masked = mask_map[unmask]
    map_masked = map_masked[mask_patch]
    
    # Perform Q-measure over the selected mask
    df_mask = Q_measure(map_masked, theta1=theta, phi1=phi, delta=delta_mask, smoothing=smoothing)
    return df_mask


def calculate_qmax_per_realization(densitymap_sim, mask_map_smooth, unmask, delta, smoothing, rng):
    """
    Calculate Qmax, Tmax, Qmin, Tmin for one randomized realization of the density map.

    Parameters:
        densitymap_sim (array): Simulated density map values.
        mask_map_smooth (DataFrame): Smoothed mask map.
        unmask (array): Boolean mask indicating unmasked pixels.
        delta (float): Angular window (radians).
        smoothing (str): Smoothing method.
        rng (Generator): Random number generator instance.

    Returns:
        Tuple (DataFrames): Qmax, Tmax, Qmin, Tmin dataframes.
    """
    densitymap_sim_scaled = np.copy(densitymap_sim)

    m = np.arange(hp.nside2npix(NSIDE))
    theta, phi = hp.pix2ang(NSIDE, m)
    phi = phi[unmask]
    theta = theta[unmask]
    vec = hp.ang2vec(theta=theta, phi=phi)
    theta2_antipodal, phi2_antipodal = hp.vec2ang(-vec)
    pix = hp.ang2pix(nside=NSIDE, theta=theta2_antipodal, phi=phi2_antipodal)
    mask_patch = [unmask[x] for x in pix]

    phi = phi[mask_patch]
    theta = theta[mask_patch]
    densitymap_sim_masked = densitymap_sim_scaled[mask_patch]
    rng.shuffle(densitymap_sim_masked)

    df_Qsim = Q_measure(densitymap_sim_masked, theta, phi, delta, smoothing)
    
    # Normalize simulated Q values by the mask's Q values
    df_Qsim.Q = df_Qsim.Q / mask_map_smooth.Q
    df_Qsim.Q_d = df_Qsim.Q_d / mask_map_smooth.Q_d
    df_Qsim.delta_Qd = df_Qsim.Q - df_Qsim.Q_d

    # Find maximum and minimum locations
    Qmax = df_Qsim[df_Qsim['Q'] == df_Qsim['Q'].max()]
    Tmax = df_Qsim[df_Qsim['delta_Qd'] == df_Qsim['delta_Qd'].max()]
    Qmin = df_Qsim[df_Qsim['Q'] == df_Qsim['Q'].min()]
    Tmin = df_Qsim[df_Qsim['delta_Qd'] == df_Qsim['delta_Qd'].min()]
    
    return Qmax, Tmax, Qmin, Tmin


def calculate_qmax_mpi(densitymap, mask_map_smooth, unmask, delta1, nsim, output_files, smoothing):
    """
    Perform distributed MPI computation of Qmax, Tmax, Qmin, Tmin across multiple realizations.

    Parameters:
        densitymap (array): Full-sky density map (masked values).
        mask_map_smooth (DataFrame): Smoothed mask.
        unmask (array): Boolean mask.
        delta1 (float): Delta divisor (delta = pi/delta1).
        nsim (int): Total number of realizations.
        output_files (dict): Output file templates with {rank} placeholders.
        smoothing (str): Smoothing method applied.

    Output:
        Saves intermediate and final simulation results to CSV files.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    rng = np.random.default_rng(seed=rank + 100)

    # Distribute realizations across available MPI ranks
    realizations_per_process = nsim // size
    extra_realizations = nsim % size
    if rank < extra_realizations:
        realizations_per_process += 1

    delta = np.pi / delta1

    # Define output file paths specific to each rank
    filename_Qmax = output_files["Qmax"].replace("{rank}", str(rank))
    filename_Tmax = output_files["Tmax"].replace("{rank}", str(rank))
    filename_Qmin = output_files["Qmin"].replace("{rank}", str(rank))
    filename_Tmin = output_files["Tmin"].replace("{rank}", str(rank))

    Qmax_sim, Tmax_sim, Qmin_sim, Tmin_sim = [], [], [], []

    for i in range(realizations_per_process):
        Qmax, Tmax, Qmin, Tmin = calculate_qmax_per_realization(densitymap, mask_map_smooth, unmask, delta, smoothing, rng)
        Qmax_sim.append(Qmax)
        Tmax_sim.append(Tmax)
        Qmin_sim.append(Qmin)
        Tmin_sim.append(Tmin)

        # Periodically save progress
        if (i + 1) % 30 == 0:
            print(f"Rank {rank}: Completed {i + 1} realizations")
            pd.concat(Qmax_sim).to_csv(filename_Qmax, mode='a', header=not pd.io.common.file_exists(filename_Qmax), index=False)
            pd.concat(Tmax_sim).to_csv(filename_Tmax, mode='a', header=not pd.io.common.file_exists(filename_Tmax), index=False)
            pd.concat(Qmin_sim).to_csv(filename_Qmin, mode='a', header=not pd.io.common.file_exists(filename_Qmin), index=False)
            pd.concat(Tmin_sim).to_csv(filename_Tmin, mode='a', header=not pd.io.common.file_exists(filename_Tmin), index=False)
            Qmax_sim, Tmax_sim, Qmin_sim, Tmin_sim = [], [], [], []

    # Save remaining results after all realizations
    pd.concat(Qmax_sim).to_csv(filename_Qmax, mode='a', header=not pd.io.common.file_exists(filename_Qmax), index=False)
    pd.concat(Tmax_sim).to_csv(filename_Tmax, mode='a', header=not pd.io.common.file_exists(filename_Tmax), index=False)
    pd.concat(Qmin_sim).to_csv(filename_Qmin, mode='a', header=not pd.io.common.file_exists(filename_Qmin), index=False)
    pd.concat(Tmin_sim).to_csv(filename_Tmin, mode='a', header=not pd.io.common.file_exists(filename_Tmin), index=False)

    print(f"Rank {rank}: Finished all realizations")


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == '__main__':
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Define parameters
    NSIDE = 64
    nsim = 10000
    delta1 = 3/2
    # Three possible smoothing configurations 1. gaussian 2. tophat 3. gaussiantophat
    smoothing = 'gaussian'
    densitymap_file = './densitymap_eclcor_nocmb_nomonopole.txt'
    map_rootname = densitymap_file.replace('./densitymap_', '').replace('_nomonopole.txt', '')
    unmask_file = './unmask.txt'
    log_file = f"sim_data/run_{map_rootname}_{smoothing}_delta_piby{delta1}_nsim_{nsim//1000}K.log"

    output_files = {
        "Qmax": f"sim_data/Qmax_{map_rootname}_{smoothing}_nsim_{nsim//1000}K_delta_piby_{delta1}_rank_{rank}.csv",
        "Tmax": f"sim_data/Tmax_{map_rootname}_{smoothing}_nsim_{nsim//1000}K_delta_piby_{delta1}_rank_{rank}.csv",
        "Qmin": f"sim_data/Qmin_{map_rootname}_{smoothing}_nsim_{nsim//1000}K_delta_piby_{delta1}_rank_{rank}.csv",
        "Tmin": f"sim_data/Tmin_{map_rootname}_{smoothing}_nsim_{nsim//1000}K_delta_piby_{delta1}_rank_{rank}.csv",
    }

    if rank == 0:
        print("Log file:", log_file)
        print("Generating smooth mask...")
        start_time = time.time()

        # Load data and mask
        log_run_details(log_file, densitymap_file, delta1, nsim, smoothing, output_files)
        densitymap = np.loadtxt(densitymap_file)
        lores_mask = np.loadtxt(unmask_file)
        unmask = lores_mask == 1
        mask_map_smooth = generate_smoothmask_realspace(lores_mask, NSIDE, np.pi / delta1, smoothing)

        print(f"Mask generated in {time.time() - start_time:.2f} seconds.")
    else:
        densitymap, mask_map_smooth, unmask = None, None, None

    # Broadcast shared data across processes
    densitymap = comm.bcast(densitymap, root=0)
    mask_map_smooth = comm.bcast(mask_map_smooth, root=0)
    unmask = comm.bcast(unmask, root=0)

    # Begin parallel computation
    start_time = time.time()
    calculate_qmax_mpi(densitymap[unmask], mask_map_smooth, unmask, delta1, nsim, output_files, smoothing)

    comm.Barrier()  # Wait until all processes finish
    if rank == 0:
        print(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")
