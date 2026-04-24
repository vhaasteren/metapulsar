#!/usr/bin/env python

import json
import os

# Force JAX single core, single threaded
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"

import shutil
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import astropy.coordinates as coords
import astropy.units as u

try:
    import cyclopts  # noqa: F401
except ImportError:  # tutorial fallback - see _shims.py
    from _shims import _CycloptsModule as cyclopts
import dill as pickle
import discovery as ds
import numpy as np
import pandas as pd
from astropy.time import Time
from enterprise.pulsar import FeatherPulsar
from loguru import logger
from pta_replicator.red_noise import add_gwb, add_red_noise
from pta_replicator.simulate import load_pulsar, make_ideal, simulate_pulsar
from pta_replicator.white_noise import add_jitter, add_measurement_noise

try:
    from schwimmbad import MPIPool, MultiPool
except ImportError:  # tutorial fallback - see _shims.py
    from _shims import MPIPool, MultiPool

from discovery_utils import read_pulsar_feathers
from frankenstat import frankenize_duplicate_pulsar

warnings.filterwarnings("ignore")
logger.disable("pint")
logger.remove()
logger.add(sys.stderr, colorize=False, enqueue=True)
app = cyclopts.App()


def prep_pint_psr_for_feather(pint_ent_psr):
    """Prepare PINT pulsar object for saving as feather file.

    Handles data type conversions and position array formatting required
    for compatibility with the feather file format.

    Parameters
    ----------
    pint_ent_psr : enterprise.Pulsar
        PINT/Enterprise pulsar object to prepare.

    Returns
    -------
    enterprise.Pulsar
        Modified pulsar object ready for feather export.

    """
    pos_t = pint_ent_psr._pos_t  # noqa: SLF001
    # Handle unchanging position
    pint_ent_psr._pos_t = (  # noqa: SLF001
        np.tile(pos_t, (pint_ent_psr.toas.size, 1)) if pos_t.ndim == 1 else pos_t
    )
    # handle long doubles
    pint_ent_psr._raj = float(pint_ent_psr._raj)  # noqa: SLF001
    pint_ent_psr._decj = float(pint_ent_psr._decj)  # noqa: SLF001
    pint_ent_psr._pos = pint_ent_psr._pos.astype(float)  # noqa: SLF001
    pint_ent_psr._pos_t = pint_ent_psr._pos_t.astype(float)  # noqa: SLF001
    return pint_ent_psr


def make_fibonacci_lattice_psrs(npsr: int) -> tuple[list[str], list[str], list[str]]:
    """Generate pulsar names and positions using Fibonacci lattice on sphere.

    Creates evenly distributed pulsar positions on the celestial sphere using
    the golden spiral (Fibonacci lattice) algorithm.

    Parameters
    ----------
    npsr : int
        Number of pulsars to generate.

    Returns
    -------
    psr_names : list[str]
        Pulsar names in J-name format (e.g., 'J1234+5678').
    raj_strings : list[str]
        Right ascension strings in HMS format.
    decj_strings : list[str]
        Declination strings in DMS format.

    """
    sphere_grid = coords.golden_spiral_grid(npsr)
    ra_dec = coords.SkyCoord(sphere_grid.lon, sphere_grid.lat).to_string(
        style="hmsdms",
        sep=":",
    )
    raj_strings: list[str] = []
    decj_strings: list[str] = []
    psr_names: list[str] = []

    for coord in ra_dec:
        ra, dec = coord.split()
        raj_strings.append(ra)
        decj_strings.append(dec)

        psr_name = f"J{ra.replace(':', '')[:4] + dec.replace(':', '')[:5]}"
        psr_names.append(psr_name)
    return psr_names, raj_strings, decj_strings


def simulate_single_pulsar(tuple_arg):
    """Simulate a single pulsar with realistic timing and noise parameters.

    Generates synthetic pulsar data including TOAs, timing model parameters,
    white noise (EFAC, EQUAD, ECORR), and red noise, then saves as par/tim
    files and feather format.

    Parameters
    ----------
    tuple_arg : tuple
        Packed tuple containing all simulation parameters:
        - psr (str): Pulsar name
        - idx_psr (int): Pulsar index
        - ra (str): Right ascension
        - dec (str): Declination
        - F0 (float): Spin frequency
        - F0_unc (float): Spin frequency uncertainty
        - F1 (float): Spin frequency derivative
        - F1_unc (float): Spin frequency derivative uncertainty
        - PX (float): Parallax
        - PX_unc (float): Parallax uncertainty
        - toaerr (float): TOA measurement uncertainty
        - measurement_seed (int): Random seed for measurements
        - efac_seed (int): Random seed for EFAC
        - equad_seed (int): Random seed for EQUAD
        - ecorr_seed (int): Random seed for ECORR
        - rn_seed (int): Random seed for red noise
        - PEPOCH (float): Reference epoch
        - CLOCK (str): Clock correction
        - UNITS (str): Unit system
        - TIMEEPH (str): Time ephemeris
        - EPHEM (str): Solar system ephemeris
        - obstimes (np.ndarray): Observation times
        - fake_par_dir (Path): Directory for output par files
        - ideal_tim_dir (Path): Directory for output ideal tim files
        - ecorr_include (bool): Whether to include ECORR
        - ecorr_bounds (tuple): ECORR amplitude bounds (log10)
        - wn_const (bool): Whether to use constant white noise
        - efac_bounds (tuple): EFAC bounds
        - equad_bounds (tuple): EQUAD bounds (log10)
        - rn_include (bool): Whether to include red noise
        - rn_comp (int): Number of red noise components
        - rn_amp_bounds (tuple): Red noise amplitude bounds (log10)
        - rn_gamma_bounds (tuple): Red noise spectral index bounds
        - npsr (int): Total number of pulsars
        - decrease_irn_amp (bool): Whether to decrease intrinsic red noise
        - const_seed (int): Constant random seed

    Returns
    -------
    tuple
        (pulsar_name, par_path, tim_path) for the simulated pulsar.

    """
    (
        psr,
        idx_psr,
        ra,
        dec,
        F0,
        F0_unc,
        F1,
        F1_unc,
        PX,
        PX_unc,
        toaerr,
        measurement_seed,
        efac_seed,
        equad_seed,
        ecorr_seed,
        rn_seed,
        PEPOCH,
        CLOCK,
        UNITS,
        TIMEEPH,
        EPHEM,
        obstimes,
        fake_par_dir,
        ideal_tim_dir,
        ecorr_include,
        ecorr_bounds,
        wn_const,
        efac_bounds,
        equad_bounds,
        rn_include,
        rn_comp,
        rn_amp_bounds,
        rn_gamma_bounds,
        npsr,
        decrease_irn_amp,
        const_seed,
    ) = tuple_arg
    par_df = pd.DataFrame.from_dict(
        {
            "name": [
                "PSRJ",
                "RAJ",
                "DECJ",
                "F0",
                "F1",
                "PX",
                "PEPOCH",
                "CLOCK",
                "UNITS",
                "TIMEEPH",
                "EPHEM",
                "PLANET_SHAPIRO",
            ],
            "col1": [
                psr,
                ra,
                dec,
                F0,
                F1,
                PX,
                PEPOCH,
                CLOCK,
                UNITS,
                TIMEEPH,
                EPHEM,
                "Y",
            ],
            "col2": [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            "col3": [
                None,
                None,
                None,
                F0_unc,
                F1_unc,
                PX_unc,
                None,
                None,
                None,
                None,
                None,
                None,
            ],
        },
    )

    # write fake par
    par_df.to_csv(fake_par_dir / f"{psr}.par", sep=" ", header=False, index=False)

    flag_dict_list = [
        {"fe": "fake_1", "be": "AXIS", "f": "fake_1_AXIS"},
        {"fe": "fake_1", "be": "AXIS", "f": "fake_2_AXIS"},
        {"fe": "fake_1", "be": "AXIS", "f": "fake_3_AXIS"},
    ]

    f_list = ["fake_1_AXIS", "fake_2_AXIS", "fake_3_AXIS"]
    fake_psr = simulate_pulsar(
        str(fake_par_dir / f"{psr}.par"),
        obstimes,
        toaerr,
        flags=(flag_dict_list * int(np.ceil(obstimes.size / len(flag_dict_list))))[
            : obstimes.size
        ],
    )

    efac_rng = (
        np.random.default_rng(const_seed + 1)
        if const_seed is not None
        else np.random.default_rng(efac_seed)
    )
    equad_rng = (
        np.random.default_rng(const_seed + 2)
        if const_seed is not None
        else np.random.default_rng(equad_seed)
    )
    ecorr_rng = (
        np.random.default_rng(const_seed + 3)
        if const_seed is not None
        else np.random.default_rng(ecorr_seed)
    )

    rn_rng = (
        np.random.default_rng(const_seed + 4)
        if const_seed is not None
        else np.random.default_rng(rn_seed)
    )

    make_ideal(fake_psr)
    fake_psr.toas.write_TOA_file(str(ideal_tim_dir / f"{psr}.tim"))

    if wn_const:
        fake_psr.sim_efac = np.full(len(f_list), 1.0)
        fake_psr.sim_equad = np.full(len(f_list), -20.0)
    else:
        fake_psr.sim_efac = efac_rng.uniform(*efac_bounds, len(f_list))
        fake_psr.sim_equad = equad_rng.uniform(*equad_bounds, len(f_list))
    add_measurement_noise(
        fake_psr,
        efac=fake_psr.sim_efac,
        log10_equad=fake_psr.sim_equad,
        flagid="f",
        flags=f_list,
        seed=measurement_seed,
    )

    # If you want red noise or ecorr uncomment the following and add appropriate seed variables
    # and parameter values
    if ecorr_include:
        fake_psr.sim_ecorr = ecorr_rng.uniform(*ecorr_bounds, len(f_list))
        add_jitter(
            fake_psr,
            log10_ecorr=fake_psr.sim_ecorr,
            flagid="f",
            flags=f_list,
            coarsegrain=1.0 / 86400.0,
            seed=ecorr_seed,
        )

    if rn_include:
        fake_psr.sim_rn_amp = rn_rng.uniform(*rn_amp_bounds) - decrease_irn_amp
        fake_psr.sim_rn_gamma = rn_rng.uniform(*rn_gamma_bounds)
        add_red_noise(
            fake_psr,
            log10_amplitude=fake_psr.sim_rn_amp,
            spectral_index=fake_psr.sim_rn_gamma,
            components=rn_comp,
            seed=rn_seed,
        )

    logger.debug(f"Done simulating pulsar {fake_psr.name} {idx_psr + 1}/{npsr}")
    return fake_psr


def fit_one_pulsar(tuple_arg):
    """Fit timing model for a single pulsar and save par/tim files.

    Parameters
    ----------
    tuple_arg : tuple
        Packed tuple containing:
        - i (int): Pulsar index for logging
        - psr (pint.Pulsar): Pulsar object to fit
        - fake_par_dir (Path): Directory for output par file
        - fake_tim_dir (Path): Directory for output tim file
        - npsr (int): Total number of pulsars for logging

    """
    i, psr, fake_par_dir, fake_tim_dir, npsr = tuple_arg
    logger.debug(f"Now fitting {psr.name} {i + 1}/{npsr}")
    psr.fit(fitter="gls", maxiter=500)
    psr.model.write_parfile(str(fake_par_dir / f"{psr.name}.par"))
    psr.toas.write_TOA_file(str(fake_tim_dir / f"{psr.name}.tim"))


@app.command
def simulate_pull_from_priors_parallel(
    data_dir: str | os.PathLike,
    npsr: int,
    gw_log10_A: float = -15.0,  # noqa: N803
    gw_gamma: float = 13 / 3,
    ecorr_include: bool = False,
    wn_const: bool = True,
    rn_include: bool = True,
    rn_comp: int = 30,
    decrease_irn_amp: float = 0.0,
    rng_seed: int = 42,
    const_rng_seed: None | int = None,
    use_mpi: bool = True,
) -> None:
    """Simulate a PTA by pulling from usual PTA priors.

    Parameters
    ----------
    data_dir : Path
        Path to store data in
    npsr : int
        Number of pulsars to simulate
    gw_log10_A : float
        GW background amplitude. Defaults to -15.0
    gw_gamma : float
        GW spectral index. Defaults to 13/3
    ecorr_include: bool
       Whether to include ecorr. Defaults to False.
    wn_const: bool
       Whether to pull WN or set it to efac,equad = 1.0, -20. Defaults to True
    rn_include: bool
        Whether to include intrinsic red noise. Defaults to True.
    decrease_irn_amp: float
        Factor to decrease intrinsic red noise amplitude by. Subtracts from log10_A
    rn_comp: int
        Number of frequencies to use when generating red noise, if generating red noise.
        Defaults to 30.
    rng_seed : int
        random number seed. Defaults to 42.

    """
    pool = MPIPool() if use_mpi else MultiPool()

    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True, parents=True)

    logger.debug(f"{data_dir=}")
    logger.debug(f"{npsr=}")
    logger.debug(f"{ecorr_include=}")
    logger.debug(f"{gw_log10_A=}")
    logger.debug(f"{gw_gamma=}")
    logger.debug(f"{rn_include=}")
    logger.debug(f"{rng_seed=}")

    fake_par_dir = data_dir / "fake_pars/"
    fake_par_dir.mkdir(exist_ok=True)

    fake_tim_dir = data_dir / "fake_tims/"
    fake_tim_dir.mkdir(exist_ok=True)

    ideal_tim_dir = data_dir / "ideal_tims/"
    ideal_tim_dir.mkdir(exist_ok=True)

    fake_feather_dir = data_dir / "fake_feathers/"
    fake_feather_dir.mkdir(exist_ok=True)

    logger.debug("All directories created")

    psr_names, raj_strings, decj_strings = make_fibonacci_lattice_psrs(npsr)

    logger.debug("Fibonacci lattice created")

    # noise bounds
    efac_bounds = (0.1, 10)
    equad_bounds = (-9, -5)
    ecorr_bounds = (-9, -5)
    rn_amp_bounds = (-20, -12)
    rn_gamma_bounds = (1, 6)

    F0_bounds = (2.47, 3)  # log10(Hz)
    F0_unc_bounds_fraction = (-16, -13)  # dimensionless

    F1_bounds = (-17, -15)  # Hz^2
    F1_unc_bounds_fraction = (-5, -3)  # dimensionless

    PX_bounds = (0.5, 2)  # mas
    PX_unc_bounds_fraction = (0.1, 0.8)  # dimensionless, multiply by PX value

    PEPOCH = 56166.0000000000000000
    CLOCK = "TT(BIPM2019)"
    UNITS = "TDB"
    TIMEEPH = "FB90"
    EPHEM = "DE440"

    measurement_seeds = (rng_seed + 100) * 1000 + np.arange(npsr)
    efac_seeds = (rng_seed + 100) * 2000 + np.arange(npsr)
    equad_seeds = (rng_seed + 100) * 3000 + np.arange(npsr)
    ecorr_seeds = (rng_seed + 100) * 4000 + np.arange(npsr)
    rn_seeds = (rng_seed + 100) * 5000 + np.arange(npsr)
    gwb_seed = (rng_seed + 100) * 6000
    const_seeds = (
        [None] * npsr if const_rng_seed is None else const_rng_seed + np.arange(npsr)
    )

    rng = (
        np.random.default_rng(const_rng_seed)
        if const_rng_seed is not None
        else np.random.default_rng(rng_seed + 100)
    )

    t_init = Time("2010-01-01").mjd
    t_span = (20 * u.yr).to(u.day).value
    n_toa = np.ceil(t_span / 365 * 36).astype(int)
    obstimes = np.linspace(t_init, t_init + t_span, n_toa)

    toaerrs = rng.normal(
        (100 * u.ns).to(u.us).value,
        (10 * u.ns).to(u.us).value,
        size=(npsr, obstimes.size),
    )  # us

    F0s_log10 = rng.uniform(*F0_bounds, size=npsr)
    F0_uncs = 10 ** (rng.uniform(*F0_unc_bounds_fraction, size=npsr) + F0s_log10)
    F0s = 10**F0s_log10

    F1s_log10 = rng.uniform(*F1_bounds, size=npsr)
    F1_uncs = 10 ** (rng.uniform(*F1_unc_bounds_fraction, size=npsr) + F1s_log10)
    F1s = 10**F1s_log10

    PXs = rng.uniform(*PX_bounds, size=npsr)
    PX_uncs = rng.uniform(*PX_unc_bounds_fraction, size=npsr) * PXs

    fake_psrs = list(
        pool.map(
            simulate_single_pulsar,
            tuple(
                zip(
                    psr_names,
                    range(len(psr_names)),
                    raj_strings,
                    decj_strings,
                    F0s,
                    F0_uncs,
                    F1s,
                    F1_uncs,
                    PXs,
                    PX_uncs,
                    toaerrs,
                    measurement_seeds,
                    efac_seeds,
                    equad_seeds,
                    ecorr_seeds,
                    rn_seeds,
                    [PEPOCH] * len(psr_names),
                    [CLOCK] * len(psr_names),
                    [UNITS] * len(psr_names),
                    [TIMEEPH] * len(psr_names),
                    [EPHEM] * len(psr_names),
                    [obstimes] * len(psr_names),
                    [fake_par_dir] * len(psr_names),
                    [ideal_tim_dir] * len(psr_names),
                    [ecorr_include] * len(psr_names),
                    [ecorr_bounds] * len(psr_names),
                    [wn_const] * len(psr_names),
                    [efac_bounds] * len(psr_names),
                    [equad_bounds] * len(psr_names),
                    [rn_include] * len(psr_names),
                    [rn_comp] * len(psr_names),
                    [rn_amp_bounds] * len(psr_names),
                    [rn_gamma_bounds] * len(psr_names),
                    [npsr] * len(psr_names),
                    [decrease_irn_amp] * len(psr_names),
                    const_seeds,
                    strict=True,
                ),
            ),
        ),
    )

    logger.debug("Add HD correlated GWB to all pulsars")
    add_gwb(fake_psrs, gw_log10_A, gw_gamma, seed=gwb_seed)
    logger.debug("Done adding HD correlated GWB to all pulsars")

    logger.debug("All combined pulsars simulated. Starting to fit.")

    # with pool as p:
    pool.map(
        fit_one_pulsar,
        tuple(
            zip(
                range(len(fake_psrs)),
                fake_psrs,
                [fake_par_dir] * len(fake_psrs),
                [fake_tim_dir] * len(fake_psrs),
                [npsr] * len(fake_psrs),
                strict=False,
            ),
        ),
    )
    pool.close()

    pkl_path = fake_feather_dir / "fake_psrs.pkl"
    with (pkl_path).open("wb") as f:
        pickle.dump(fake_psrs, f)
    logger.debug("All combined pulsars fit + pars and tims saved.")

    injected_params = {}
    for psr in fake_psrs:
        psr_name = psr.name
        frontends = ["fake_1", "fake_2", "fake_3"]
        backend = "AXIS"
        noisedict = {}
        if rn_include:
            noisedict = noisedict | {
                f"{psr.name}_red_noise_log10_A": psr.sim_rn_amp,
                f"{psr.name}_red_noise_gamma": psr.sim_rn_gamma,
            }
        for i, fe in enumerate(frontends):
            name_backend = f"{psr_name}_{fe}_{backend}"
            if ecorr_include:
                noisedict = noisedict | {
                    f"{name_backend}_log10_ecorr": psr.sim_ecorr[i],
                }
            noisedict = noisedict | {
                f"{name_backend}_efac": psr.sim_efac[i],
                f"{name_backend}_log10_t2equad": psr.sim_equad[i],
            }

        pint_ent_psr = prep_pint_psr_for_feather(psr.to_enterprise())

        save_loc = str(fake_feather_dir / f"{psr.name}.feather")
        FeatherPulsar.save_feather(
            pint_ent_psr,
            save_loc,
            noisedict=noisedict,
        )
        injected_params.update(noisedict)

    injected_params.update({"gw_log10_A": gw_log10_A, "gw_gamma": gw_gamma})

    save_loc = data_dir / "injected-parameters.json"
    with (save_loc).open("w") as f:
        json.dump(
            {
                p: val.tolist() if hasattr(val, "__array__") else val
                for p, val in injected_params.items()
            },
            f,
        )
    logger.debug("All combined pulsars saved as feathers.")
    logger.debug(f"Injected parameters saved at {save_loc}.")


def _split_pulsar_into_ptas_helper(tuple_arg):
    """Helper function to split a single pulsar into 3 PTAs and fit."""
    (
        par_file,
        tim_file,
        par_dirs,
        tim_dirs,
    ) = tuple_arg

    with tim_file.open("r") as t:
        tim_lines = t.readlines()

    result_psrs = []
    for par_dir, tim_dir, start_idx in zip(par_dirs, tim_dirs, range(3), strict=False):
        new_par_file = shutil.copy(par_file, par_dir / par_file.name)

        new_tim_file = tim_dir / tim_file.name
        with new_tim_file.open("w") as f:
            f.writelines(tim_lines[0])
            i = 1
            while tim_lines[i].lower().startswith("c"):
                f.writelines(tim_lines[i])
                i += 1
            f.writelines(tim_lines[i + start_idx :: 3])

        fake_psr = load_pulsar(
            parfile=str(new_par_file),
            timfile=str(new_tim_file),
        )
        psr_name = fake_psr.model.name

        logger.debug(f"Fitting tm for {psr_name}")
        fake_psr.fit(fitter="gls", maxiter=500)

        fake_psr.model.write_parfile(str(new_par_file))
        fake_psr.toas.write_TOA_file(str(new_tim_file))
        result_psrs.append(fake_psr)

    return result_psrs


def _process_split_pulsar_feather_helper(tuple_arg):
    """Helper function to process a split pulsar and create feather file."""
    (
        psr,
        feather_dir,
        pta_num,
        data_dir,
        do_spna_max_likelihood,
    ) = tuple_arg

    psr_name = psr.name
    if do_spna_max_likelihood:
        orig_psr = ds.Pulsar.read_feather(
            str(data_dir / "fake_feathers_spna" / f"{psr.name}.feather"),
        )
    else:
        orig_psr = ds.Pulsar.read_feather(
            str(data_dir / "fake_feathers" / f"{psr.name}.feather"),
        )

    frontend = f"fake_{pta_num}"
    psr_noisedict = {
        key: val
        for key, val in orig_psr.noisedict.items()
        if (frontend in key or "red_noise" in key)
    }

    ent_psr = prep_pint_psr_for_feather(psr.to_enterprise())

    feather_name = str(feather_dir / f"{psr.name}.feather")

    FeatherPulsar.save_feather(ent_psr, feather_name, noisedict=psr_noisedict)

    return psr_name


@app.command
def split_simulated_pta_parallel(
    data_dir: str | os.PathLike,
    do_spna_max_likelihood: bool = False,
    use_mpi: bool = False,
):
    """Slice a large PTA into 3 equal, smaller PTAs and make franken pulsars.

    Parameters
    ----------
    data_dir : Path
        Path to store data in

    """
    pool = MPIPool() if use_mpi else MultiPool()
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)

    if not data_dir.exists():
        err_msg = (
            "The data directory doesn't exist! Please simulate combined PTA first."
        )
        raise FileExistsError(err_msg)

    if do_spna_max_likelihood:
        source_pars = data_dir / "fake_pars_spna"
    else:
        source_pars = data_dir / "fake_pars"

    source_tims = data_dir / "fake_tims"

    fake_par_dir_1 = data_dir / "fake_pars_pta_1/"
    fake_par_dir_1.mkdir(exist_ok=True)
    fake_par_dir_2 = data_dir / "fake_pars_pta_2/"
    fake_par_dir_2.mkdir(exist_ok=True)
    fake_par_dir_3 = data_dir / "fake_pars_pta_3/"
    fake_par_dir_3.mkdir(exist_ok=True)

    fake_tim_dir_1 = data_dir / "fake_tims_pta_1/"
    fake_tim_dir_1.mkdir(exist_ok=True)
    fake_tim_dir_2 = data_dir / "fake_tims_pta_2/"
    fake_tim_dir_2.mkdir(exist_ok=True)
    fake_tim_dir_3 = data_dir / "fake_tims_pta_3/"
    fake_tim_dir_3.mkdir(exist_ok=True)

    fake_feather_dir_1 = data_dir / "fake_feathers_pta_1/"
    fake_feather_dir_1.mkdir(exist_ok=True)
    fake_feather_dir_2 = data_dir / "fake_feathers_pta_2/"
    fake_feather_dir_2.mkdir(exist_ok=True)
    fake_feather_dir_3 = data_dir / "fake_feathers_pta_3/"
    fake_feather_dir_3.mkdir(exist_ok=True)

    par_files = sorted(source_pars.glob("*.par"))
    tim_files = sorted(source_tims.glob("*.tim"))

    # Parallelize splitting of pulsars into 3 PTAs
    par_dirs = (fake_par_dir_1, fake_par_dir_2, fake_par_dir_3)
    tim_dirs = (fake_tim_dir_1, fake_tim_dir_2, fake_tim_dir_3)

    list_of_psr_triplets = list(
        pool.map(
            _split_pulsar_into_ptas_helper,
            tuple(
                zip(
                    par_files,
                    tim_files,
                    [par_dirs] * len(par_files),
                    [tim_dirs] * len(tim_files),
                    strict=False,
                ),
            ),
        ),
    )

    # Unpack results into separate PTA lists
    fake_psrs_pta_1 = [triplet[0] for triplet in list_of_psr_triplets]
    fake_psrs_pta_2 = [triplet[1] for triplet in list_of_psr_triplets]
    fake_psrs_pta_3 = [triplet[2] for triplet in list_of_psr_triplets]

    for feather_dir, fake_psrs in zip(
        [fake_feather_dir_1, fake_feather_dir_2, fake_feather_dir_3],
        [fake_psrs_pta_1, fake_psrs_pta_2, fake_psrs_pta_3],
        strict=True,
    ):
        pkl_path = feather_dir / "fake_psrs.pkl"
        with (pkl_path).open("wb") as f:
            pickle.dump(fake_psrs, f)
    logger.debug("All pulsars split into three, refit, and saved as par and tim.")

    # Parallelize processing of split pulsars to create feather files
    all_psrs_with_params = []
    for fake_psrs, feather_dir, pta_num in zip(
        (fake_psrs_pta_1, fake_psrs_pta_2, fake_psrs_pta_3),
        (fake_feather_dir_1, fake_feather_dir_2, fake_feather_dir_3),
        range(1, 4),
        strict=False,
    ):
        for psr in fake_psrs:
            all_psrs_with_params.append(
                (
                    psr,
                    feather_dir,
                    pta_num,
                    data_dir,
                    do_spna_max_likelihood,
                ),
            )

    pool.map(
        _process_split_pulsar_feather_helper,
        tuple(all_psrs_with_params),
    )
    pool.close()

    logger.debug("All split pulsars saved as feathers.")


@app.command
def frankenize_split_pta(data_dir: Path):
    """Create FrankenPulsars from split PTAs.

    Combines observations of the same pulsar from three split PTAs into
    composite FrankenPulsars. Identifies duplicate pulsars across the three
    sub-PTAs, merges their data, and saves the results.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing fake_feathers_pta_1, fake_feathers_pta_2,
        and fake_feathers_pta_3 subdirectories.

    Notes
    -----
    The function performs the following operations:
    - Reads all pulsars from the three split PTAs
    - Identifies duplicates (same pulsar observed by multiple PTAs)
    - Creates FrankenPulsars by combining duplicate observations
    - Saves non-duplicate pulsars with cleaned names
    - Outputs franken_psr_list.txt and psr_backends_by_pta.json metadata

    """
    fake_feather_dir_1 = data_dir / "fake_feathers_pta_1/"
    fake_feather_dir_2 = data_dir / "fake_feathers_pta_2/"
    fake_feather_dir_3 = data_dir / "fake_feathers_pta_3/"

    logger.debug("Starting to frankenize pulsars.")

    psrs = read_pulsar_feathers(fake_feather_dir_1)
    psrs.extend(read_pulsar_feathers(fake_feather_dir_2))
    psrs.extend(read_pulsar_feathers(fake_feather_dir_3))

    output_dir = data_dir / "franken_psrs"
    output_dir.mkdir(exist_ok=True)

    # ## Find out which pulsars need to become FrankenPulsars (which ones have duplicates)

    psr_indices_dict = {
        psr.name.rsplit("_")[0]: [
            i for i, j in enumerate(psrs) if psr.name.rsplit("_")[0] in j.name
        ]
        for psr in psrs
    }
    psr_duplicate_indices_dict = {
        name: item for name, item in psr_indices_dict.items() if len(item) > 1
    }

    # Save which pulsars are franken pulsars
    with (output_dir / "franken_psr_list.txt").open("w") as f:
        np.savetxt(
            f,
            np.array(list(psr_duplicate_indices_dict.keys())),
            delimiter=" ",
            fmt="%s",
        )

    # Get all backends
    backends = {"AXIS": set(), "ng": set(), "epta": set(), "ppta": set()}
    for psr in psrs:
        if "AXIS" in psr.flags["be"]:
            backends["AXIS"] = backends["AXIS"] | set(psr.backend_flags)
        # only the above really matters because this is simulated data
        elif "ng" in psr.name:
            backends["ng"] = backends["ng"] | set(psr.backend_flags)
        elif "epta" in psr.name:
            backends["epta"] = backends["epta"] | set(psr.backend_flags)
        elif "ppta" in psr.name:
            backends["ppta"] = backends["ppta"] | set(psr.backend_flags)

    # need list for json
    backends = {key: list(val) for key, val in backends.items()}
    # write the dict as a json file
    with (output_dir / "psr_backends_by_pta.json").open("w") as f:
        json.dump(backends, f)

    # Iterate over pulsars that have duplicates
    prefix = "franken"
    for psr, indices in psr_duplicate_indices_dict.items():
        # Get list of pulsars to combine, sorted alphabetically
        name_attr = "name"
        pulsars = sorted(
            [psrs[i] for i in indices],
            key=lambda psr: getattr(psr, name_attr),
        )
        frankenize_duplicate_pulsar(pulsars, outdir=output_dir, prefix=prefix)

    # Now, read in other 3p pulsars and fix their names
    for psr in psrs:
        if psr.name.split("_")[0] not in psr_duplicate_indices_dict:
            noisedict = {
                key.replace("_epta", "").replace("_ppta", "").replace("_ng", ""): val
                for key, val in psr.noisedict.items()
            }
            psr_copy = deepcopy(psr)
            psr_copy.name = psr.name.split("_")[0]
            psr_copy.noisedict = noisedict
            psr_copy.save_feather(output_dir / f"{prefix}-{psr_copy.name}.feather")


if __name__ == "__main__":
    app()
