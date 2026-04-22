#!/usr/bin/env python
"""SPNA and HD noise analysis functions for pulsar timing arrays."""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

# Force JAX single core, single threaded
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREAD"] = "1"
os.environ["NPROC"] = "1"

try:
    import cyclopts  # noqa: F401
except ImportError:  # tutorial fallback - see _shims.py
    from _shims import _CycloptsModule as cyclopts
import discovery as ds
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from loguru import logger

import discovery_utils as du
from optimization import run_svi_early_stopping, setup_svi

warnings.filterwarnings("ignore")
logger.info(f"Using {jax.default_backend()} with {jax.local_device_count()} devices")
logger.disable("pint")
logger.remove()
logger.add(sys.stderr, colorize=False, enqueue=True)
app = cyclopts.App()


def run_spna(
    psr: ds.Pulsar,
    rng_key: int | jax.Array = 117,
    batch_size: int = 1000,
    svi_samples: int = 20_000,
    repar_params: list[str] = list(),
):
    """Run SVI-based single pulsar noise analysis.

    Estimates white noise (EFAC, EQUAD) and red noise (amplitude, spectral index)
    parameters for a single pulsar using stochastic variational inference.

    Parameters
    ----------
    psr : ds.Pulsar
        Pulsar object to analyze.
    rng_key : int or jax.Array, optional
        Random number generator key, by default 117.
    batch_size : int, optional
        Number of SVI steps per batch, by default 1000.
    svi_samples : int, optional
        Maximum total number of SVI iterations, by default 20000.
    repar_params : list[str], optional
        List of parameters to reparameterize (currently unused), by default empty list.

    Returns
    -------
    dict
        Dictionary of maximum a posteriori (MAP) parameter estimates.

    """
    logger.debug("read in psrs")
    rng_key = jax.random.key(rng_key) if isinstance(rng_key, int) else rng_key
    map_params = {}
    logger.debug(f"working on {psr=}")
    spna = du.make_spna(psr)

    params = sorted(spna.logL.params)
    logger.debug(f"{params=}")

    red_noise_gamma_params = []
    red_noise_amp_params = []
    efac_params = []
    equad_params = []
    ecorr_params = []

    for p in params:
        if "red_noise_log10_A" in p:
            red_noise_amp_params.append(p)
        elif "red_noise_gamma" in p:
            red_noise_gamma_params.append(p)
        elif "efac" in p:
            efac_params.append(p)
        elif "equad" in p:
            equad_params.append(p)
        elif "ecorr" in p:
            ecorr_params.append(p)
        else:
            logger.error(f"{p} is not a recognized parameter.")

    def model_spna():
        params = {}
        for amp, gam in zip(red_noise_amp_params, red_noise_gamma_params, strict=False):
            params.update(
                {
                    amp: numpyro.sample(amp, dist.Uniform(-20, -11)),
                    gam: numpyro.sample(gam, dist.Uniform(0, 7)),
                },
            )

        # for efac, equad, ecorr in zip(efac_params, equad_params, ecorr_params):
        for efac, equad in zip(efac_params, equad_params, strict=False):
            params.update(
                {
                    efac: numpyro.sample(efac, dist.Uniform(0.1, 10)),
                    equad: numpyro.sample(equad, dist.Uniform(-20, -5)),
                    # ecorr: numpyro.sample(ecorr, dist.Uniform(-8.5, -5)),
                },
            )

        numpyro.factor("ll", spna.logL(params))

    map_params = {}
    start = datetime.now()
    if len(repar_params) > 0:
        if not set(repar_params) < set(params):
            err = f"{repar_params=} but is not in model {params=}."
            raise KeyError(err)

    autoguide_map = numpyro.infer.autoguide.AutoDelta(model_spna)
    svi = setup_svi(
        model=model_spna,
        guide=autoguide_map,
        max_epochs=svi_samples,
        num_warmup_steps=batch_size,
    )

    svi_key, rng_key = jax.random.split(rng_key)

    svi_results = run_svi_early_stopping(
        svi_key,
        svi,
        batch_size=batch_size,
        max_num_batches=int(jnp.ceil(svi_samples / batch_size)),
    )
    finish = datetime.now()
    logger.debug(f"Ran {svi_samples} iterations in {(finish - start).seconds}s")
    map_params = map_params | {
        "_".join(key.split("_")[:-2]).removesuffix("_base"): value
        for key, value in svi_results.items()
    }

    logger.debug(map_params)
    return map_params


def run_spna_mcmc(
    psr: ds.Pulsar,
    rng_key: int | jax.Array = 117,
    num_warmup: int = 1000,
    num_samples: int = 2000,
    num_chains: int = 4,
):
    """Run MCMC NUTS sampling for single pulsar noise analysis.

    Parameters
    ----------
    psr : ds.Pulsar
        Pulsar object to analyze
    rng_key : int | jax.Array, optional
        Random number generator key, by default 117
    num_warmup : int, optional
        Number of warmup/burnin iterations, by default 1000
    num_samples : int, optional
        Number of posterior samples per chain, by default 2000
    num_chains : int, optional
        Number of MCMC chains to run, by default 4

    Returns
    -------
    dict
        Dictionary containing MCMC samples for each parameter

    """
    logger.debug("read in psrs")
    rng_key = jax.random.key(rng_key) if isinstance(rng_key, int) else rng_key
    logger.debug(f"working on {psr=}")
    spna = du.make_spna(psr)

    params = sorted(spna.logL.params)
    logger.debug(f"{params=}")

    red_noise_gamma_params = []
    red_noise_amp_params = []
    efac_params = []
    equad_params = []

    for p in params:
        if "red_noise_log10_A" in p:
            red_noise_amp_params.append(p)
        elif "red_noise_gamma" in p:
            red_noise_gamma_params.append(p)
        elif "efac" in p:
            efac_params.append(p)
        elif "equad" in p:
            equad_params.append(p)
        else:
            logger.error(f"{p} is not a recognized parameter.")

    def model_spna():
        params = {}
        for amp, gam in zip(red_noise_amp_params, red_noise_gamma_params, strict=False):
            params.update(
                {
                    amp: numpyro.sample(amp, dist.Uniform(-20, -11)),
                    gam: numpyro.sample(gam, dist.Uniform(0, 7)),
                },
            )

        for efac, equad in zip(efac_params, equad_params, strict=False):
            params.update(
                {
                    efac: numpyro.sample(efac, dist.Uniform(0.1, 10)),
                    equad: numpyro.sample(equad, dist.Uniform(-20, -5)),
                },
            )

        numpyro.factor("ll", spna.logL(params))

    # Setup NUTS kernel
    nuts_kernel = numpyro.infer.NUTS(model_spna)

    # Setup MCMC
    mcmc = numpyro.infer.MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )

    start = datetime.now()
    logger.debug(
        f"Starting MCMC with {num_chains} chains, {num_warmup} warmup, {num_samples} samples",
    )

    # Run MCMC
    mcmc.run(rng_key)

    finish = datetime.now()
    logger.debug(f"Ran MCMC in {(finish - start).seconds}s")

    # Get samples
    samples = mcmc.get_samples()

    # Print diagnostics
    mcmc.print_summary()

    logger.debug(f"Posterior samples collected for {list(samples.keys())}")
    return samples


def run_array_no_rn(
    psrs: list[ds.Pulsar],
    rng_key: int = 117,
    output_dir: str | Path = "./outputs",
    batch_size: int = 1_000,
    svi_samples: int = 20_000,
    gw_comp: int = 7,
):
    """Run HD array analysis without per-pulsar red noise.

    Estimates common red noise (CRN) parameters using SVI for a pulsar timing
    array without modeling individual pulsar red noise.

    Parameters
    ----------
    psrs : list[ds.Pulsar]
        List of Pulsar objects in the array.
    rng_key : int, optional
        Random number generator seed, by default 117.
    output_dir : str or Path, optional
        Output directory for results, by default "./outputs".
    batch_size : int, optional
        Number of SVI steps per batch, by default 1000.
    svi_samples : int, optional
        Maximum total number of SVI iterations, by default 20000.
    gw_comp : int, optional
        Number of gravitational wave frequency components, by default 7.

    Returns
    -------
    dict
        Dictionary of MAP parameter estimates for CRN.

    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.debug("read in psrs")
    rng_key = jax.random.key(rng_key)
    array = du.make_curn_maxlike(psrs, include_red_noise=False, gw_comp=gw_comp)

    logL = array.logL

    def model_array():
        params = {
            "crn_gamma": numpyro.sample("crn_gamma", dist.Uniform(0, 7)),
            "crn_log10_A": numpyro.sample("crn_log10_A", dist.Uniform(-18, -11)),
        }

        numpyro.factor("ll", logL(params))

    # Define a learning rate schedule
    map_params = {}
    start = datetime.now()
    autoguide_map = numpyro.infer.autoguide.AutoDelta(model_array)
    svi = setup_svi(
        model=model_array,
        guide=autoguide_map,
        max_epochs=svi_samples,
        num_warmup_steps=int(0.1 * svi_samples),
    )
    svi_key, rng_key = jax.random.split(rng_key)

    svi_results = run_svi_early_stopping(
        svi_key,
        svi,
        batch_size=batch_size,
        max_num_batches=int(jnp.ceil(svi_samples / batch_size)),
    )
    finish = datetime.now()
    logger.debug(f"Ran {svi_samples} iterations in {(finish - start).seconds}s")
    map_params = map_params | {
        "_".join(key.split("_")[:-2]): value for key, value in svi_results.items()
    }

    logger.debug(map_params)
    return map_params


def run_array(
    psrs: list[ds.Pulsar],
    rng_key: int = 117,
    output_dir: str | Path = "./outputs",
    batch_size: int = 1_000,
    svi_samples: int = 20_000,
    rn_comp: int = 30,
    gw_comp: int = 7,
):
    """Run HD array analysis with per-pulsar red noise.

    Estimates both common red noise (CRN) and per-pulsar red noise parameters
    using SVI for a pulsar timing array.

    Parameters
    ----------
    psrs : list[ds.Pulsar]
        List of Pulsar objects in the array.
    rng_key : int, optional
        Random number generator seed, by default 117.
    output_dir : str or Path, optional
        Output directory for results, by default "./outputs".
    batch_size : int, optional
        Number of SVI steps per batch, by default 1000.
    svi_samples : int, optional
        Maximum total number of SVI iterations, by default 20000.
    rn_comp : int, optional
        Number of red noise frequency components per pulsar, by default 30.
    gw_comp : int, optional
        Number of gravitational wave frequency components, by default 7.

    Returns
    -------
    dict
        Dictionary of MAP parameter estimates for CRN and all pulsar red noise.

    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    logger.debug("read in psrs")
    rng_key = jax.random.key(rng_key)
    array = du.make_curn_maxlike(psrs, rn_comp=rn_comp, gw_comp=gw_comp)

    logL_params = sorted(array.logL.params)

    logL = array.logL

    red_noise_gamma_params = []
    red_noise_amp_params = []

    for p in logL_params:
        if "red_noise_log10_A" in p:
            red_noise_amp_params.append(p)
        elif "red_noise_gamma" in p:
            red_noise_gamma_params.append(p)

    def model_array():
        params = {
            "crn_gamma": numpyro.sample("crn_gamma", dist.Uniform(0, 7)),
            "crn_log10_A": numpyro.sample("crn_log10_A", dist.Uniform(-18, -11)),
        }
        for amp, gam in zip(red_noise_amp_params, red_noise_gamma_params, strict=False):
            params.update(
                {
                    amp: numpyro.sample(amp, dist.Uniform(-20, -11)),
                    gam: numpyro.sample(gam, dist.Uniform(0, 7)),
                },
            )

        numpyro.factor("ll", logL(params))

    # Define a learning rate schedule
    map_params = {}
    start = datetime.now()
    autoguide_map = numpyro.infer.autoguide.AutoDelta(model_array)
    svi = setup_svi(
        model=model_array,
        guide=autoguide_map,
        max_epochs=svi_samples,
        num_warmup_steps=int(0.1 * svi_samples),
    )
    svi_key, rng_key = jax.random.split(rng_key)

    svi_results = run_svi_early_stopping(
        svi_key,
        svi,
        batch_size=batch_size,
        max_num_batches=int(jnp.ceil(svi_samples / batch_size)),
    )
    finish = datetime.now()
    logger.debug(f"Ran {svi_samples} iterations in {(finish - start).seconds}s")
    map_params = map_params | {
        "_".join(key.split("_")[:-2]): value for key, value in svi_results.items()
    }

    logger.debug(map_params)
    return map_params


@app.command
def run_array_max_like(
    data_dir,
    svi_samples=10_000,
    batch_size=500,
    psrs_prefix="",
    rn_include: bool = True,
    rn_comp: int = 30,
    gw_comp: int = 7,
):
    """CLI command to run HD maximum likelihood analysis on a PTA.

    Loads pulsar data and performs maximum likelihood estimation of common
    red noise (CRN) and optionally per-pulsar red noise parameters. Saves
    results to a JSON file.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing pulsar feather files.
    svi_samples : int, optional
        Maximum number of SVI iterations, by default 10000.
    batch_size : int, optional
        Number of SVI steps per batch, by default 500.
    psrs_prefix : str, optional
        Prefix to filter pulsar files, by default "".
    rn_include : bool, optional
        Whether to include per-pulsar red noise, by default True.
    rn_comp : int, optional
        Number of red noise frequency components, by default 30.
    gw_comp : int, optional
        Number of gravitational wave frequency components, by default 7.

    Returns
    -------
    dict
        Dictionary of MAP parameter estimates.

    """
    if not isinstance(data_dir, Path):
        data_dir = Path(data_dir)
        data_dir.mkdir(exist_ok=True, parents=True)
    discovery_psrs = du.read_pulsar_feathers(data_dir, prefix=psrs_prefix)
    if rn_include:
        map_params = run_array(
            discovery_psrs,
            svi_samples=svi_samples,
            batch_size=batch_size,
            rn_comp=rn_comp,
            gw_comp=gw_comp,
        )
    else:
        map_params = run_array_no_rn(
            discovery_psrs,
            svi_samples=svi_samples,
            batch_size=batch_size,
            gw_comp=gw_comp,
        )

    save_loc = data_dir / "max-likelihood-estimate.json"
    with (save_loc).open("w") as f:
        json.dump(
            {
                p: val.tolist() if hasattr(val, "__array__") else val
                for p, val in map_params.items()
            },
            f,
        )
    logger.debug(f"Maximum likelihood parameters saved at {save_loc}.")
    return map_params


if __name__ == "__main__":
    app()
