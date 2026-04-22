#!/usr/bin/env python3

from pathlib import Path

import discovery as ds
import jax
from loguru import logger

# Change this to whatever you'd like
cache_dir = Path.cwd() / "jax-cache"
cache_dir.mkdir(exist_ok=True)

JAX_CACHE_DIR = str(cache_dir.resolve())

# (For now) Don't change this! Single precision is still being worked on.
# I am just leaving it for when we have figured it out.
USE_64BIT = True

# This can be tweaked to fix GPU memory issues
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.25"

# Set JAX config
jax.config.update("jax_compilation_cache_dir", JAX_CACHE_DIR)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 10)
jax.config.update(
    "jax_persistent_cache_enable_xla_caches",
    "xla_gpu_per_fusion_autotune_cache_dir",
)
jax.config.update("jax_enable_x64", USE_64BIT)

logger.info(f"Using {jax.default_backend()} with {jax.local_device_count()} devices")


def read_pulsar_feathers(data_dir: Path, prefix: str = "") -> list[ds.Pulsar]:
    """Read Pulsar data from feather files in the specified directory and with a given prefix.

    Parameters
    ----------
    data_dir : Path
        Directory containing subdirectory with feather files.
    prefix : str, optional
        Prefix of subdirectory within `data_dir` to search for feather files (default is no prefix).

    Returns
    -------
    list[ds.Pulsar]
        A list of Pulsar objects read from the feather files.

    """
    search_string = f"{prefix}*-[JB]*.feather" if prefix != "" else "[JB]*.feather"
    return [
        ds.Pulsar.read_feather(psrfile)
        for psrfile in sorted(data_dir.glob(search_string))
    ]


def make_curn_maxlike(
    psrs, array=True, broken=False, include_red_noise=True, rn_comp=30, gw_comp=7,
):
    """Create an HD likelihood.

    This function constructs either a GlobalLikelihood or ArrayLikelihood model
    for a set of pulsars.

    Parameters
    ----------
    psrs : list[ds.Pulsar]
        List of Pulsar objects to be included in the model.
    array : bool, optional
        If True, use ArrayLikelihood; if False, use GlobalLikelihood.
        Default is True.
    broken : bool, optional
        If True, use broken power law. Default is False.
    include_red_noise : bool, optional
        If True, include per-pulsar red noise in the model. Default is True.
    rn_comp : int, optional
        Number of red noise frequency components. Default is 30.
    gw_comp : int, optional
        Number of gravitational wave frequency components. Default is 7.

    Returns
    -------
    ds.GlobalLikelihood or ds.ArrayLikelihood
        The chosen HD likelihood.

    """
    Tspan = ds.getspan(psrs)
    if jax.config.jax_enable_x64:
        logger.info("X64 enabled")
        if not array:
            # Build pulsar likelihood components
            if include_red_noise:
                psr_components = (
                    ds.PulsarLikelihood(
                        [
                            psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict),
                            ds.makegp_timing(psr, svd=True),
                            ds.makegp_fourier(
                                psr,
                                ds.powerlaw,
                                rn_comp,
                                T=ds.getspan(psr),
                                name="red_noise",
                            ),
                        ],
                    )
                    for psr in psrs
                )
            else:
                psr_components = (
                    ds.PulsarLikelihood(
                        [
                            psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict),
                            ds.makegp_timing(psr, svd=True),
                        ],
                    )
                    for psr in psrs
                )

            curn = ds.GlobalLikelihood(
                psr_components,
                ds.makeglobalgp_fourier(
                    psrs,
                    ds.brokenpowerlaw if broken else ds.powerlaw,
                    gw_comp,
                    T=Tspan,
                    name="gw",
                ),
            )
        else:
            logger.info("ArrayLikelihood")
            curn = ds.ArrayLikelihood(
                psls=(
                    ds.PulsarLikelihood(
                        [
                            psr.residuals,
                            ds.makenoise_measurement(psr, psr.noisedict),
                            ds.makegp_timing(psr, svd=True),
                        ],
                    )
                    for psr in psrs
                ),
                commongp=ds.makecommongp_fourier(
                    psrs,
                    ds.makepowerlaw_crn(gw_comp) if include_red_noise else ds.powerlaw,
                    rn_comp,
                    T=Tspan,
                    common=["crn_log10_A", "crn_gamma"],
                    name="red_noise" if include_red_noise else "crn",
                ),
            )
    else:
        curn = None

    return curn


def make_spna_rn(psr, rn_comp=30):
    """Create SPNA likelihood with fixed timing model parameters.

    Parameters
    ----------
    psr : ds.Pulsar
        Pulsar object with data and noise dictionary.
    rn_comp : int, optional
        Number of red noise frequency components, by default 30.

    Returns
    -------
    ds.PulsarLikelihood
        Likelihood object for single pulsar noise analysis with variable=False timing.

    """
    return ds.PulsarLikelihood(
        [
            psr.residuals,
            ds.makenoise_measurement(psr, psr.noisedict),
            ds.makegp_timing(psr, svd=True, variable=False),
            ds.makegp_fourier(
                psr,
                ds.powerlaw,
                rn_comp,
                T=ds.getspan(psr),
                name="red_noise",
            ),
        ],
    )


def make_spna_ecorr(psr, rn_comp=30):
    """Create SPNA likelihood with ECORR noise modeling.

    Parameters
    ----------
    psr : ds.Pulsar
        Pulsar object.
    rn_comp : int, optional
        Number of red noise frequency components, by default 30.

    Returns
    -------
    ds.PulsarLikelihood
        Likelihood object for SPNA with ECORR (jitter) noise.

    """
    return ds.PulsarLikelihood(
        [
            psr.residuals,
            ds.makenoise_measurement(psr, ecorr=True),
            ds.makegp_timing(psr, svd=True, variable=True),
            ds.makegp_fourier(
                psr,
                ds.powerlaw,
                rn_comp,
                T=ds.getspan(psr),
                name="red_noise",
            ),
        ],
    )


def make_spna(psr, rn_comp=30):
    """Create standard SPNA likelihood for a single pulsar.

    Parameters
    ----------
    psr : ds.Pulsar
        Pulsar object.
    rn_comp : int, optional
        Number of red noise frequency components, by default 30.

    Returns
    -------
    ds.PulsarLikelihood
        Likelihood object for single pulsar noise analysis.

    """
    return ds.PulsarLikelihood(
        [
            psr.residuals,
            ds.makenoise_measurement(psr),
            ds.makegp_timing(psr, svd=True, variable=True),
            ds.makegp_fourier(
                psr,
                ds.powerlaw,
                rn_comp,
                T=ds.getspan(psr),
                name="red_noise",
            ),
        ],
    )


def make_spna_gbl(psrs, rn_comp=30):
    """Create GlobalLikelihood for multiple pulsars with independent red noise.

    Parameters
    ----------
    psrs : list[ds.Pulsar]
        List of Pulsar objects.
    rn_comp : int, optional
        Number of red noise frequency components per pulsar, by default 30.

    Returns
    -------
    ds.GlobalLikelihood
        Global likelihood combining all pulsars with independent noise.

    """
    pslmodels = [
        ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr),
                ds.makegp_timing(psr, svd=True, variable=True),
                ds.makegp_fourier(
                    psr,
                    ds.powerlaw,
                    rn_comp,
                    T=ds.getspan(psr),
                    name="red_noise",
                ),
            ],
        )
        for psr in psrs
    ]

    return ds.GlobalLikelihood(pslmodels)


def make_spna_array(psrs, rn_comp=30):
    """Create ArrayLikelihood with common red noise across pulsars.

    Parameters
    ----------
    psrs : list[ds.Pulsar]
        List of Pulsar objects.
    rn_comp : int, optional
        Number of red noise frequency components, by default 30.

    Returns
    -------
    ds.ArrayLikelihood
        Array likelihood with common Gaussian process red noise.

    """
    pslmodels = (
        ds.PulsarLikelihood(
            [
                psr.residuals,
                ds.makenoise_measurement(psr),
                ds.makegp_timing(psr, svd=True, variable=False),
            ],
        )
        for psr in psrs
    )
    commongp = (
        ds.makecommongp_fourier(
            psrs,
            ds.powerlaw,
            rn_comp,
            T=[ds.getspan(psr) for psr in psrs],
            name="red_noise",
        ),
    )

    return ds.ArrayLikelihood(pslmodels, commongp=commongp)


def make_curn_theorist_os(
    psrs, array=True, broken_crn=False, include_red_noise=True, rn_comp=30, gw_comp=7,
):
    """Create a CURN likelihood.

    This function constructs either a GlobalLikelihood or ArrayLikelihood model
    for a set of pulsars.

    Parameters
    ----------
    psrs : list[ds.Pulsar]
        List of Pulsar objects to be included in the model.
    array : bool, optional
        If True, use ArrayLikelihood; if False, use GlobalLikelihood.
        Default is True.
    broken_crn : bool, optional
        If True, use broken power law for CRN. Default is False.
    include_red_noise : bool, optional
        If True, include per-pulsar red noise in the model. Default is True.
    rn_comp : int, optional
        Number of red noise frequency components. Default is 30.
    gw_comp : int, optional
        Number of gravitational wave frequency components. Default is 7.

    Returns
    -------
    ds.GlobalLikelihood or ds.ArrayLikelihood
        The constructed CURN likelihood.

    """
    Tspan = ds.getspan(psrs)
    if jax.config.jax_enable_x64:
        logger.info("X64 enabled")

        if not array:
            logger.info("Using GlobalLikelihood")
            # Build pulsar components list
            psr_components_list = []
            if include_red_noise:
                psr_components = [
                    psr.residuals,
                    ds.makenoise_measurement(psr, psr.noisedict, ecorr=False),
                    ds.makegp_timing(psr, svd=True, variable=True),
                    ds.makegp_fourier(
                        psr,
                        ds.powerlaw,
                        rn_comp,
                        T=Tspan,
                        name="red_noise",
                    ),
                    ds.makegp_fourier(
                        psr,
                        ds.powerlaw,
                        gw_comp,
                        T=Tspan,
                        name="gw",
                        common=["gw_log10_A", "gw_gamma"],
                    ),
                ]
            else:
                psr_components = [
                    psr.residuals,
                    ds.makenoise_measurement(psr, psr.noisedict, ecorr=False),
                    ds.makegp_timing(psr, svd=True, variable=True),
                    ds.makegp_fourier(
                        psr,
                        ds.powerlaw,
                        gw_comp,
                        T=Tspan,
                        name="gw",
                        common=["gw_log10_A", "gw_gamma"],
                    ),
                ]

            curn = ds.GlobalLikelihood(
                [ds.PulsarLikelihood(psr_components) for psr in psrs],
            )
        else:
            logger.info("ArrayLikelihood")
            pslmodels = (
                ds.PulsarLikelihood(
                    [
                        psr.residuals,
                        ds.makenoise_measurement(psr, psr.noisedict),
                        ds.makegp_timing(psr, svd=True),
                    ],
                )
                for psr in psrs
            )

            commongp = [
                ds.makecommongp_fourier(
                    psrs,
                    (
                        (
                            ds.powerlaw_brokencrn
                            if broken_crn
                            else ds.makepowerlaw_crn(gw_comp)
                        )
                        if include_red_noise
                        else (ds.brokenpowerlaw if broken_crn else ds.powerlaw)
                    ),
                    rn_comp if include_red_noise else gw_comp,
                    T=Tspan,
                    common=["crn_log10_A", "crn_gamma"]
                    + (["crn_log10_fb"] if broken_crn else []),
                    name="red_noise" if include_red_noise else "crn",
                ),
            ]

            curn = ds.ArrayLikelihood(psls=pslmodels, commongp=commongp)
    else:
        curn = None

    return curn
