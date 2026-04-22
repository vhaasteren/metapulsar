#!/usr/bin/env python3

import numpy as np
import pint.models as pm
from loguru import logger as log
from pint.models.parameter import maskParameter


def convert_to_RNAMP(value):
    """Convert enterprise RN amplitude to tempo2/PINT parfile RN amplitude."""
    return (86400.0 * 365.24 * 1e6) / (2.0 * np.pi * np.sqrt(3.0)) * 10**value


def add_noise_to_model(
    model,
    noise_dict,
):
    """Add WN, RN, DMGP, ChromGP, and SW parameters to timing model.

    Parameters
    ----------
    model: PINT (or tempo2) timing model
    use_noise_point: point to use for noise analysis; Default: 'mean_large_likelihood'.
        Options: 'MAP', 'median', 'mean_large_likelihood'
        Note that the MAP is the the same as the maximum likelihood value when all the priors are uniform.
        Mean large likelihood takes N of the largest likelihood values and then takes the mean of those. (Recommended).
    burn_frac: fraction of chain to use for burn-in; Default: 0.25
    save_corner: Flag to toggle saving of corner plots; Default: True
    ignore_red_noise: Flag to manually force RN exclusion from timing model. When False,
        code determines whether
    RN is necessary based on whether the RN BF > 1e3. Default: False
    using_wideband: Flag to toggle between narrowband and wideband datasets; Default: False
    base_dir: directory containing {psr}_nb and {psr}_wb chains directories; if None, will
        check for results in the current working directory './'.
    return_noise_core: Flag to return the la_forge.core object; Default: False

    Returns
    -------
    model: New timing model which includes WN and RN (and potentially dmgp, chrom_gp, and solar wind) parameters
    (optional)
    noise_core: la_forge.core object which contains noise chains and run metadata

    """
    # Assume results are in current working directory if not specified
    # Create the maskParameter for EFACS
    efac_params = []
    equad_params = []
    ecorr_params = []

    efac_idx = 1
    equad_idx = 1
    ecorr_idx = 1

    psr_name = list(noise_dict.keys())[0].split("_")[0]
    noise_pars = np.array(list(noise_dict.keys()))
    wn_dict = {
        key: val
        for key, val in noise_dict.items()
        if "efac" in key or "equad" in key or "ecorr" in key
    }
    for key, val in wn_dict.items():
        if "_efac" in key:
            param_name = key.split("_efac")[0].split(psr_name)[1][1:]

            tp = maskParameter(
                name="EFAC",
                index=efac_idx,
                key="-f",
                key_value=param_name,
                value=val,
                units="",
                convert_tcb2tdb=False,
            )
            efac_params.append(tp)
            efac_idx += 1

        # See https://github.com/nanograv/enterprise/releases/tag/v3.3.0
        # ..._t2equad uses PINT/Tempo2/Tempo convention, resulting in total variance EFAC^2 x (toaerr^2 + EQUAD^2)
        elif "_t2equad" in key:
            param_name = (
                key.split("_t2equad")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="EQUAD",
                index=equad_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            equad_params.append(tp)
            equad_idx += 1

        # ..._equad uses temponest convention; generated with enterprise pre-v3.3.0
        elif "_equad" in key:
            param_name = (
                key.split("_equad")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="EQUAD",
                index=equad_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            equad_params.append(tp)
            equad_idx += 1

        elif "_ecorr" in key:
            param_name = (
                key.split("_ecorr")[0].split(psr_name)[1].split("_log10")[0][1:]
            )

            tp = maskParameter(
                name="ECORR",
                index=ecorr_idx,
                key="-f",
                key_value=param_name,
                value=10**val / 1e-6,
                units="us",
                convert_tcb2tdb=False,
            )
            ecorr_params.append(tp)
            ecorr_idx += 1

    # Create white noise components and add them to the model
    ef_eq_comp = pm.ScaleToaError()
    ef_eq_comp.remove_param(param="EFAC1")
    ef_eq_comp.remove_param(param="EQUAD1")
    ef_eq_comp.remove_param(param="TNEQ1")
    for efac_param in efac_params:
        ef_eq_comp.add_param(param=efac_param, setup=True)
    for equad_param in equad_params:
        ef_eq_comp.add_param(param=equad_param, setup=True)
    model.add_component(ef_eq_comp, validate=True, force=True)

    if len(ecorr_params) > 0:
        ec_comp = pm.EcorrNoise()
        ec_comp.remove_param("ECORR1")
        for ecorr_param in ecorr_params:
            ec_comp.add_param(param=ecorr_param, setup=True)
        model.add_component(ec_comp, validate=True, force=True)

    log.info(f"Including red noise for {psr_name}")
    # Add the ML RN parameters to their component
    rn_comp = pm.PLRedNoise()

    rn_keys = np.array([key for key, val in noise_dict.items() if "_red_" in key])
    rn_comp.RNAMP.quantity = convert_to_RNAMP(
        noise_dict[psr_name + "_red_noise_log10_A"],
    )
    rn_comp.RNIDX.quantity = -1 * noise_dict[psr_name + "_red_noise_gamma"]
    # Add red noise to the timing model
    model.add_component(rn_comp, validate=True, force=True)

    # Setup and validate the timing model to ensure things are correct
    model.setup()
    model.validate()

    return model
