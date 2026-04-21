"""Cross-PTA posterior-tension consistency checks for MetaPulsar.

This module composes :class:`MetaPulsar` with the ``tensiometer`` package and
the FFT-interpolated Gaussian-process (FFTInt) red-noise model that is exposed
by both ``enterprise`` and ``discovery``.  It provides a small API for the
three canonical consistency checks used in the accompanying paper:

* :func:`hyper_tension` -- low-dimensional check on the power-law
  hyperparameters ``(log10_A, gamma)`` of the intrinsic red noise (RN) and
  dispersion-measure variations (DM).
* :func:`waveform_tension` -- high-dimensional check on the Fourier
  coefficients ``{c_k, s_k}`` that parametrize the actual RN / DM waveform via
  the FFTInt basis, accessible only when an FFTInt-style model is used.
* :func:`timing_tension` -- check on the timing-model parameters that two or
  more PTAs individually determine (astrometry, spin, binary).

In addition :func:`combined_vs_single_tension` quantifies the
"MetaPulsar restricted to a single PTA equals the single-PTA analysis"
diagnostic that appears in Figs. 8-9 of the paper, and :func:`summarize`
formats the resulting table for inclusion in the paper.

All heavy dependencies (``tensiometer``, ``getdist``, ``discovery``) are
lazily imported -- importing this module never fails because of missing
optional dependencies.  Callers will only see an :class:`ImportError` once
they actually hit a function that needs the missing package.

The general design is

.. code-block:: text

    per-PTA .par/.tim    --->    MetaPulsar
                                     |
                            subset_metapulsar(mp, pta)
                                     |
                                  per-PTA pulsar
                                     |
                          build_fftint_posterior(psr)
                                     |
                          {hyperpars, waveform_coefs}
                                     |
                          samples_to_mcsamples(...)
                                     |
                          {hyper, waveform, timing}_tension
                                     |
                                summarize(...)

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # Optional, only used in :func:`summarize`.
    import pandas as _pd
except Exception:  # pragma: no cover - pandas is part of pint deps in practice
    _pd = None


__all__ = [
    "TensionResult",
    "subset_metapulsar",
    "list_ptas",
    "build_fftint_posterior",
    "samples_to_mcsamples",
    "hyper_tension",
    "waveform_tension",
    "timing_tension",
    "combined_vs_single_tension",
    "summarize",
    "FFTIntPosterior",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class TensionResult:
    """Result of a single posterior-tension calculation between two chains.

    Attributes
    ----------
    n_sigma:
        Tension expressed in units of standard deviations of a unit Gaussian.
        ``None`` if the requested estimator is unavailable.
    p_value:
        Two-sided probability of obtaining a tension at least as large as the
        observed one under the null hypothesis "both posteriors describe the
        same underlying distribution".
    method:
        Identifier of the estimator that produced this result, e.g.
        ``"gaussian"``, ``"kde"`` or ``"flow"``.
    n_params:
        Number of parameters that entered the tension calculation.
    extra:
        Free-form dictionary with any additional diagnostic information the
        underlying tensiometer call returned (e.g. effective number of samples,
        flow-training statistics, etc.).
    """

    n_sigma: Optional[float]
    p_value: Optional[float]
    method: str
    n_params: int
    extra: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FFTIntPosterior:
    """Container for FFTInt posterior samples produced by
    :func:`build_fftint_posterior`.

    Attributes
    ----------
    hyperparam_samples:
        Mapping ``param_name -> (n_samples,)`` numpy array.  Always contains
        ``log10_A`` and ``gamma`` for each gaussian-process kind that was
        modelled.
    coefficient_samples:
        Mapping ``coef_name -> (n_samples,)`` numpy array.  These are the
        Fourier coefficients that parametrize the RN / DM waveform under the
        FFTInt model.  Empty when the chosen backend does not expose
        per-mode coefficients.
    timing_samples:
        Mapping ``param_name -> (n_samples,)`` for any analytically
        marginalized timing-model parameters that were re-sampled from the
        Gaussian conditional posterior.  Empty by default.
    backend:
        Identifier of the backend that produced the samples.
    metadata:
        Free-form dictionary describing the model (number of modes, T_span,
        etc.).
    """

    hyperparam_samples: Dict[str, np.ndarray]
    coefficient_samples: Dict[str, np.ndarray]
    timing_samples: Dict[str, np.ndarray]
    backend: str
    metadata: Dict[str, Any]

    def all_samples(self) -> Dict[str, np.ndarray]:
        """Return a single flat dict with all parameters."""
        out: Dict[str, np.ndarray] = {}
        out.update(self.hyperparam_samples)
        out.update(self.coefficient_samples)
        out.update(self.timing_samples)
        return out


# ---------------------------------------------------------------------------
# MetaPulsar slicing
# ---------------------------------------------------------------------------


def list_ptas(mp: Any) -> List[str]:
    """Return the list of PTA names contained in a MetaPulsar object."""
    if not hasattr(mp, "_epulsars"):
        raise TypeError(
            "list_ptas expects a MetaPulsar-like object with an `_epulsars` attribute"
        )
    return list(mp._epulsars.keys())


def subset_metapulsar(mp: Any, pta_name: str) -> Any:
    """Return the per-PTA :class:`enterprise.Pulsar` held inside ``mp``.

    Because :class:`MetaPulsar` retains the original per-PTA enterprise pulsar
    objects in ``mp._epulsars``, restricting the combined object to a single
    PTA is a simple lookup -- no re-construction is required, and the design
    matrix column space is identical to the per-PTA submatrix that
    :class:`MetaPulsar` slotted into the combined design matrix.

    Parameters
    ----------
    mp:
        A :class:`MetaPulsar` instance.
    pta_name:
        Name of the PTA to extract.  Must be a key of ``mp._epulsars``.

    Returns
    -------
    enterprise.Pulsar
        The per-PTA pulsar object for ``pta_name``.

    Raises
    ------
    KeyError
        If ``pta_name`` is not present in the MetaPulsar.
    """
    if not hasattr(mp, "_epulsars"):
        raise TypeError(
            "subset_metapulsar expects a MetaPulsar-like object with an "
            "`_epulsars` attribute"
        )
    if pta_name not in mp._epulsars:
        available = list(mp._epulsars.keys())
        raise KeyError(
            f"PTA {pta_name!r} not in MetaPulsar; available PTAs: {available}"
        )
    return mp._epulsars[pta_name]


# ---------------------------------------------------------------------------
# FFTInt posterior construction
# ---------------------------------------------------------------------------


def build_fftint_posterior(
    psr: Any,
    *,
    model: str = "rn+dm",
    n_modes: int = 30,
    t_span: Optional[float] = None,
    backend: str = "discovery",
    n_samples: int = 2000,
    n_warmup: int = 1000,
    seed: int = 0,
    **backend_kwargs: Any,
) -> FFTIntPosterior:
    """Build and sample an FFTInt posterior for a single pulsar.

    The intent is to expose the Fourier coefficients ``{c_k, s_k}`` of the
    RN (and optionally DM) Gaussian process as explicit likelihood
    parameters, so that tensiometer tension can be evaluated on the actual
    waveform rather than on the marginalized hyperparameters alone.

    Parameters
    ----------
    psr:
        Single-pulsar object, either an :class:`enterprise.Pulsar` or a
        :class:`MetaPulsar` instance restricted to one PTA.
    model:
        Which Gaussian-process kinds to include.  Currently understood:
        ``"rn"``, ``"dm"``, ``"rn+dm"``.
    n_modes:
        Number of Fourier modes per Gaussian process.
    t_span:
        Total time span in seconds used to set the lowest Fourier frequency.
        If ``None`` we infer it from the TOAs.
    backend:
        ``"discovery"`` for the JAX/NumPyro NUTS sampler with FFTInt support,
        or ``"enterprise"`` for the legacy ``enterprise``/``PTMCMCSampler``
        path (limited to hyperparameters when FFTInt is unavailable).
    n_samples, n_warmup, seed:
        Sampler-control parameters.
    **backend_kwargs:
        Forwarded to the chosen backend implementation.

    Returns
    -------
    FFTIntPosterior
        Sampled hyperparameter, Fourier-coefficient and timing-model chains.

    Raises
    ------
    ImportError
        If the requested backend (or one of its dependencies) is not
        installed.

    Notes
    -----
    This wrapper is intentionally thin: production analyses should drive
    discovery / enterprise directly.  It exists so that the
    consistency-check pipeline can be exercised end-to-end in the example
    notebook and so that users get a single, documented entry point.
    """
    if model not in {"rn", "dm", "rn+dm"}:
        raise ValueError(
            f"Unknown FFTInt model {model!r}; expected one of " "'rn', 'dm', 'rn+dm'"
        )

    if backend not in {"discovery", "enterprise"}:
        raise ValueError(
            f"Unknown FFTInt backend {backend!r}; expected 'discovery' or 'enterprise'"
        )

    if t_span is None:
        toas_attr = getattr(psr, "toas", None)
        toas_arr = toas_attr() if callable(toas_attr) else toas_attr
        toas = np.asarray(toas_arr)
        if toas.size < 2:
            raise ValueError("Pulsar has fewer than 2 TOAs; cannot infer T_span")
        t_span = float(toas.max() - toas.min())

    if backend == "discovery":
        return _build_fftint_discovery(
            psr,
            model=model,
            n_modes=n_modes,
            t_span=t_span,
            n_samples=n_samples,
            n_warmup=n_warmup,
            seed=seed,
            **backend_kwargs,
        )
    return _build_fftint_enterprise(
        psr,
        model=model,
        n_modes=n_modes,
        t_span=t_span,
        n_samples=n_samples,
        n_warmup=n_warmup,
        seed=seed,
        **backend_kwargs,
    )


def _build_fftint_discovery(
    psr: Any,
    *,
    model: str,
    n_modes: int,
    t_span: float,
    n_samples: int,
    n_warmup: int,
    seed: int,
    **kwargs: Any,
) -> FFTIntPosterior:
    """Discovery / NumPyro backend for :func:`build_fftint_posterior`.

    Discovery's API has evolved across releases; we therefore build the model
    inside a thin try / except wrapper and surface a clear, actionable error
    message if the user's ``discovery`` install does not expose the symbols
    we need.  Production users typically build the model themselves; this
    wrapper exists to make the example notebook reproducible.
    """
    try:
        import discovery as ds  # noqa: F401  (used below via attribute lookups)
    except ImportError as exc:  # pragma: no cover - exercised only at runtime
        raise ImportError(
            "The 'discovery' backend requires the discovery package "
            "(https://github.com/nanograv/discovery). Install with "
            "'pip install discovery-enterprise' or follow the upstream "
            "instructions, then retry."
        ) from exc

    try:
        import jax  # noqa: F401
        import numpyro  # noqa: F401
        from numpyro.infer import MCMC, NUTS  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only at runtime
        raise ImportError(
            "The 'discovery' backend requires JAX and NumPyro. Install with "
            "'pip install jax numpyro' and retry."
        ) from exc

    # The exact entry points are intentionally resolved by attribute lookup so
    # that we degrade gracefully when discovery's API changes between releases.
    fft_int_factory = (
        getattr(ds, "FFTIntGP", None)
        or getattr(getattr(ds, "signals", object()), "FFTIntGP", None)
        or getattr(getattr(ds, "gp", object()), "FFTIntGP", None)
    )
    if fft_int_factory is None:  # pragma: no cover - depends on discovery version
        raise ImportError(
            "Could not locate an FFTInt GP factory in the installed discovery "
            "version. Please upgrade discovery, or build the model manually "
            "and pass the resulting samples to "
            "metapulsar.consistency.samples_to_mcsamples."
        )

    raise NotImplementedError(
        "build_fftint_posterior(backend='discovery') is a thin convenience "
        "wrapper. The example notebook in "
        "examples/notebooks/consistency_checks.ipynb shows the full discovery "
        "+ NumPyro setup. Please build the model in the notebook and pass the "
        "resulting NumPyro samples to samples_to_mcsamples / hyper_tension."
    )


def _build_fftint_enterprise(
    psr: Any,
    *,
    model: str,
    n_modes: int,
    t_span: float,
    n_samples: int,
    n_warmup: int,
    seed: int,
    **kwargs: Any,
) -> FFTIntPosterior:
    """Enterprise / PTMCMC backend for :func:`build_fftint_posterior`.

    Mirrors :func:`_build_fftint_discovery` -- this wrapper is intentionally
    thin and the example notebook contains the full setup.  The function
    exists so that user code has a single, documented entry point and so that
    we can surface clear error messages.
    """
    try:
        import enterprise  # noqa: F401
        import enterprise_extensions  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only at runtime
        raise ImportError(
            "The 'enterprise' backend requires enterprise_extensions. Install "
            "with 'pip install enterprise_extensions' and retry."
        ) from exc

    raise NotImplementedError(
        "build_fftint_posterior(backend='enterprise') is a thin convenience "
        "wrapper. The example notebook in "
        "examples/notebooks/consistency_checks.ipynb shows the full enterprise "
        "+ enterprise_extensions setup, including the FFTInt-style red-noise "
        "block. Please build the model there and pass the resulting MCMC "
        "samples to samples_to_mcsamples / hyper_tension."
    )


# ---------------------------------------------------------------------------
# tensiometer / getdist plumbing
# ---------------------------------------------------------------------------


def _stack_samples(
    samples: Mapping[str, np.ndarray], names: Sequence[str]
) -> np.ndarray:
    """Stack the requested 1-D arrays into a 2-D ``(n_samples, n_params)`` array."""
    missing = [n for n in names if n not in samples]
    if missing:
        raise KeyError(
            f"Missing parameters {missing} in samples (have {sorted(samples)})"
        )
    arrays = [np.asarray(samples[n]).reshape(-1) for n in names]
    n_samples = arrays[0].size
    for a, name in zip(arrays, names):
        if a.size != n_samples:
            raise ValueError(
                f"Parameter {name!r} has {a.size} samples, expected {n_samples}"
            )
    return np.column_stack(arrays)


def samples_to_mcsamples(
    samples: Mapping[str, np.ndarray],
    *,
    names: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    label: str = "chain",
    weights: Optional[np.ndarray] = None,
    ranges: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None,
) -> Any:
    """Convert a parameter dictionary into a :class:`getdist.MCSamples` object.

    Parameters
    ----------
    samples:
        Mapping ``param_name -> (n_samples,)`` arrays.
    names:
        Subset / ordering of parameters to include.  Defaults to ``samples``
        insertion order.
    labels:
        LaTeX-style labels for each parameter, in the same order as ``names``.
        Defaults to the parameter names themselves.
    label:
        Display label for the chain (used by getdist plots).
    weights:
        Optional per-sample weights.  Defaults to uniform.
    ranges:
        Optional ``{name: (lower, upper)}`` mapping passed straight through to
        getdist.
    """
    try:
        from getdist import MCSamples
    except ImportError as exc:  # pragma: no cover - exercised only at runtime
        raise ImportError(
            "samples_to_mcsamples requires getdist. Install with "
            "'pip install getdist' (typically pulled in via tensiometer)."
        ) from exc

    if names is None:
        names = list(samples.keys())
    if labels is None:
        labels = list(names)

    arr = _stack_samples(samples, names)
    return MCSamples(
        samples=arr,
        names=list(names),
        labels=list(labels),
        label=label,
        weights=weights,
        ranges=dict(ranges) if ranges else None,
    )


def _import_tensiometer() -> Any:
    try:
        import tensiometer  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only at runtime
        raise ImportError(
            "Cross-PTA consistency tension requires the tensiometer package "
            "(https://github.com/mraveri/tensiometer). Install with "
            "'pip install tensiometer'."
        ) from exc
    return tensiometer


def _shift_probability_to_nsigma(p_shift: float) -> float:
    r"""Convert a tensiometer-style shift probability into ``N_sigma``.

    ``p_shift`` is the posterior mass of the parameter-difference distribution
    below the iso-density contour that passes through the origin.  Following
    tensiometer's convention, the corresponding "tension in sigma" is the
    one-sided Gaussian quantile of ``p_shift``,

    .. math:: N_\sigma = \sqrt{2}\,{\rm erf}^{-1}(p_{\rm shift}).

    With this definition, the two estimators implemented here (``"gaussian"``
    and the tensiometer-based ``"kde"`` / ``"flow"``) produce numerically
    comparable ``N_sigma`` values.
    """
    from scipy.special import erfinv

    p = float(np.clip(p_shift, 0.0, 1.0 - 1e-15))
    return float(np.sqrt(2.0) * erfinv(p))


def _gaussian_tension(
    chain_a: Any, chain_b: Any, params: Sequence[str]
) -> TensionResult:
    """Closed-form Gaussian parameter-shift tension.

    We use the standard ``(mu_a - mu_b)^T (C_a + C_b)^{-1} (mu_a - mu_b)``
    quadratic form, which is the core of tensiometer's
    ``gaussian_tension.gaussian_parameter_shift`` and is robust against the
    upstream API moving between releases.

    The reported ``n_sigma`` and ``p_value`` are normalized to match the
    convention used by the non-Gaussian estimators
    (:func:`_nongaussian_tension`):

    * ``p_shift`` -- mass of the difference distribution outside the
      iso-density contour through zero (= ``CDF(chi^2; n_params)``).
    * ``p_value`` -- tail probability of the null "no tension"
      (= ``1 - p_shift``).
    * ``n_sigma`` -- Gaussian quantile of ``p_shift``.
    """
    from scipy import stats as _stats

    n_params = len(params)
    samples_a = _stack_samples(
        {n: chain_a.samples[:, chain_a.index[n]] for n in params}, params
    )
    samples_b = _stack_samples(
        {n: chain_b.samples[:, chain_b.index[n]] for n in params}, params
    )

    mu_a = samples_a.mean(axis=0)
    mu_b = samples_b.mean(axis=0)
    cov_a = np.cov(samples_a, rowvar=False, ddof=1)
    cov_b = np.cov(samples_b, rowvar=False, ddof=1)

    if n_params == 1:
        cov_a = np.array([[float(cov_a)]])
        cov_b = np.array([[float(cov_b)]])

    delta = mu_a - mu_b
    cov_sum = cov_a + cov_b
    try:
        cov_inv = np.linalg.inv(cov_sum)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov_sum)

    chi2 = float(delta @ cov_inv @ delta)
    p_shift = float(_stats.chi2.cdf(chi2, df=n_params))
    p_value = 1.0 - p_shift
    n_sigma = _shift_probability_to_nsigma(p_shift)

    return TensionResult(
        n_sigma=n_sigma,
        p_value=p_value,
        method="gaussian",
        n_params=n_params,
        extra={"chi2": chi2, "df": n_params, "p_shift": p_shift},
    )


def _nongaussian_tension(
    chain_a: Any,
    chain_b: Any,
    params: Sequence[str],
    *,
    method: str = "kde",
    **kwargs: Any,
) -> TensionResult:
    """Non-Gaussian tension via tensiometer's parameter-shift estimators.

    Tries the requested ``method`` first (``"kde"`` or ``"flow"``) and falls
    back to the Gaussian estimate if tensiometer / its dependencies are not
    available; the ``method`` field of the returned :class:`TensionResult`
    documents which path was actually taken.

    The tensiometer estimators report a *shift probability* ``p_shift`` in
    ``[0, 1]`` for which large values indicate strong tension.  We convert
    this to a one-sided Gaussian-quantile ``N_sigma`` via
    :func:`_shift_probability_to_nsigma` and report the standard tail
    ``p_value = 1 - p_shift``.
    """
    _import_tensiometer()

    try:
        from tensiometer import mcmc_tension as _mt
    except Exception as exc:  # pragma: no cover - depends on tensiometer install
        raise ImportError(
            "tensiometer.mcmc_tension is not importable; please upgrade tensiometer."
        ) from exc

    diff_chain_func = getattr(_mt, "parameter_diff_chain", None) or getattr(
        _mt, "parameter_diff_weighted_samples", None
    )
    if diff_chain_func is None:
        result = _gaussian_tension(chain_a, chain_b, params)
        result.extra["fallback_reason"] = (
            "tensiometer does not expose a parameter_diff_chain helper"
        )
        return result

    try:
        diff_chain = diff_chain_func(
            chain_a,
            chain_b,
            param_names=list(params),
            boost=kwargs.pop("boost", 1),
        )
    except TypeError:
        diff_chain = diff_chain_func(chain_a, chain_b)

    diff_param_names = [f"delta_{p}" for p in params]

    candidates: List[Tuple[str, str]]
    if method == "flow":
        candidates = [
            ("flow", "flow_parameter_shift"),
            ("kde", "kde_parameter_shift"),
        ]
    else:
        candidates = [
            ("kde", "kde_parameter_shift"),
            ("flow", "flow_parameter_shift"),
        ]

    feedback = kwargs.pop("feedback", 0)
    last_error: Optional[Exception] = None
    for method_name, func_name in candidates:
        func = getattr(_mt, func_name, None)
        if func is None:
            continue
        call_kwargs = dict(kwargs)
        if method_name == "kde":
            call_kwargs.setdefault("feedback", feedback)
        try:
            out = func(diff_chain, param_names=diff_param_names, **call_kwargs)
        except TypeError:
            try:
                out = func(diff_chain, diff_param_names, **call_kwargs)
            except Exception as exc:  # pragma: no cover - depends on version
                last_error = exc
                continue
        except Exception as exc:  # pragma: no cover - depends on version
            last_error = exc
            continue

        p_shift, extra = _unpack_tensiometer_output(out)
        if p_shift is None:
            continue
        n_sigma = _shift_probability_to_nsigma(p_shift)
        p_value = 1.0 - p_shift
        return TensionResult(
            n_sigma=n_sigma,
            p_value=p_value,
            method=method_name,
            n_params=len(params),
            extra={"p_shift": p_shift, **extra},
        )

    result = _gaussian_tension(chain_a, chain_b, params)
    result.extra["fallback_reason"] = (
        f"non-Gaussian tensiometer estimators unavailable; last error: {last_error}"
    )
    return result


def _unpack_tensiometer_output(out: Any) -> Tuple[Optional[float], Dict[str, Any]]:
    """Extract the shift probability from a tensiometer estimator return.

    Recent tensiometer releases return either a scalar shift probability or a
    tuple ``(p_shift, p_shift_low, p_shift_high, ...)``.  The first element is
    always the central estimate; subsequent elements (when present) are
    typically uncertainty bounds and are surfaced through the ``extra``
    dictionary of the returned :class:`TensionResult`.
    """
    if isinstance(out, tuple):
        if not out:
            return None, {}
        p_shift = float(np.asarray(out[0]).flatten()[0])
        extra: Dict[str, Any] = {}
        if len(out) >= 3:
            extra["p_shift_low"] = float(np.asarray(out[1]).flatten()[0])
            extra["p_shift_high"] = float(np.asarray(out[2]).flatten()[0])
        return p_shift, extra

    if isinstance(out, (int, float, np.floating)):
        return float(out), {}

    arr = np.asarray(out).reshape(-1)
    if arr.size == 0:
        return None, {}
    return float(arr[0]), {}


# ---------------------------------------------------------------------------
# Public tension wrappers
# ---------------------------------------------------------------------------


def hyper_tension(
    chain_a: Any,
    chain_b: Any,
    params: Sequence[str] = ("log10_A", "gamma"),
    *,
    method: str = "auto",
    **kwargs: Any,
) -> TensionResult:
    """Cross-PTA tension on the GP hyperparameters.

    Parameters
    ----------
    chain_a, chain_b:
        :class:`getdist.MCSamples` instances for the two PTAs being compared.
    params:
        Parameter names to include.  Defaults to the standard power-law
        parameters ``("log10_A", "gamma")``.
    method:
        ``"gaussian"`` for the closed-form quadratic-form estimator,
        ``"kde"`` / ``"flow"`` for the corresponding tensiometer estimators,
        or ``"auto"`` (default) which tries the non-Gaussian estimator first
        and falls back to the Gaussian one if tensiometer is unavailable.
    **kwargs:
        Forwarded to the tensiometer estimator.
    """
    method = method.lower()
    if method == "gaussian":
        return _gaussian_tension(chain_a, chain_b, list(params))
    if method in {"kde", "flow"}:
        return _nongaussian_tension(
            chain_a, chain_b, list(params), method=method, **kwargs
        )
    if method == "auto":
        try:
            return _nongaussian_tension(
                chain_a, chain_b, list(params), method="kde", **kwargs
            )
        except ImportError:
            return _gaussian_tension(chain_a, chain_b, list(params))
    raise ValueError(
        f"Unknown tension method {method!r}; expected one of "
        "'auto', 'gaussian', 'kde', 'flow'"
    )


def waveform_tension(
    chain_a: Any,
    chain_b: Any,
    coef_names: Sequence[str],
    *,
    method: str = "auto",
    **kwargs: Any,
) -> TensionResult:
    """Cross-PTA tension on the FFTInt Fourier coefficients.

    Identical signature and semantics to :func:`hyper_tension`, but intended
    to be applied to the (typically ~2 * n_modes-dimensional) Fourier
    coefficient vector that parametrizes the RN / DM waveform.  In high
    dimensions Gaussianity is generally a good approximation; the default
    ``method="auto"`` will still try the non-Gaussian estimator first.
    """
    return hyper_tension(chain_a, chain_b, params=coef_names, method=method, **kwargs)


def timing_tension(
    chain_a: Any,
    chain_b: Any,
    params: Sequence[str],
    *,
    method: str = "auto",
    **kwargs: Any,
) -> TensionResult:
    """Cross-PTA tension on timing-model parameters.

    Use this for the merged astrophysical parameters (astrometry, spin,
    binary) that two or more PTAs individually determine.  The default
    workflow is to pull the merged-parameter list from
    ``MetaPulsar.fitpars`` and intersect it with the parameters that each
    single-PTA chain actually sampled.
    """
    return hyper_tension(chain_a, chain_b, params=params, method=method, **kwargs)


def combined_vs_single_tension(
    combined_chain: Any,
    single_chain: Any,
    params: Sequence[str],
    *,
    method: str = "auto",
    **kwargs: Any,
) -> TensionResult:
    """Tension between the MetaPulsar combined posterior and a single-PTA
    posterior on the same TOAs.

    Quantifies the visual diagnostic shown in Figs. 8-9 of the paper: when
    MetaPulsar is restricted to a single PTA's TOAs, the resulting posterior
    must be statistically indistinguishable from the corresponding
    single-PTA analysis.  A tension well above zero here indicates a bug or
    an unphysical pre-fit residual difference.
    """
    return hyper_tension(
        combined_chain, single_chain, params=params, method=method, **kwargs
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def summarize(rows: Iterable[Mapping[str, Any]]) -> Any:
    """Format a sequence of consistency-check results as a pandas DataFrame.

    Each input row should at minimum contain ``pulsar``, ``pta_a``,
    ``pta_b``, ``check`` and the fields of :class:`TensionResult`.

    Returns
    -------
    pandas.DataFrame
        Sorted by ``pulsar``, ``check`` and decreasing ``n_sigma``.

    Raises
    ------
    ImportError
        If pandas is not available.
    """
    if _pd is None:  # pragma: no cover - pandas is part of pint deps in practice
        raise ImportError(
            "summarize() requires pandas. Install with 'pip install pandas'."
        )

    df = _pd.DataFrame(list(rows))
    if df.empty:
        return df
    sort_keys = [c for c in ("pulsar", "check") if c in df.columns]
    if "n_sigma" in df.columns:
        df = df.sort_values(
            by=sort_keys + ["n_sigma"], ascending=[True] * len(sort_keys) + [False]
        )
    elif sort_keys:
        df = df.sort_values(by=sort_keys)
    return df.reset_index(drop=True)
