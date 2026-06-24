"""Coordinate bijectors for timing-parameter transforms."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _is_jax_xp(xp) -> bool:
    return getattr(xp, "__name__", "").startswith("jax")


def _erf(x, xp):
    if _is_jax_xp(xp):
        import jax.scipy.special as jsp

        return jsp.erf(x)
    from scipy.special import erf

    return erf(x)


def _erfinv(x, xp):
    if _is_jax_xp(xp):
        import jax.scipy.special as jsp

        return jsp.erfinv(x)
    from scipy.special import erfinv

    return erfinv(x)


def _standard_normal_cdf(x, xp):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0), xp))


def _standard_normal_ppf(u, xp):
    clipped = xp.clip(u, 1e-12, 1.0 - 1e-12)
    return math.sqrt(2.0) * _erfinv(2.0 * clipped - 1.0, xp)


def _standard_normal_logpdf(x, xp):
    return -0.5 * x * x - 0.5 * math.log(2.0 * math.pi)


def _standard_normal_pdf(x, xp):
    return xp.exp(_standard_normal_logpdf(x, xp))


@dataclass(frozen=True)
class AxisPrior:
    """Per-parameter prior used for PIT-style delta<->z maps."""

    family: str
    lower: float | None = None
    upper: float | None = None
    mean: float = 0.0
    std: float = 1.0
    offset: float = 0.0


class PriorBijector:
    """Per-axis prior bijector mapping standardized z to physical delta."""

    def __init__(self, names: tuple[str, ...], priors: tuple[AxisPrior, ...]):
        if len(names) != len(priors):
            raise ValueError("names and priors length mismatch")
        self.names = names
        self.priors = priors

    @classmethod
    def from_normal(
        cls, names: tuple[str, ...], means: np.ndarray, stds: np.ndarray
    ) -> "PriorBijector":
        priors = tuple(
            AxisPrior(family="normal", mean=float(m), std=float(s))
            for m, s in zip(means, stds, strict=True)
        )
        return cls(names=names, priors=priors)

    @classmethod
    def from_uniform(
        cls, names: tuple[str, ...], lowers: np.ndarray, uppers: np.ndarray
    ) -> "PriorBijector":
        priors = tuple(
            AxisPrior(family="uniform", lower=float(lo), upper=float(hi))
            for lo, hi in zip(lowers, uppers, strict=True)
        )
        return cls(names=names, priors=priors)

    def delta_from_z(self, z, xp):
        z = xp.asarray(z)
        out = []
        for idx, prior in enumerate(self.priors):
            zi = z[idx]
            if prior.family == "normal":
                out.append(prior.mean + prior.std * zi)
            elif prior.family == "uniform":
                u = _standard_normal_cdf(zi, xp)
                out.append(prior.lower + (prior.upper - prior.lower) * u)
            elif prior.family == "log_uniform":
                u = _standard_normal_cdf(zi, xp)
                log_lower = math.log(prior.lower)
                log_width = math.log(prior.upper) - log_lower
                out.append(xp.exp(log_lower + log_width * u) - prior.offset)
            elif prior.family == "truncated_normal":
                alpha = (prior.lower - prior.mean) / prior.std
                beta = (prior.upper - prior.mean) / prior.std
                cdf_alpha = _standard_normal_cdf(alpha, xp)
                z_norm = _standard_normal_cdf(beta, xp) - cdf_alpha
                u = _standard_normal_cdf(zi, xp)
                out.append(
                    prior.mean
                    + prior.std * _standard_normal_ppf(cdf_alpha + u * z_norm, xp)
                )
            else:
                raise ValueError(f"Unsupported prior family: {prior.family}")
        return xp.stack(out)

    def z_from_delta(self, delta, xp):
        delta = xp.asarray(delta)
        out = []
        for idx, prior in enumerate(self.priors):
            di = delta[idx]
            if prior.family == "normal":
                out.append((di - prior.mean) / prior.std)
            elif prior.family == "uniform":
                u = (di - prior.lower) / (prior.upper - prior.lower)
                out.append(_standard_normal_ppf(u, xp))
            elif prior.family == "log_uniform":
                absolute = di + prior.offset
                u = (xp.log(absolute) - math.log(prior.lower)) / (
                    math.log(prior.upper) - math.log(prior.lower)
                )
                out.append(_standard_normal_ppf(u, xp))
            elif prior.family == "truncated_normal":
                alpha = (prior.lower - prior.mean) / prior.std
                beta = (prior.upper - prior.mean) / prior.std
                cdf_alpha = _standard_normal_cdf(alpha, xp)
                z_norm = _standard_normal_cdf(beta, xp) - cdf_alpha
                cdf_di = _standard_normal_cdf((di - prior.mean) / prior.std, xp)
                out.append(_standard_normal_ppf((cdf_di - cdf_alpha) / z_norm, xp))
            else:
                raise ValueError(f"Unsupported prior family: {prior.family}")
        return xp.stack(out)

    def delta_from_u(self, u, xp):
        u = xp.asarray(u)
        out = []
        for idx, prior in enumerate(self.priors):
            ui = xp.clip(u[idx], 1e-12, 1.0 - 1e-12)
            if prior.family == "normal":
                out.append(prior.mean + prior.std * _standard_normal_ppf(ui, xp))
            elif prior.family == "uniform":
                out.append(prior.lower + (prior.upper - prior.lower) * ui)
            elif prior.family == "log_uniform":
                log_lower = math.log(prior.lower)
                log_width = math.log(prior.upper) - log_lower
                out.append(xp.exp(log_lower + log_width * ui) - prior.offset)
            elif prior.family == "truncated_normal":
                alpha = (prior.lower - prior.mean) / prior.std
                beta = (prior.upper - prior.mean) / prior.std
                cdf_alpha = _standard_normal_cdf(alpha, xp)
                z_norm = _standard_normal_cdf(beta, xp) - cdf_alpha
                out.append(
                    prior.mean
                    + prior.std * _standard_normal_ppf(cdf_alpha + ui * z_norm, xp)
                )
            else:
                raise ValueError(f"Unsupported prior family: {prior.family}")
        return xp.stack(out)

    def logprior_physical(self, delta, xp):
        delta = xp.asarray(delta)
        terms = []
        for idx, prior in enumerate(self.priors):
            di = delta[idx]
            if prior.family == "normal":
                zi = (di - prior.mean) / prior.std
                terms.append(_standard_normal_logpdf(zi, xp) - math.log(prior.std))
            elif prior.family == "uniform":
                inside = xp.logical_and(di >= prior.lower, di <= prior.upper)
                terms.append(
                    xp.where(inside, -math.log(prior.upper - prior.lower), -xp.inf)
                )
            elif prior.family == "log_uniform":
                absolute = di + prior.offset
                inside = xp.logical_and(
                    absolute >= prior.lower, absolute <= prior.upper
                )
                terms.append(
                    xp.where(
                        inside,
                        -xp.log(absolute)
                        - math.log(math.log(prior.upper) - math.log(prior.lower)),
                        -xp.inf,
                    )
                )
            elif prior.family == "truncated_normal":
                zi = (di - prior.mean) / prior.std
                alpha = (prior.lower - prior.mean) / prior.std
                beta = (prior.upper - prior.mean) / prior.std
                z_norm = _standard_normal_cdf(beta, xp) - _standard_normal_cdf(
                    alpha, xp
                )
                inside = xp.logical_and(di >= prior.lower, di <= prior.upper)
                terms.append(
                    xp.where(
                        inside,
                        _standard_normal_logpdf(zi, xp)
                        - math.log(prior.std)
                        - xp.log(z_norm),
                        -xp.inf,
                    )
                )
            else:
                raise ValueError(f"Unsupported prior family: {prior.family}")
        return xp.sum(xp.stack(terms))

    def logabsdet_delta_from_z(self, z, xp):
        z = xp.asarray(z)
        terms = []
        for idx, prior in enumerate(self.priors):
            zi = z[idx]
            if prior.family == "normal":
                terms.append(xp.asarray(math.log(prior.std)))
            elif prior.family == "uniform":
                terms.append(
                    xp.asarray(math.log(prior.upper - prior.lower))
                    + _standard_normal_logpdf(zi, xp)
                )
            elif prior.family == "log_uniform":
                u = _standard_normal_cdf(zi, xp)
                log_lower = math.log(prior.lower)
                log_width = math.log(prior.upper) - log_lower
                absolute = xp.exp(log_lower + log_width * u)
                terms.append(
                    xp.log(absolute)
                    + math.log(log_width)
                    + _standard_normal_logpdf(zi, xp)
                )
            elif prior.family == "truncated_normal":
                delta = self.delta_from_z(z, xp)[idx]
                logpdf = PriorBijector((self.names[idx],), (prior,)).logprior_physical(
                    xp.stack([delta]), xp
                )
                terms.append(_standard_normal_logpdf(zi, xp) - logpdf)
            else:
                raise ValueError(f"Unsupported prior family: {prior.family}")
        return xp.sum(xp.stack(terms))

    def jacobian_diag_delta_from_z(self, z, xp):
        """Return per-axis d(delta_i) / d(z_i) at z."""
        z = xp.asarray(z)
        out = []
        for idx, prior in enumerate(self.priors):
            zi = z[idx]
            if prior.family == "normal":
                out.append(xp.asarray(prior.std))
            elif prior.family == "uniform":
                out.append(
                    xp.asarray(prior.upper - prior.lower) * _standard_normal_pdf(zi, xp)
                )
            elif prior.family == "log_uniform":
                u = _standard_normal_cdf(zi, xp)
                log_lower = math.log(prior.lower)
                log_width = math.log(prior.upper) - log_lower
                absolute = xp.exp(log_lower + log_width * u)
                out.append(absolute * log_width * _standard_normal_pdf(zi, xp))
            elif prior.family == "truncated_normal":
                delta = self.delta_from_z(z, xp)[idx]
                logpdf = PriorBijector((self.names[idx],), (prior,)).logprior_physical(
                    xp.stack([delta]), xp
                )
                out.append(_standard_normal_pdf(zi, xp) / xp.exp(logpdf))
            else:
                raise ValueError(f"Unsupported prior family: {prior.family}")
        return xp.stack(out)


class WhiteningLinear:
    """Joint linear layer mapping x <-> z for whitening/standardization."""

    def __init__(self, C: np.ndarray | None = None, z0: np.ndarray | None = None):
        if C is None:
            C = np.eye(0, dtype=float)
        C = np.asarray(C, dtype=float)
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("C must be square")
        if z0 is None:
            z0 = np.zeros(C.shape[0], dtype=float)
        z0 = np.asarray(z0, dtype=float)
        if z0.shape != (C.shape[0],):
            raise ValueError("z0 shape mismatch with C")
        self.C = C
        self.z0 = z0
        sign, logdet = np.linalg.slogdet(C) if C.size else (1.0, 0.0)
        if sign <= 0:
            raise ValueError("C must have positive determinant")
        self.logabsdet = float(logdet)

    @classmethod
    def identity(cls, ndim: int) -> "WhiteningLinear":
        return cls(C=np.eye(ndim, dtype=float), z0=np.zeros(ndim, dtype=float))

    def z_from_x(self, x, xp):
        x = xp.asarray(x)
        return xp.asarray(self.C) @ x + xp.asarray(self.z0)

    def x_from_z(self, z, xp):
        z = xp.asarray(z)
        return xp.linalg.solve(xp.asarray(self.C), z - xp.asarray(self.z0))


class Chain:
    """Convenience chain implementing x -> z -> delta composition."""

    def __init__(self, prior_bijector: PriorBijector, linear: WhiteningLinear):
        self.prior_bijector = prior_bijector
        self.linear = linear

    def delta_from_x(self, x, xp):
        z = self.linear.z_from_x(x, xp)
        return self.prior_bijector.delta_from_z(z, xp)

    def x_from_delta(self, delta, xp):
        z = self.prior_bijector.z_from_delta(delta, xp)
        return self.linear.x_from_z(z, xp)
