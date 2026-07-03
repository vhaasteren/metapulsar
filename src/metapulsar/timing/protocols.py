"""Protocols for pulsars and timing-engine adapters.

Stack layering:
- **Timing backend / timing engine** — residuals and design matrix (JUG, PINT, tempo2).
- **Likelihood frontend** — Enterprise / Discovery signal assembly (``frontends/*``).
- **Sampler** — user-owned posterior driver (PTMCMC, NumPyro NUTS, …); not imported here.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EphemerisExtras(Protocol):
    """Optional ephemeris-related pulsar fields used by some signal blocks."""

    @property
    def pos_t(self) -> np.ndarray: ...

    @property
    def planetssb(self) -> np.ndarray: ...

    @property
    def sunssb(self) -> np.ndarray: ...

    @property
    def theta(self) -> float: ...

    @property
    def phi(self) -> float: ...

    @property
    def pdist(self) -> tuple[float, float] | Any: ...

    @property
    def dm(self) -> float | Any: ...

    @property
    def dmx(self) -> Mapping[str, Any]: ...

    @property
    def telescope(self) -> np.ndarray: ...


@runtime_checkable
class EnterprisePulsarLike(Protocol):
    """Duck-typed pulsar surface consumed by likelihood frontends."""

    name: str
    fitpars: list[str] | tuple[str, ...]

    @property
    def toas(self) -> np.ndarray: ...

    @property
    def residuals(self) -> np.ndarray: ...

    @property
    def toaerrs(self) -> np.ndarray: ...

    @property
    def freqs(self) -> np.ndarray: ...

    @property
    def Mmat(self) -> np.ndarray: ...

    @property
    def flags(self) -> Mapping[str, np.ndarray]: ...

    @property
    def backend_flags(self) -> np.ndarray: ...


@runtime_checkable
class TimingBackend(Protocol):
    """Timing-engine adapter around theta-native engines in canonical pulsar order."""

    fitpars: tuple[str, ...]
    native_units: Mapping[str, str]

    def reference_theta(self) -> np.ndarray: ...

    def reference_theta_exact(self) -> Mapping[str, str]: ...

    def residual_delta(self, delta_theta: np.ndarray) -> np.ndarray: ...

    def design_matrix(self, params: Any | None = None) -> np.ndarray: ...


@runtime_checkable
class JaxTimingBackend(TimingBackend, Protocol):
    """JAX-capable timing backend for traced residuals on the NumPyro NUTS tier."""

    def residual_delta_jax(self, delta_theta: Any) -> Any: ...

    def precision_critical_fitpars(self) -> frozenset[str]: ...


@runtime_checkable
class PulsarInterface(EnterprisePulsarLike, Protocol):
    """Pulsar protocol: frozen arrays plus timing-engine accessors."""

    def pint_model(self) -> Any: ...

    def timing_backend(self, engines="jug") -> TimingBackend: ...

    def can_use_engines(self, engines="jug") -> bool: ...

    def cache_token(self) -> str | None: ...
