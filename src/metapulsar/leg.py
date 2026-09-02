"""One PTA leg as its timing package emitted it.

The object this module exists for is a *pair*: the frozen record and the engine
that produced it, from one read of one par/tim. That is the arrangement the
rest of this package has been working around. Today a MetaPulsar leg is built
by PINT or libstempo at construction and an engine is chosen afterwards, so
the design matrix the likelihood marginalizes and the residual the sampler
moves come from two different codes -- which is why
``validate_engine_against_pulsar`` existed and why it could never be turned on
in general.

A :class:`TimingLeg` removes the gap rather than validating it: the record's
``Mmat`` *is* the engine's own ``-J``, its residuals are the engine's, and
neither was computed twice.

``record`` holds views of the engine's arrays, not copies. Nothing may write
to either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from psrdata import PulsarData


@dataclass(frozen=True)
class TimingLeg:
    """A frozen record and the engine that emitted it, for one PTA."""

    record: PulsarData
    engine: Any  # ``vela_jax.Engine`` today
    engine_name: str  # "vela_jax"
    timing_package: str  # which package read the files: "pint" | "tempo2"

    @property
    def name(self) -> str:
        """The pulsar's name, as the package that read the files spelled it.

        A leg stands where a PINT ``(model, toas)`` pair or a libstempo pulsar
        used to, and ``MetaPulsar._extract_pulsar_names`` asks that object for
        its name.
        """
        return self.record.name

    @classmethod
    def vela_jax(
        cls,
        par_path,
        tim_path,
        *,
        timing_package: str,
        binary_conventions: str = "pint",
    ) -> "TimingLeg":
        """Read one leg with vela-jax and keep both halves of what it produced.

        ``binary_conventions`` stays Vela's on both timing packages, which is
        today's MetaPulsar default: every leg of one pulsar uses one residual
        formula unless a caller deliberately asks otherwise.
        """
        from vela_jax import Engine

        engine = Engine.from_files(
            str(par_path),
            str(tim_path),
            timing_package=timing_package,
            binary_conventions=binary_conventions,
        )
        return cls(
            record=engine.pulsar_data(),
            engine=engine,
            engine_name="vela_jax",
            timing_package=timing_package,
        )


__all__ = ["TimingLeg"]
