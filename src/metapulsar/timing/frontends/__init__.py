"""Thin likelihood-frontend adapters for Discovery and Enterprise.

``discovery_signals`` and ``enterprise_signal`` translate a bound
``NonLinearTimingModel`` (partition, ``ParameterSpace``, priors) into
Enterprise/Discovery signal objects. Timing priors are owned by
``ParameterSpace``; see the module docstrings in ``discovery`` and
``enterprise`` for how fallback cheat priors and PIT bounds are handled.
"""

from .discovery import discovery_signals
from .enterprise import enterprise_signal

__all__ = ["discovery_signals", "enterprise_signal"]
