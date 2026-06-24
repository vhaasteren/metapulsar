"""Thin frontend adapters for Discovery and Enterprise."""

from .discovery import discovery_signals
from .enterprise import enterprise_signal

__all__ = ["discovery_signals", "enterprise_signal"]
