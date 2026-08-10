"""Guard the MetaPulsar ↔ nltiming import boundary.

Basic MetaPulsar operations must not import ``nltiming``: the nonlinear-timing
coupling is lazy (function-local, on the ``timing_engine()``/engines path), and
the pure PINT name/unit helpers are MetaPulsar-owned in ``pint_compat.py``. A
module-level ``nltiming`` import reachable from ``import metapulsar`` regresses
this and re-breaks the field container that ships an older ``nltiming``.

Runs in a subprocess because the pytest process itself imports ``nltiming`` via
other test modules, so an in-process ``sys.modules`` check would be worthless.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_import_metapulsar_does_not_load_nltiming():
    script = textwrap.dedent(
        """
        import sys
        import metapulsar  # noqa: F401
        leaked = sorted(
            m for m in sys.modules
            if m == "nltiming" or m.startswith("nltiming.")
        )
        if leaked:
            raise SystemExit("nltiming leaked into `import metapulsar`: " + repr(leaked))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
