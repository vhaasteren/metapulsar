"""Default loguru sink for MetaPulsar.

loguru ships with one handler (id ``0``) that writes every ``DEBUG`` record to
stderr. PINT, nltiming, and MetaPulsar all log through loguru, so an
unconfigured session drowns in per-parameter debug lines. :func:`configure_logging`
installs a ``WARNING``-and-above stderr sink instead, and is called once at
``import metapulsar``. It only replaces loguru's untouched default handler, so
any configuration the user performed *before* the import is left alone, and
``pint.logging.setup()`` / ``logger.add`` calls *after* the import win as usual.
"""

from __future__ import annotations

import sys

from loguru import logger

DEFAULT_LEVEL = "WARNING"


def configure_logging(
    level: str = DEFAULT_LEVEL, *, force: bool = False, log_file: str | None = None
) -> bool:
    """Route loguru output to stderr (and optionally a file) at ``level`` and above.

    Args:
        level: loguru level name (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ...).
        force: Replace *all* current handlers. When ``False`` (the default),
            only loguru's built-in handler ``0`` is replaced, so an explicit
            user configuration is respected.
        log_file: Also write records at ``level`` and above to this file.

    Returns:
        ``True`` if a sink was installed, ``False`` if loguru was already
        configured and ``force`` was not set.
    """
    if force:
        logger.remove()
    else:
        try:
            logger.remove(0)
        except ValueError:
            return False
    logger.add(sys.stderr, level=level)
    if log_file:
        logger.add(log_file, level=level)
    return True


# Installed at import so that package ``__init__`` can simply import this module
# first; see the module docstring for the back-off rule.
configure_logging()
