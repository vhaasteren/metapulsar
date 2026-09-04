"""Default loguru sink: WARNING and above; user configuration is respected."""

import sys
import warnings

from loguru import logger

from metapulsar.log_config import configure_logging


def _stderr_has(capsys, level: str) -> bool:
    token = f"probe-{level.lower()}-line"
    getattr(logger, level.lower())(token)
    return token in capsys.readouterr().err


def test_backs_off_from_user_configuration():
    # After `import metapulsar` handler 0 is gone, so a user-installed sink
    # must be left alone.
    logger.remove()
    hid = logger.add(sys.stderr, level="DEBUG")
    try:
        assert configure_logging() is False
    finally:
        logger.remove(hid)
        configure_logging(force=True)


def test_force_installs_warning_sink(capsys):
    assert configure_logging(force=True) is True
    assert not _stderr_has(capsys, "DEBUG")
    assert not _stderr_has(capsys, "INFO")
    assert _stderr_has(capsys, "WARNING")


def test_log_file_sink(tmp_path):
    path = tmp_path / "metapulsar.log"
    configure_logging("INFO", force=True, log_file=str(path))
    try:
        logger.info("to-the-file")
    finally:
        configure_logging(force=True)
    assert "to-the-file" in path.read_text()


def test_pta_summary_leaves_logging_and_warnings_alone(capsys):
    """pta_summary must not remove sinks, disable loggers, or keep filters."""
    from metapulsar import MetaPulsarFactory

    configure_logging("DEBUG", force=True)
    filters_before = list(warnings.filters)
    try:
        MetaPulsarFactory().pta_summary({})
        capsys.readouterr()
        # The user's DEBUG sink still receives records...
        assert _stderr_has(capsys, "DEBUG")
        # ...including from metapulsar's own modules (nothing left disabled).
        from metapulsar.metapulsar_factory import logger as factory_logger

        factory_logger.debug("factory-probe-line")
        assert "factory-probe-line" in capsys.readouterr().err
        assert list(warnings.filters) == filters_before
    finally:
        configure_logging(force=True)
