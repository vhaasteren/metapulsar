"""Tests for MetaPulsar parfile comment headers."""

from __future__ import annotations

from metapulsar.parameter_manager import AlignmentPolicy
from metapulsar.parfile_header import (
    alignment_policy_header_items,
    combination_options_header_items,
    ensure_metapulsar_par_header,
    format_metapulsar_par_header,
    strip_metapulsar_par_header,
)


def test_format_metapulsar_par_header_basic():
    text = format_metapulsar_par_header(
        created="2026-01-01T00:00:00",
        extra={"Product": "combination", "reference_pta": "ng9"},
    )
    assert text.startswith("# Created: 2026-01-01T00:00:00\n")
    assert "# Format:  PINT\n" in text
    assert "# By:      MetaPulsar\n" in text
    assert "# Product: combination\n" in text
    assert "# reference_pta: ng9\n" in text


def test_strip_and_ensure_replace_header():
    body = "# Created: old\n# Format:  PINT\n# By:      MetaPulsar\nPSR J0000+0000\n"
    stripped = strip_metapulsar_par_header(body)
    assert stripped.startswith("PSR J0000+0000")
    out = ensure_metapulsar_par_header(
        body, created="2026-08-09T12:00:00", extra={"Product": "gls-optimized"}
    )
    assert out.count("# Created:") == 1
    assert "# Created: 2026-08-09T12:00:00" in out
    assert "# Product: gls-optimized" in out
    assert "PSR J0000+0000" in out


class TestStripLeadingHeaderBlock:
    """The whole leading MetaPulsar ``#`` block goes, extras included."""

    def test_removes_product_extras_with_the_trio(self):
        body = (
            "# Created: old\n"
            "# Format:  PINT\n"
            "# By:      MetaPulsar\n"
            "# Product: combination\n"
            "# reference_pta: nanograv_9y\n"
            "# alignment_policy.ephem: DE421\n"
            "\n"
            "PSR J0000+0000\n"
        )
        assert strip_metapulsar_par_header(body) == "PSR J0000+0000\n"

    def test_foreign_leading_comment_is_kept(self):
        body = "# Simple pulsar parfile for testing\nPSR J0000+0000\n"
        assert strip_metapulsar_par_header(body) == body

    def test_body_comments_survive(self):
        body = (
            "# Created: old\n"
            "# Format:  PINT\n"
            "# By:      MetaPulsar\n"
            "PSR J0000+0000\n"
            "\n"
            "# MetaPulsar combination: JUMP from all PTAs\n"
            "JUMP -pta ppta_dr2 0.0 1\n"
        )
        stripped = strip_metapulsar_par_header(body)
        assert stripped.startswith("PSR J0000+0000")
        assert "# MetaPulsar combination: JUMP from all PTAs" in stripped

    def test_ensure_is_idempotent(self):
        body = "PSR J0000+0000\n"
        once = ensure_metapulsar_par_header(body, extra={"Product": "combination"})
        twice = ensure_metapulsar_par_header(once, extra={"Product": "gls-optimized"})
        assert twice.count("# Created:") == 1
        assert twice.count("# Product:") == 1
        assert "# Product: gls-optimized" in twice
        assert "# Product: combination" not in twice


def test_combination_options_include_alignment_policy():
    policy = AlignmentPolicy(unsupported="error", ephem="DE440")
    items = combination_options_header_items(
        reference_pta="nanograv_9y",
        combination_strategy="shared",
        use_pulse_numbers="yes",
        canonicalize_tim=True,
        convert_jump_mjd=True,
        alignment_policy=policy,
    )
    assert items["Product"] == "combination"
    assert items["canonicalize_tim"] is True
    assert items["alignment_policy.unsupported"] == "error"
    assert items["alignment_policy.ephem"] == "DE440"
    # Defaults that are None should be omitted from alignment flatten.
    assert "alignment_policy.clock" not in alignment_policy_header_items(policy)
