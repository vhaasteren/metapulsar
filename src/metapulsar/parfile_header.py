"""Shared MetaPulsar ``.par`` comment headers.

Shared / combination / GLS-optimized products should all carry the same
``# Created`` / ``# Format`` / ``# By`` block that
:func:`metapulsar.pint_helpers.dict_to_parfile_string` historically emitted,
plus optional product-specific ``# key: value`` lines (factory options,
:class:`~metapulsar.parameter_manager.AlignmentPolicy`, …).
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

_HEADER_LEAD_KEYS = ("Created", "Format", "By")


def _format_header_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_format_header_value(v) for v in value)
    if isinstance(value, Mapping):
        parts = [f"{k}={_format_header_value(v)}" for k, v in value.items()]
        return "{" + ", ".join(parts) + "}"
    text = str(value).replace("\n", " ").strip()
    return text


def format_metapulsar_par_header(
    *,
    format: str = "PINT",
    created: datetime | str | None = None,
    by: str = "MetaPulsar",
    extra: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
) -> str:
    """Return the standard MetaPulsar par comment header (trailing newline)."""
    if created is None:
        created_s = datetime.now().isoformat()
    elif isinstance(created, datetime):
        created_s = created.isoformat()
    else:
        created_s = str(created)

    lines = [
        f"# Created: {created_s}",
        f"# Format:  {format}",
        f"# By:      {by}",
    ]
    if extra:
        for key, value in extra.items():
            lines.append(f"# {key}: {_format_header_value(value)}")
    if notes:
        for note in notes:
            note_s = str(note).replace("\n", " ").strip()
            if note_s:
                lines.append(f"# {note_s}")
    return "\n".join(lines) + "\n"


def _is_standard_header_line(line: str) -> bool:
    s = line.strip()
    if not s.startswith("#"):
        return False
    body = s[1:].strip()
    for key in _HEADER_LEAD_KEYS:
        if body.startswith(f"{key}:") or body.startswith(f"{key} :"):
            return True
    return False


def strip_metapulsar_par_header(text: str) -> str:
    """Remove a leading MetaPulsar header block (and blank gap).

    Consumes the whole contiguous leading ``#`` block -- the
    ``Created``/``Format``/``By`` trio *and* the product's own
    ``# key: value`` extras (``# Product: combination``,
    ``# alignment_policy.*``, ...) -- so a rewritten product never carries the
    source product's provenance under its own header. Text whose first line is
    not one of ours is returned unchanged, so a foreign leading comment
    survives.
    """
    lines = text.splitlines()
    if not lines or not _is_standard_header_line(lines[0]):
        return text
    i = 0
    while i < len(lines) and lines[i].strip().startswith("#"):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:]) + (
        "\n" if text.endswith("\n") and i < len(lines) else ""
    )


def ensure_metapulsar_par_header(
    text: str,
    *,
    format: str = "PINT",
    created: datetime | str | None = None,
    by: str = "MetaPulsar",
    extra: Mapping[str, Any] | None = None,
    notes: Sequence[str] | None = None,
) -> str:
    """Prepend a fresh MetaPulsar header, replacing any prior leading ``#`` block."""
    body = strip_metapulsar_par_header(text)
    if body and not body.endswith("\n"):
        body += "\n"
    return (
        format_metapulsar_par_header(
            format=format, created=created, by=by, extra=extra, notes=notes
        )
        + body
    )


def alignment_policy_header_items(policy: Any | None) -> dict[str, str]:
    """Flatten an ``AlignmentPolicy`` (or similar dataclass) for header extras."""
    if policy is None:
        return {}
    if is_dataclass(policy) and not isinstance(policy, type):
        raw = asdict(policy)
    elif isinstance(policy, Mapping):
        raw = dict(policy)
    else:
        return {"alignment_policy": repr(policy)}
    out: dict[str, str] = {}
    for key, value in raw.items():
        if value is None:
            continue
        out[f"alignment_policy.{key}"] = _format_header_value(value)
    return out


def combination_options_header_items(
    *,
    reference_pta: str,
    combination_strategy: str = "shared",
    use_pulse_numbers: str | None = None,
    canonicalize_tim: bool | None = None,
    convert_jump_mjd: bool | None = None,
    exclude_from_shared: Sequence[str] | None = None,
    combine_components: Sequence[str] | None = None,
    add_dm_derivatives: bool | None = None,
    alignment_policy: Any | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """User-facing combination options for a combination-product par header."""
    items: dict[str, Any] = {
        "Product": "combination",
        "combination_strategy": combination_strategy,
        "reference_pta": reference_pta,
    }
    if use_pulse_numbers is not None:
        items["use_pulse_numbers"] = use_pulse_numbers
    if canonicalize_tim is not None:
        items["canonicalize_tim"] = canonicalize_tim
    if convert_jump_mjd is not None:
        items["convert_jump_mjd"] = convert_jump_mjd
    if exclude_from_shared is not None:
        items["exclude_from_shared"] = list(exclude_from_shared)
    if combine_components is not None:
        items["combine_components"] = list(combine_components)
    if add_dm_derivatives is not None:
        items["add_dm_derivatives"] = add_dm_derivatives
    items.update(alignment_policy_header_items(alignment_policy))
    if extra:
        items.update(extra)
    return items
