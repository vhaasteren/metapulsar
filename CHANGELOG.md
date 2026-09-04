# Changelog

All notable changes to MetaPulsar will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Package version is derived from git tags via setuptools-scm (`vX.Y.Z` → `X.Y.Z`).
This file tracks user-facing changes; it is not a substitute for the tag.

## [Unreleased]

### Logging
- `import metapulsar` now defaults loguru to `WARNING` and above (loguru's own
  default is `DEBUG`), replacing only loguru's untouched built-in handler so a
  user's own configuration wins. `metapulsar.configure_logging(level, force=,
  log_file=)` is the one knob; `sandbox_tempo2.configure_logging` wraps it.
- `MetaPulsarFactory.pta_summary` no longer removes loguru handlers and re-adds
  a `DEBUG` sink on exit, and its PINT-warning suppression is scoped to the call.
- `tests/test_nltiming_quickstart.py` runs the nltiming README quickstart
  scripts end to end in the devcontainer (skips when the checkout or engines
  are absent).

The `feat/nlt` line landing on `main`. Alpha: install from GitHub. Runtime still
pins git PINT (`vhaasteren/PINT@metapulsar`, [nanograv/PINT#2023](https://github.com/nanograv/PINT/pull/2023))
and git [nltiming](https://github.com/vhaasteren/nltiming). That is not a PyPI
v1.0 dependency set.

### Added
- Standalone **nltiming** client: MetaPulsar implements `TimingPulsar`; sampler
  protocols, engine config, and nonlinear math live in nltiming. The nltiming
  coupling is lazy — `import metapulsar` and combination must not import it.
- `metapulsar.engines`: PINT, libstempo, JUG, Vela, and composite adapters.
  Production engines arrive only via `TimingPulsar.timing_engine()`.
- `AlignmentPolicy` for shared-stack conventions (ephemeris, clock, ecliptic
  astrometry, binary-family conversion including `h3_only="sample_stigma"`).
- Canonical `.tim` writer (`canonicalize_tim=True`): flatten `INCLUDE`, bake
  `TIME`, stamp PTA metadata / `-mjd_jump_pta`, transfer `MODE`. Default is
  `False` (load each release `.tim` in place).
- File-discovery `par_precedence` / `tim_precedence` and per-pulsar overrides;
  unresolved ties raise `AmbiguousFileError`.
- Pulse-number modes as strings: `"no"` / `"yes"` / `"reuse"` / `"overwrite"`.
- `pylk.flexfit` incubation (flexible-Φ quick-look EB fits).
- InPTA DR2 layout; PPTA DR3 letter-suffix names; NANOGrav 9y `.t2` par
  precedence.

### Changed
- **Python ≥ 3.11** (was documented as 3.8+).
- Combination strategies are `"shared"` and `"per_pta"`. Legacy
  `"consistent"` / `"composite"` still work as deprecated aliases that warn.
- `MetaPulsar` requires retained `pta_files` for every PTA and reads par text
  from those files only — never PINT `as_parfile()` / libstempo `savepar()`.
- Shared `DM` is PTA-local by default (`exclude_from_shared=("DM",)`).
- Enterprise is an optional extra, not a hard dependency.

### Removed
- `metapulsar.legacy` and the dependent integration suite.

### Fixed
- Hybrid `PB+FBn` orbital charts aligned to free `FB0` before merge.
- Cross-engine FDJUMP / JUMP / TZR / MODE handling; GLS value transplant;
  zero-valued `JUMP MJD` windows dropped on conversion.

## [0.9.6] - 2026-04-01

Last tagged release before the nltiming split. Tags `v0.9.0`–`v0.9.6` were
published on GitHub; this file was not updated for those cuts.

## [0.1.0] - 2025-09-30

Historical snapshot of the first packaged implementation. Strategy names,
Python floor, Enterprise hard-dep, and the “219 tests” figure in that entry
are obsolete — see [Unreleased].

### Added
- Core `MetaPulsar` / `MetaPulsarFactory`, dual PINT and libstempo support,
  file-based creation, and the original test suite.

## [0.0.1] - 2025-09-01

### Added
- Initial project structure and package setup.
