# Release-shape fixtures

Compact par files representing the deterministic-model shapes MetaPulsar meets
in current and development PTA releases. They exist so the consistent
combination strategy can be exercised end-to-end without any local data
release; see `tests/test_release_shape_alignment.py`.

All files describe the same synthetic source so they can be combined:
an ecliptic-coordinate binary MSP at `ELONG 244.35 / ELAT -10.07`.

| File | Engine | Shape under test |
|------|--------|------------------|
| `nanograv_style.par` | pint | `SOLARN0 0`, `T2CMETHOD TEMPO`, `ECL IERS2010`, DMX bins, FD + JUMP + TZR |
| `epta_style.par` | tempo2 | all-TCB, `TIMEEPH IF99`, troposphere/planet Shapiro on, `NO_SS_SHAPIRO`, `T2` binary with `H3`+`H4` and no harmonic count |
| `ppta_style.par` | tempo2 | aggregate `TEMPO1`, incidental `DMMODEL` grid with `CONSTRAIN DMMODEL`, `ECL IERS2003` |
| `mpta_style.par` | pint | development PINT extensions: `SWM 1` + `SWX*` + `NE_SW1`, DMWaveX, WaveX, `CM`/`CMX` |
| `pint_only_a.par`, `pint_only_b.par` | pint | PINT-only multi-PTA stack with troposphere and planetary Shapiro enabled |
| `binary_ell1h_h3stig.par` | pint | ELL1H orthometric `H3`+`STIG` (absorbed-Shapiro path) |
| `binary_ddk.par` | tempo2 | DDK with `KIN`/`KOM`, for the ecliptic `KOM` transformation |

The values are synthetic and deliberately unphysical in detail; only the
*shape* of the deterministic model matters here.
