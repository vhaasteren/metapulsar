# Pulse Tracking Fixtures

These fixtures support `tests/test_pulse_tracking.py`.

They encode a deterministic synthetic pulsar scenario with two individually
coherent PTA-like datasets:

- `epta_like.par` / `epta_like.tim`
  - 100 observing epochs
  - 3 frequency channels per epoch
  - coherent with a `DM`, `DM1`, `DM2` timing model
- `nanograv_like.par` / `nanograv_like.tim`
  - 20 observing epochs
  - 15 frequency channels per epoch
  - coherent with a DMX model containing one bin per epoch

## Scientific Intent

The synthetic pulsar is deliberately configured so that dispersion evolution
matters over a 10-year span:

- `F0 = 300 Hz`
- `DM = 120 pc cm^-3`
- `DM1 = 0.1 pc cm^-3 / yr`
- `DM2 = 0.05 pc cm^-3 / yr^2`
- observing band spanning `300-1600 MHz`

When `MetaPulsarFactory.create_metapulsar(..., combination_strategy="shared")`
processes these datasets, the consistency path removes DMX and resets
`DM1`/`DM2`, producing a wrong pre-fit timing model. A weighted linear solve on
the merged design matrix behaves very differently depending on whether pulse
numbers are preserved from the original coherent solutions:

- with pulse-number tracking: the fit recovers a coherent post-fit solution
- without pulse-number tracking: the fit remains incoherent at roughly
  pulse-period-scale residuals

The regression test uses absolute post-fit RMS thresholds rather than
fixture-dependent ratios:

- coherent solution: `< 1e-6 s`
- incoherent solution: `> 1e-5 s`

These values are intentionally tied to physically meaningful scales:
`1e-6 s` is still far below any phase-wrap-scale failure while allowing for
environment-dependent effects such as clock-correction differences, and
`1e-5 s` is still well below the pulse period for this synthetic pulsar but
firmly in the phase-wrap regime the test is designed to expose.

## Important Fixture Rule

The checked-in `.tim` files intentionally do **not** contain `-pn` flags.
`MetaPulsar` must derive pulse numbers through its production helpers from the
original coherent `par + tim` pair at test time.

## Regeneration

From the repository root:

```bash
python tests/fixtures/pulse_tracking/generate_pulse_tracking_fixtures.py --force
```

The generator writes all four fixture files and verifies that each individual
par/tim pair remains coherent with a very small pre-fit RMS.
