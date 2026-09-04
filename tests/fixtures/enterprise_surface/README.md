# Enterprise surface golden fixtures

Migration oracle for MetaPulsar-owned PTA materializers
(`metapulsar.pta_data`). Generated against Enterprise while it is still
installed; ordinary tests only *read* the committed NPZ/JSON files.

## Contents

| File | Source |
|---|---|
| `pint_equatorial.npz` | `sample_parfiles/simple.par` + `simple.tim` via `PintPulsar` |
| `pint_ecliptic.npz` | `pulse_tracking/nanograv_like.par` + `.tim` via `PintPulsar` |
| `tempo2_mock_equatorial.npz` | `create_mock_libstempo(..., seed=10)` via `Tempo2Pulsar` |
| `metapulsar_tempo2_pair.npz` | Two mock tempo2 legs through `MetaPulsar` public surface |
| `manifest.json` | Enterprise and scientific-stack versions |

## Regeneration

Requires the optional Enterprise environment:

```bash
python tests/fixtures/enterprise_surface/generate_enterprise_surface.py --overwrite
```

Refuse to overwrite without `--overwrite`.
