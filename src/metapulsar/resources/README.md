# MetaPulsar packaged resources

Non-code assets shipped with the `metapulsar` package (see
`[tool.setuptools.package-data]`).

## `pulsar_distances.json`

Catalog of approximate pulsar distances used by `_PtaTimingData._pdist`
materialization. Copied from Enterprise 3.x
(`enterprise/datafiles/pulsar_distances.json`) to preserve numerical
compatibility with the historical Enterprise pulsar surface.

- **Provenance:** [enterprise-pulsar](https://github.com/nanograv/enterprise)
  datafiles (MIT License)
- **Lookup policy:** see `metapulsar.pta_data.pulsar_distance` — try the
  supplied name when it begins with `J` or `B`; otherwise try `"J" + name`,
  then `"B" + name`; fall back to `(1.0, 0.2)`.
