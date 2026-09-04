#!/usr/bin/env bash
set -euo pipefail

: "${SOFTWARE_DIR:?}"

# Dockerfile sets TEMPO2=${SOFTWARE_DIR}/tempo2/T2runtime; fall back if unset.
TEMPO2="${TEMPO2:-${SOFTWARE_DIR}/tempo2/T2runtime}"
CLOCK_DST="${TEMPO2}/clock"
CLOCK_REPO="${SOFTWARE_DIR}/pulsar-clock-corrections"
CLOCK_SRC="${CLOCK_REPO}/T2runtime/clock"
TEMPO_CLOCK_SRC="${CLOCK_REPO}/tempo/clock"
PINT_CLOCK_DST="${SOFTWARE_DIR}/pint-clock-override"

mkdir -p "${CLOCK_DST}"

repo_git() {
  git -c "safe.directory=${CLOCK_REPO}" -C "${CLOCK_REPO}" "$@"
}

# Install the latest accepted IPTA clock snapshot. If a checkout is already
# present (for example in an existing image layer), refresh it and discard any
# files synthesized by older versions of this script.
if [ ! -e "${CLOCK_REPO}" ]; then
  git clone --depth=1 \
    https://github.com/ipta/pulsar-clock-corrections.git \
    "${CLOCK_REPO}"
elif [ -d "${CLOCK_REPO}/.git" ]; then
  repo_git fetch --depth=1 origin main
  repo_git reset --hard FETCH_HEAD
else
  echo "Clock repository path exists but is not a Git checkout: ${CLOCK_REPO}" >&2
  exit 1
fi

# Guard against the historical workaround that copied TT(BIPM2019) under
# newer realization names before those files were available upstream. Validate
# the source snapshot before changing the live Tempo2 clock directory.
for year in 2020 2021 2022 2023 2024 2025; do
  clock_file="${CLOCK_SRC}/tai2tt_bipm${year}.clk"
  if ! grep -Fqx "# TAI TT(BIPM${year})" "${clock_file}"; then
    echo "Incorrect or missing TT(BIPM${year}) clock: ${clock_file}" >&2
    exit 1
  fi
done

for year in 2022 2023 2024 2025; do
  if cmp -s \
    "${CLOCK_SRC}/tai2tt_bipm2019.clk" \
    "${CLOCK_SRC}/tai2tt_bipm${year}.clk"; then
    echo "TT(BIPM${year}) is unexpectedly identical to TT(BIPM2019)" >&2
    exit 1
  fi
done

# Install the updated Tempo2-format clocks where Tempo2 actually reads them.
# Merge (not replace): keep Tempo2-only files that the IPTA tree does not ship
# (e.g. coe2utc.clk, time_ao.dat), overwrite/extend everything else, and add
# new site files such as gmrt2gps.clk.
cp -a "${CLOCK_SRC}/." "${CLOCK_DST}/"

# Curated PINT override: Tempo2 .clk files + Tempo-format time_*.dat.
# Do NOT expose the Tempo2 runtime's leftover Tempo1 time_*.dat to PINT —
# those can be unordered (time_ao.dat) and diverge from ao2gps.clk/gbt2gps.clk.
rm -rf "${PINT_CLOCK_DST}"
mkdir -p "${PINT_CLOCK_DST}"
cp -a "${CLOCK_SRC}"/*.clk "${PINT_CLOCK_DST}/"
cp -a "${TEMPO_CLOCK_SRC}"/time_*.dat "${PINT_CLOCK_DST}/"
if [ -f "${CLOCK_SRC}/leap.sec" ]; then
  cp -a "${CLOCK_SRC}/leap.sec" "${PINT_CLOCK_DST}/"
elif [ -f "${TEMPO_CLOCK_SRC}/leap.sec" ]; then
  cp -a "${TEMPO_CLOCK_SRC}/leap.sec" "${PINT_CLOCK_DST}/"
fi

# Leave an auditable record of the exact clock snapshot installed in the image.
repo_git rev-parse HEAD \
  > "${SOFTWARE_DIR}/CLOCK_CORRECTIONS_REVISION"
