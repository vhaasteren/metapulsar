#!/usr/bin/env bash
# Build a curated PINT_CLOCK_OVERRIDE from the IPTA clock snapshot already
# present in the anpta image. Prefer $SOFTWARE_DIR/pint-clock-override when the
# image provides it; otherwise populate $HOME/pint-clock-override.
set -euo pipefail

SOFTWARE_DIR="${SOFTWARE_DIR:-/opt/software}"
CLOCK_REPO="${SOFTWARE_DIR}/pulsar-clock-corrections"
CLOCK_SRC="${CLOCK_REPO}/T2runtime/clock"
TEMPO_CLOCK_SRC="${CLOCK_REPO}/tempo/clock"
IMAGE_DST="${SOFTWARE_DIR}/pint-clock-override"
HOME_DST="${HOME}/pint-clock-override"

if [ -d "${IMAGE_DST}" ] && [ -f "${IMAGE_DST}/time_ao.dat" ] && [ -f "${IMAGE_DST}/ao2gps.clk" ]; then
  echo "Using image PINT clock override: ${IMAGE_DST}"
  exit 0
fi

if [ ! -d "${CLOCK_SRC}" ] || [ ! -d "${TEMPO_CLOCK_SRC}" ]; then
  echo "IPTA clock sources not found under ${CLOCK_REPO}; skipping" >&2
  exit 0
fi

DEST="${HOME_DST}"
rm -rf "${DEST}"
mkdir -p "${DEST}"
cp -a "${CLOCK_SRC}"/*.clk "${DEST}/"
cp -a "${TEMPO_CLOCK_SRC}"/time_*.dat "${DEST}/"
if [ -f "${CLOCK_SRC}/leap.sec" ]; then
  cp -a "${CLOCK_SRC}/leap.sec" "${DEST}/"
elif [ -f "${TEMPO_CLOCK_SRC}/leap.sec" ]; then
  cp -a "${TEMPO_CLOCK_SRC}/leap.sec" "${DEST}/"
fi
echo "Built curated PINT clock override at ${DEST} ($(ls "${DEST}" | wc -l) files)"
