# metapulsar.py
"""Class containing pulsar data from multiple data releases"""

from io import StringIO
from pathlib import Path
import subprocess
import tempfile

import copy

import logging
import os
import glob
import re
import shutil

from itertools import groupby
from collections import defaultdict

import numpy as np

from pint.models.model_builder import parse_parfile, ModelBuilder
from pint.logging import setup as setup_log

from metapulsar.pint_helpers import detect_astrometry_style

setup_log(level="WARNING")

import enterprise.pulsar as ep
import h5pulsar as h5p

from astropy.coordinates import SkyCoord
from astropy.coordinates import FK5

logger = logging.getLogger(__name__)

try:
    import libstempo as t2
except ImportError:
    logger.warning(
        "libstempo not installed. Will use PINT instead."
    )  # pragma: no cover
    t2 = None

# Import the new sandbox tempo2 implementation
try:
    from ..sandbox_tempo2 import tempopulsar as sandbox_tempopulsar
except ImportError as e:
    logger.warning(f"Sandbox tempo2 not available: {e}. Will use original libstempo.")
    sandbox_tempopulsar = None

try:
    import pint
    from pint.models import TimingModel, get_model_and_toas
    from pint.residuals import Residuals as resids
    from pint.toa import TOAs
except ImportError:
    logger.warning(
        "PINT not installed. Will use libstempo instead."
    )  # pragma: no cover
    pint = None

if pint is None and t2 is None:
    err_msg = "Must have either PINT or libstempo timing package installed"
    raise ImportError(err_msg)


# # tempo2 units
# import astropy.constants
import astropy.units as u

#
# lts = astropy.units.def_unit(['lightsecond','ls','lts'],astropy.constants.c * u.s)
#
# map_units = {
#                 'F0': u.Hz,'F1': u.Hz/u.s,'F2': u.Hz/u.s**2,
#                 'RAJ': u.rad,'DECJ': u.rad,'ELONG': u.rad,'ELAT': u.rad,
#                 'PMRA': u.mas / u.yr,'PMDEC': u.mas / u.yr,'PMELONG': u.mas / u.yr,'PMELAT': u.mas / u.yr,
#                 'PX': u.mas,
#                 'PB': u.d,'ECC': u.dimensionless_unscaled,'A1': lts,'OM': u.deg,
#                 'EPS1': u.dimensionless_unscaled,'EPS2': u.dimensionless_unscaled,
#                 # KOM, KIN?
#                 'SHAPMAX': u.dimensionless_unscaled,'OMDOT': u.deg/u.yr,
#                 # PBDOT?
#                 'ECCDOT': 1/u.s,'A1DOT': lts/u.s,'GAMMA': u.s,
#                 # XPBDOT?
#                 # EPS1DOT, EPS2DOT?
#                 'MTOT': u.Msun,'M2': u.Msun,
#                 # DTHETA, XOMDOT
#                 'SIN1': u.dimensionless_unscaled,
#                 # DR, A0, B0, BP, BPP, AFAC
#                 'DM': u.cm**-3 * u.pc,'DM1': u.cm**-3 * u.pc * u.yr**-1, # how many should we do?
#                 'POSEPOCH': u.day,'T0': u.day,'TASC': u.day
#                 }
#
# map_times = ['POSEPOCH','T0','TASC']


# How many do we need? Or make a template perhaps?
spin_parameters = (
    ["PEPOCH"] + [f"F{ii}" for ii in range(20)] + [f"P{ii}" for ii in range(20)]
)

astrometry_parameters = ["POSEPOCH"] + [
    "RAJ",
    "DECJ",
    "PMRA",
    "PMDEC",
    "LAMBDA",
    "BETA",
    "PMLAMBDA",
    "PMBETA",
    "ELONG",
    "ELAT",
    "PMELONG",
    "PMELAT",
    "PMELONG2",
    "PMLAMBDA2",
    "PMELAT2",
    "PMBETA2",
    "PX",
]

# NOTE: use PINT to get binary parameters, this is not exhaustive
binary_parameters = ["BINARY"] + [
    "PB",
    "PBDOT",
    "A1",
    "A1DOT",
    "XDOT",
    "OM",
    "T0",
    "OMDOT",
    "TASC",
    "GAMMA",
    "ECC",
    "E",
    "ECCDOT",
    "EDOT",
    "KOM",
    "KIN",
    "SINI",
    "SHAPMAX",
    "H4",
    "H3",
    "STIG",
    "STIGMA",
    "EPS1",
    "EPS2",
    "EPS1DOT",
    "EPS2DOT",
    "DR",
    "DTH",
    "SHAPIRO",
    "M2",
    "A0",
    "B0",
    "OM2DOT",
    "FB0",
    "FB1",
    "FB2",
    "FB3",
    "FB4",
    "FB5",
    "FB6",
    "FB7",
    "FB8",
    "FB9",
    "FB10",
    "FB11",
    "FB12",
    "FB13",
    "FB14",
    "FB15",
    "FB16",
    "FB17",
    "A1_2",
    "A1_3",
    "ECC_2",
    "ECC_3",
    "T0_2",
    "T0_3",
    "OM_2",
    "OM_3",
    "VARSIGMA",
]

dm_parameters = ["DMEPOCH", "DM", "DM1", "DM2"]

# Other parameters in the parfile that need to match
init_parameters = ["EPHEM", "CLOCK", "CLK"]

# Init parameters with aliases need to be mapped
map_init_parameters = {"CLOCK": "CLK"}

#                     Tempo2: PINT
parameter_aliases = {
    "XDOT": "A1DOT",
    "E": "ECC",
    "EDOT": "ECCDOT",
    "STIG": "STIGMA",
    "VARSIGMA": "STIGMA",
}
#                         PINT: Tempo2
parameter_rev_aliases = {
    "A1DOT": "XDOT",
    "ECC": "E",
    "ECCDOT": "EDOT",
    "STIGMA": ["STIG", "VARSIGMA"],
}

equivalence_parameter_lists = [
    ["RAJ", "ELONG", "LAMBDA"],
    ["DECJ", "ELAT", "BETA"],
    ["PMRA", "PMELONG", "PMLAMBDA"],
    ["PMDEC", "PMELAT", "PMBETA"],
    ["XDOT", "A1DOT"],
    ["VARSIGMA", "STIGMA"],
    ["E", "ECC"],
    ["EDOT", "ECCDOT"],
]

epoch_map = {
    "J1857+0943": "B1855+09",
    "J1939+2134": "B1937+21",
    "J1955+2908": "B1953+29",
}


def all_equal(iterable):
    psr_list = [epoch_map.get(x, x) for x in iterable]
    g = groupby(psr_list)
    return next(g, True) and not next(g, False)


def get_pta_release_files(base_dir, par_pattern, tim_pattern, **kwargs):
    """For a specific PTA data release, obtain the par/tim files

    :param base_dir:    The base directory of the data release
    :par_pattern:       Regular expression that matches the par files
    :tim_pattern:       Regular expression that matches the tim files

    :returns:           Dictionary: {pulsar_name: (parfile,timfile), ...}

    Example
    -------

    # NANOGrav_12yr (exclude the .t2 solution for J1713)
    ipta_data_releases_dir = 'some_directo'
    get_pta_release_files(
        base_dir = f"{ipta_data_releases_dir}/NANOGrav_12y/",
        par_pattern = r"par/([BJ]\d{4}[+-]\d{2,4})(?!.*\.t2)_NANOGrav_12yv2\.gls\.par",
        tim_pattern = r"tim/([BJ]\d{4}[+-]\d{2,4})_NANOGrav_12yv2\.tim",
    )

    """
    pulsar_files = {}

    def list_files(directory, pattern):
        files = glob.glob(f"{directory}/**/*", recursive=True)
        regex = re.compile(f"{pattern}")
        return [f for f in files if os.path.isfile(f) and regex.search(f)]

    # List .par and .tim files based on the patterns
    par_files = list_files(base_dir, par_pattern)
    tim_files = list_files(base_dir, tim_pattern)

    # Match .par and .tim files
    for par_file in par_files:
        pulsar_name_match = re.search(
            par_pattern, par_file
        )  # Match against the full path
        if pulsar_name_match:
            pulsar_name = pulsar_name_match.group(
                1
            )  # 0 would be all, 1 is just PSR name

            # Find the corresponding .tim file that contains the pulsar_name
            tim_file = next(
                (
                    f
                    for f in tim_files
                    if pulsar_name in f and re.search(tim_pattern, f)
                ),
                None,
            )

            if tim_file:
                pulsar_files[pulsar_name] = (par_file, tim_file)

    return pulsar_files


# Can select on multiple flags -- staggered (IPTA-like)
# "-group, None" (all for that flag)
# "-pta PPTA" (only that flag)
# "(group, f), None (all for group, if no group, all for f)"
def create_selection_stag(name, flagdict={}, lowfreq=None, highfreq=None):

    def get_flagvals(flags, flag, flagval):
        if flagval is not None:
            return [flagval]
        else:
            # Never use empty flag values
            return set(flags[flag]) - set([""])

    def get_flagit(flags, flag, flagval):
        """get_flagit(flags, ("group", "f"), None) -> {flagval_list: masks}"""
        try:
            if isinstance(flag, str):
                raise TypeError
            flagkeys = [f for f in flag]
        except TypeError:
            flagkeys = [flag]

        dummy_key = list(flags.keys())[0]
        msk_todo = np.ones_like(flags[dummy_key], dtype=bool)
        rd = {}

        for flagkey in flagkeys:
            try:
                flagvals = get_flagvals(flags, flagkey, flagval)
                for fv in flagvals:
                    msk_flag = flags[flagkey] == fv

                    if np.any(np.logical_and(msk_todo, msk_flag)):
                        # Yep, add this flag-key
                        rd.update({fv: msk_flag})

                        msk_todo = np.logical_and(msk_todo, np.logical_not(msk_flag))
                    else:
                        # Nothing to add
                        pass

            except KeyError:
                pass

        return rd

    def selection_func(flags, freqs):
        if lowfreq is not None and highfreq is not None:
            freq_mask = np.logical_and(
                freqs >= float(lowfreq), freqs <= float(highfreq)
            )
            suffix = f"_B_{lowfreq}_{highfreq}"
        else:
            freq_mask = np.ones_like(freqs, dtype=bool)
            suffix = ""

        sels = [
            get_flagit(flags, flag, flagval) for (flag, flagval) in flagdict.items()
        ]

        if len(sels) > 0:
            return {
                name + suffix + "_" + fv: msk for rd in sels for (fv, msk) in rd.items()
            }
        else:
            return {name + suffix: freq_mask}

    return selection_func


def get_timing_package(psr):
    if isinstance(psr, ep.PintPulsar):
        return "PINT"
    elif isinstance(psr, ep.Tempo2Pulsar):
        return "Tempo2"
    else:
        raise ValueError(f"Backend package for {psr.name} not supported")


def check_in_fitpars(parname, epulsars):
    """Check whether parname is fit for in all epulsars"""

    def has_fitpar(parname, epsr):
        len_ali = len(parameter_aliases.get(parname, parname))
        len_rev = len(parameter_rev_aliases.get(parname, parname))
        return any(
            np.atleast_1d(
                parameter_rev_aliases.get(parname, parname)[i] in epsr.fitpars
                for i in range(len_rev)
            )
        ) or any(
            np.atleast_1d(
                parameter_aliases.get(parname, parname)[i] in epsr.fitpars
                for i in range(len_ali)
            )
        )

    return all([has_fitpar(parname, epsr) for epsr in epulsars.values()])


def file_to_stringio(fpath):
    """Read from file path (can be str, Path, or StringIO)"""

    # If the value is a string or a Path object, convert it to a StringIO object
    if isinstance(fpath, (str, Path)):
        with open(Path(fpath), "r") as f:
            return StringIO(f.read())

    # If the fpath is a file object
    elif hasattr(fpath, "read"):
        # Read the content and reset the file pointer
        fpath.seek(0)
        content = fpath.read()
        fpath.seek(0)
        return StringIO(content)

    # Raise an exception for unsupported types
    else:
        raise ValueError(f"Unsupported fpath type: {type(fpath)}")


def write_stringio_to_file(string_io_object, fpobj):
    # Ensure that the string_io_object's pointer is at the beginning
    string_io_object.seek(0)
    content = string_io_object.getvalue()

    # Check if fpobj is a pathlib.Path object or a string representing a file path
    if isinstance(fpobj, (Path, str)):
        with open(fpobj, "w") as file:
            file.write(content)
    else:
        fpobj.write(content)
        # Flush the write buffer to ensure all data is written to the file
        fpobj.flush()


class MetaParfiles(object):
    """Class to manipulate multiple parfiles for combined analysis"""

    _EQUATORIAL_WARNING = (
        "Equatorial astrometry detected. PINT/tempo2 agreement after T2CMETHOD "
        "modification is typically a few ns (about 6 ns on NG5 J1600), "
        "slightly larger than ecliptic pars with full convention alignment "
        "(about 1 ns on NG11 J1600). Reason: no explicit ECL obliquity "
        "convention to align; residual differences from ecliptic-frame "
        "geometry entering delay terms may remain. For best agreement, prefer "
        "ecliptic coordinates in new timing solutions."
    )

    def __init__(
        self,
        parfiles,
        merge_astrometry,
        merge_spin,
        merge_binary,
        merge_dm,
        ref_index=0,
        convert=True,
    ):
        """Parse parfiles and modify them for combined analysis"""
        # List of dictionaries: [{pta: ptaname, parfile: filepath, package: pint}]

        # Check input, and read the original par files
        self._parfiles = copy.deepcopy(parfiles)
        self.check_and_read_input()

        # Read parfiles using PINT routine 'parse_parfile'
        self.read_parfile_dicts(parfield="parfile", pardictfield="pardict")

        # set ref index
        self.ref_index = ref_index

        if convert:
            self.convert_all(merge_astrometry, merge_spin, merge_binary, merge_dm)

    def convert_all(self, merge_astrometry, merge_spin, merge_binary, merge_dm):
        """Do all the conversions, and save to pardict_conv"""

        # Conversion, so that all UNITS are the same
        self.convert_units()

        # Read converted parfiles using PINT routine 'parse_parfile'
        self.read_parfile_dicts(
            parfield="parfile_units_converted", pardictfield="pardict_conv"
        )

        # Initialize all the init parameters
        self.initialize()

        # Merge all models
        self.merge_init()
        if merge_spin:
            self.merge_spin()

        if merge_astrometry:
            self.merge_astrometry()

        if merge_binary:
            self.merge_binary()

        if merge_dm:
            self.merge_dm()

        # Align CLOCK/EPHEM conventions for PINT/tempo2 agreement
        self.unify_clock_ephem()

        # Adjustment for PINT or Tempo2
        self.summary()

    def check_and_read_input(self):
        """Check the given input, and read parfiles from disk if necessary"""

        for pfd in self._parfiles:
            if not isinstance(pfd["pta"], str):
                logger.error("PTA type needs to be a string")  # pragma: no cover
                raise ValueError("PTA type needs to be a string")

            if pfd["package"].lower() not in ["libstempo", "tempo2", "pint"]:
                logger.error(
                    "Package needs to be tempo2/libstempo/pint"
                )  # pragma: no cover
                raise ValueError("Package needs to be tempo2/libstempo/pint")
            else:
                pfd["package"] = pfd["package"].lower()

            if isinstance(pfd["parfile"], str) and not os.path.isfile(pfd["parfile"]):
                logger.error(
                    f"Parfile {pfd['parfile']} does not exist"
                )  # pragma: no cover
                raise FileExistsError(f"Parfile {pfd['parfile']} does not exist")

            elif isinstance(pfd["parfile"], (str, Path)) or hasattr(
                pfd["parfile"], "read"
            ):
                pfd["parfile"] = file_to_stringio(pfd["parfile"])

            else:
                logger.error(f"Parfile {pfd['parfile']} invalid")  # pragma: no cover
                raise ValueError(f"Parfile {pfd['parfile']} invalid")

    def read_parfile_dicts(self, parfield="parfile", pardictfield="pardict"):
        """Read in all the parfiles"""

        for pfd in self._parfiles:
            pfd[pardictfield] = parse_parfile(pfd[parfield])
            # pfd['new_pardict'] = copy.deepcopy(pfd[pardictfield])   # Why is this here??

    def initialize(self):
        """Initialize the basic setup of the parfiles"""
        # Goal is to get new_par_dict right
        # This is not being run just yet

        for pfd in self._parfiles:
            pd = pfd["pardict_conv"]
            # npd = pfd['new_pardict']

            if "PSR" in pd:
                pfd["name"] = pd["PSR"][0]
            elif "PSRB" in pd:
                pfd["name"] = pd.pop("PSRB")[0]
            elif "PSRJ" in pd:
                pfd["name"] = pd.pop("PSRJ")[0]
            else:
                print(pd)
                logger.error(
                    f"Parfile doesn't have a pulsar name: {pfd['parfile']} "
                )  # pragma: no cover
                raise ValueError("Parfile doesn't have a pulsar name")

            # New parfile just uses the 'PSR' ID for pulsar name
            pd["PSR"] = [pfd["name"]]

            if "EPHEM" in pd:
                pfd["EPHEM"] = pd["EPHEM"][0]
                # pd['EPHEM'] = [pd['EPHEM'][0]]
            else:
                logger.warning(
                    f"No Ephemeris set for {pfd['name']}, setting DE440"
                )  # pragma: no cover
                pfd["EPHEM"] = "DE440"
                pd["EPHEM"] = ["DE440"]

            if "DILATEFREQ" in pd:
                # pd['DILATEFREQ'] = [pd['DILATEFREQ'][0]]
                pass
            elif pfd["package"] in ["tempo2", "libstempo"]:
                pd["DILATEFREQ"] = ["Y"]

            if "DM_SERIES" in pd:
                if pd["DM_SERIES"][0].upper() != "TAYLOR":
                    logger.error("DM_SERIES not set to TAYLOR")

                pd["DM_SERIES"] = ["TAYLOR"]

            elif pfd["package"] in ["tempo2", "libstempo"]:
                pd["DM_SERIES"] = ["TAYLOR"]

    def summary(self, parfile="./test.par"):
        """Adjustment for PINT or Tempo2
        params:
        ------
        parfile: default False, the parfile path
        """
        for ind, pfd in enumerate(self._parfiles):
            pd = pfd["pardict_conv"]
            if ind != self.ref_index and pfd["package"] == "pint" and "STIG" in pd:
                self.replace_parameters_from_par(
                    pop_par="STIG", re_par="STIGMA", index=ind
                )

    #        print("========================")
    #        print(pfd)
    #        print("========================")
    #        mb = ModelBuilder()
    #        model = mb(pd)
    #
    #        if parfile:
    #            model.write_parfile(parfile)

    def convert_pint_to_tdb(self, pfd):
        """Create a StringIO object with the new parfile using PINT
        Usually, we can't produce pulsar in tcb units with PINT
        """

        mb = ModelBuilder()
        model = mb(StringIO(pfd["parfile"].getvalue()), allow_tcb=True)

        new_file = StringIO("")
        model.write_parfile(new_file)

        pfd["parfile_units_converted"] = new_file
        pfd["parfile_units_converted"].seek(0)

    def convert_tempo2_to_tdb(self, pfd):
        """Create a StringIO object with the new parfile using Tempo2
        params
        ------
        pfd:list, the pta list {'parfile':xxxx, 'timfile':xxxx,}
        """
        with (
            tempfile.NamedTemporaryFile(mode="w+", delete=True) as output_file,
            tempfile.NamedTemporaryFile(mode="w+", delete=True) as input_file,
        ):

            write_stringio_to_file(pfd["parfile"], input_file.name)

            # Usage tempo2 -gr transform parFile outputFile [back|tdb]
            subprocess.run(
                [
                    "tempo2",
                    "-gr",
                    "transform",
                    input_file.name,
                    output_file.name,
                    "tdb",
                ],
                check=True,
            )

            output_file.seek(0)  # Go to the beginning of the file
            pfd["parfile_units_converted"] = StringIO(output_file.read())
            pfd["parfile_units_converted"].seek(0)

    def convert_units(self, orig_field="pardict", new_field="pardict_conv"):
        """Convert TCB to TDB if necessary"""

        for pfd in self._parfiles:
            pd = pfd["pardict"]

            if "UNITS" in pd:
                pfd["UNITS"] = pd["UNITS"][0]
                pfd["default_units"] = "no"
            elif pfd["package"] in ["tempo2", "libstempo"]:
                # logger.warning(f"Setting UNITS to TCB for {pfd['name']}-{pfd['pta']}") # pragma: no cover
                #                if pfd['EPHVER'] == "2":
                #                    EPHVER_t = "TDB"
                #                else:
                #                    EPHVER_t = "TCB"

                # logger.warning(f"Setting UNITS to {EPHVER_t} for {pfd['pta']}") # pragma: no cover
                #                pfd['UNITS'] = EPHVER_t
                #                pd['UNITS'] = [EPHVER_t]
                #                pfd['default_units'] = 'tempo2'
                logger.warning(
                    f"Setting UNITS to TCB for {pfd['pta']}"
                )  # pragma: no cover
                pfd["UNITS"] = "TCB"
                pd["UNITS"] = ["TCB"]
                pfd["default_units"] = "tempo2"
            elif pfd["package"] == "pint":
                # logger.warning(f"Setting UNITS to TDB for {pfd['name']}-{pfd['pta']}") # pragma: no cover
                logger.warning(
                    f"Setting UNITS to TDB for {pfd['pta']}"
                )  # pragma: no cover
                pfd["UNITS"] = "TDB"
                pd["UNITS"] = ["TDB"]
                pfd["default_units"] = "pint"

        all_units = [pfd["UNITS"] for pfd in self._parfiles]

        if all_equal(all_units):
            for pfd in self._parfiles:
                pfd["parfile"].seek(0)
                pfd["parfile_units_converted"] = StringIO(pfd["parfile"].read())
                pfd["parfile"].seek(0)

            return

        for pfd in self._parfiles:
            if pfd["package"] == "pint" and pfd["UNITS"] == "TCB":
                # Use PINT to convert
                self.convert_pint_to_tdb(pfd)

            elif pfd["package"] == "tempo2" and pfd["UNITS"] == "TCB":
                # Use Tempo2 to convert
                self.convert_tempo2_to_tdb(pfd)

            elif pfd["UNITS"] != "TDB":
                logger.error(
                    f"Cannot convert units for {pfd['package']}--{pfd['UNITS']}"
                )
                raise NotImplementedError(
                    f"Cannot convert units for {pfd['package']}--{pfd['UNITS']}"
                )

            else:
                pfd["parfile"].seek(0)
                pfd["parfile_units_converted"] = StringIO(pfd["parfile"].read())
                pfd["parfile"].seek(0)

    def _is_t2cmethod_tempo(self, pd):
        if "T2CMETHOD" not in pd or not pd["T2CMETHOD"]:
            return False
        parts = pd["T2CMETHOD"][0].split()
        if not parts:
            return False
        return parts[0].upper() == "TEMPO"

    def _normalize_timing_package(self, package):
        normalized = (package or "pint").strip().lower()
        if normalized == "libstempo":
            return "tempo2"
        return normalized

    def _normalized_timing_packages(self):
        return {
            self._normalize_timing_package(pfd["package"]) for pfd in self._parfiles
        }

    def _is_cross_engine_mix(self, normalized_packages):
        return {"pint", "tempo2"}.issubset(normalized_packages)

    def _parse_ecl_value(self, pd):
        if "ECL" not in pd or not pd["ECL"]:
            return None
        parts = pd["ECL"][0].split()
        if not parts:
            return None
        return parts[0].upper()

    def _parse_t2cmethod_value(self, pd):
        if "T2CMETHOD" not in pd or not pd["T2CMETHOD"]:
            return None
        parts = pd["T2CMETHOD"][0].split()
        if not parts:
            return None
        return parts[0].upper()

    def _collect_convention_states(self):
        states = {}
        for pfd in self._parfiles:
            pd = pfd["pardict_conv"]
            states[pfd["pta"]] = {
                "style": detect_astrometry_style(pd),
                "ecl": self._parse_ecl_value(pd),
                "t2cmethod": self._parse_t2cmethod_value(pd),
                "package": self._normalize_timing_package(pfd["package"]),
            }
        return states

    def unify_clock_ephem(self):
        """Apply gated CLOCK/EPHEM convention alignment in legacy conversion flow."""
        ref_pd = self._parfiles[self.ref_index]["pardict_conv"]
        if "EPHEM" not in ref_pd:
            raise ValueError(
                "Reference parfile is missing EPHEM for convention alignment"
            )
        ref_clock_key = None
        for key in ("CLOCK", "CLK"):
            if key in ref_pd:
                ref_clock_key = key
                break
        if ref_clock_key is None:
            raise ValueError(
                "Reference parfile is missing CLOCK/CLK for convention alignment"
            )
        ref_ephem = ref_pd["EPHEM"]
        ref_clock = ref_pd[ref_clock_key]

        for pfd in self._parfiles:
            pd = pfd["pardict_conv"]
            pd["EPHEM"] = ref_ephem
            pd.pop("CLOCK", None)
            pd.pop("CLK", None)
            pd[ref_clock_key] = ref_clock

        convention_states = self._collect_convention_states()
        normalized_packages = self._normalized_timing_packages()
        cross_engine = self._is_cross_engine_mix(normalized_packages)

        if cross_engine:
            for pfd in self._parfiles:
                pd = pfd["pardict_conv"]
                pta = pfd["pta"]
                actions = [
                    f"aligned EPHEM={ref_ephem[0]}",
                    f"aligned {ref_clock_key}={ref_clock[0]}",
                ]
                style = convention_states[pta]["style"]

                if style == "ecliptic":
                    old_ecl = pd.get("ECL", ["(missing)"])[0]
                    pd["ECL"] = ["IERS2003"]
                    actions.append(f"set ECL=IERS2003 (was {old_ecl})")
                else:
                    if "ECL" in pd:
                        old_ecl = pd.pop("ECL")
                        actions.append(f"removed ECL ({old_ecl[0]})")
                    logger.warning(f"[{pta}] {self._EQUATORIAL_WARNING}")
                    actions.append("emitted equatorial warning")

                if self._is_t2cmethod_tempo(pd):
                    old_t2c = pd.pop("T2CMETHOD")
                    actions.append(f"removed T2CMETHOD ({old_t2c[0]})")

                logger.info(
                    f"PTA {pta}: convention alignment applied "
                    f"(reason=cross_engine_agreement, style={style}); "
                    f"actions: {', '.join(actions)}"
                )
            return

        if len(self._parfiles) == 1:
            logger.info(
                "Convention alignment skipped (reason=single_pta_single_engine)"
            )
            return

        if normalized_packages == {"pint"}:
            ecliptic_ptas = [
                pfd["pta"]
                for pfd in self._parfiles
                if convention_states[pfd["pta"]]["style"] == "ecliptic"
            ]
            ecliptic_ecl_values = {
                convention_states[pta]["ecl"] for pta in ecliptic_ptas
            }
            if len(ecliptic_ecl_values) <= 1:
                logger.info(
                    "Convention alignment skipped (reason=homogeneous_ecl_single_engine_pint)"
                )
                return

            reference_ecl = self._parse_ecl_value(ref_pd)
            target_ecl = reference_ecl or "IERS2010"
            for pfd in self._parfiles:
                if pfd["pta"] not in ecliptic_ptas:
                    continue
                pd = pfd["pardict_conv"]
                old_ecl = pd.get("ECL", ["(missing)"])[0]
                pd["ECL"] = [target_ecl]
                logger.info(
                    f"PTA {pfd['pta']}: convention alignment applied "
                    f"(reason=pint_only_ecl_heterogeneous); set ECL={target_ecl} (was {old_ecl})"
                )
            return

        if normalized_packages == {"tempo2"}:
            ecliptic_ptas = [
                pfd["pta"]
                for pfd in self._parfiles
                if convention_states[pfd["pta"]]["style"] == "ecliptic"
            ]
            ecliptic_ecl_values = {
                convention_states[pta]["ecl"] for pta in ecliptic_ptas
            }
            if len(ecliptic_ecl_values) > 1:
                for pfd in self._parfiles:
                    if pfd["pta"] not in ecliptic_ptas:
                        continue
                    pd = pfd["pardict_conv"]
                    old_ecl = pd.get("ECL", ["(missing)"])[0]
                    pd["ECL"] = ["IERS2003"]
                    logger.info(
                        f"PTA {pfd['pta']}: convention alignment applied "
                        "(reason=tempo2_only_ecl_heterogeneous); "
                        f"set ECL=IERS2003 (was {old_ecl})"
                    )
            else:
                logger.info(
                    "Convention alignment skipped (reason=homogeneous_ecl_single_engine_tempo2)"
                )

            t2_values = {
                convention_states[pfd["pta"]]["t2cmethod"] for pfd in self._parfiles
            }
            if len(t2_values) > 1:
                if "T2CMETHOD" not in ref_pd:
                    logger.info(
                        "T2CMETHOD alignment skipped (reason=reference_missing_t2cmethod)"
                    )
                else:
                    ref_t2 = ref_pd["T2CMETHOD"]
                    for pfd in self._parfiles:
                        pd = pfd["pardict_conv"]
                        pd["T2CMETHOD"] = ref_t2
                        logger.info(
                            f"PTA {pfd['pta']}: convention alignment applied "
                            "(reason=tempo2_only_t2cmethod_heterogeneous); "
                            f"set T2CMETHOD={ref_t2[0]}"
                        )
            else:
                logger.info(
                    "T2CMETHOD alignment skipped (reason=homogeneous_t2cmethod_single_engine_tempo2)"
                )
            return

        logger.info(
            "Convention alignment skipped "
            f"(reason=unsupported_single_engine_packages:{sorted(normalized_packages)})"
        )

    def replace_pars_with_ref_model(self, ref_index=0, parameters=[], map={}):
        """Based on the reference model, replace all relevant parameters"""

        refpfd = self._parfiles[ref_index]["pardict_conv"]
        dupdate = {
            map.get(par, par): parval
            for (par, parval) in refpfd.items()
            if par in parameters
        }

        for ind, pfd in enumerate(self._parfiles):
            if ind != ref_index:
                pd = pfd["pardict_conv"]

                for parname in set(parameters) & set(pd.keys()):
                    pd.pop(parname)

                pd.update(dupdate)

    # Could be better
    def pop_parameters_from_par(self, pop_par, index=0):
        """Delete parameters in parfile"""

        for ind, pfd in enumerate(self._parfiles):
            if ind == index:
                pd = pfd["pardict_conv"]
                pd.pop(pop_par)

    # Could be better
    def replace_parameters_from_par(self, pop_par, re_par, index=0):
        """Replace parameters in parfile"""
        for ind, pfd in enumerate(self._parfiles):
            if ind == index:
                pd = pfd["pardict_conv"]
                pd[re_par] = pd.pop(pop_par)

    def choose_spin_reference(self):
        """Based on the read-in parfiles, choose a reference spin model"""
        return 0

    def choose_binary_reference(self):
        """Based on the read-in parfiles, choose a reference binary model"""
        return 0

    def choose_astrometry_reference(self):
        """Based on the read-in parfiles, choose a reference astrometry model"""
        return 0

    def merge_init(self):
        ref_index = 0

        self.replace_pars_with_ref_model(
            ref_index=ref_index,
            parameters=init_parameters,
            map=map_init_parameters,
        )

    def merge_spin(self):
        """Replace spin parameters with reference spin model parameters"""

        ref_index = self.choose_spin_reference()

        self.replace_pars_with_ref_model(
            ref_index=ref_index, parameters=spin_parameters
        )

    def merge_astrometry(self):
        """Replace astrometry parameters with reference astrometry model parameters"""

        ref_index = self.choose_astrometry_reference()

        self.replace_pars_with_ref_model(
            ref_index=ref_index, parameters=astrometry_parameters
        )

    def merge_binary(self):
        """Replace binary parameters with reference binary model parameters"""

        ref_index = self.choose_astrometry_reference()

        # TODO: if T2 model, find underlying model
        # TODO: deal with parameter aliases

        self.replace_pars_with_ref_model(
            ref_index=ref_index, parameters=binary_parameters
        )

    def merge_dm(self, dm1=True):
        """Replace DM parameters with reference DM model parameters
        param:
        ------
        dm1: defualt, True, whether to add dm1 and dm2. If you use
            the fft time domain psd method, you can set it False.
        """
        # Remove all DM parameters from the model. This includes all DMX parameters. Then
        # we add DM, DM1, and DM2, so that we can do a DMGP model

        dm_val = 0.0
        dm_epoch = 55000.0

        for pfd in self._parfiles:
            pd = pfd["pardict_conv"]
            pops = []

            for parname, parvals in pd.items():
                if parname == "DM":
                    dm_val = float(parvals[0].split()[0])
                    pops.append(parname)
                if parname == "DMEPOCH":
                    dm_epoch = float(parvals[0].split()[0])
                    pops.append(parname)
                elif (
                    parname.startswith("DM")
                    and not parname.startswith("DMJUMP")
                    and not parname == "DM"
                ):
                    pops.append(parname)
                elif parname == "DM":
                    values = parvals[0].split()

                    if len(values) > 1 and values[1] == "1":
                        # The DM is being fit for, so no change
                        pass
                    else:
                        # Change DM to 'fit'
                        pd[parname] = [f"{values[0]}     1"]

            for parname in pops:
                pd.pop(parname)

            if dm1:
                pd.update(
                    {
                        "DM": [f"{dm_val}     1"],
                        "DM1": ["0.0     1"],
                        "DM2": ["0.0     1"],
                        "DMEPOCH": [f"{dm_epoch}"],
                    }
                )
            else:
                pd.update(
                    {
                        "DM": [f"{dm_val}     1"],
                        "DMEPOCH": [f"{dm_epoch}"],
                    }
                )

        ref_index = self.choose_astrometry_reference()
        self.replace_pars_with_ref_model(ref_index=ref_index, parameters=dm_parameters)

    def get_parfile_lines(self, converted=True):
        """Get the lines of all the parfiles, ready to write to disk"""

        pardict_field = "pardict_conv" if converted else "pardict"

        parfiles_d = dict()

        for pfd in self._parfiles:
            parfile_lines = []

            for pname, pvals in pfd[pardict_field].items():
                parfile_lines += [pname + f"    {pv}" for pv in pvals]

            parfiles_d[pfd["pta"]] = parfile_lines

        return parfiles_d


class MetaPulsar(h5p.BasePulsar):
    """Composite pulsar class for multiple PINT and Tempo2 objects"""

    def __init__(
        self,
        pulsars,
        sort=True,
        planets=True,
        drop_t2pulsar=True,
        drop_pintpsr=True,
        merge_astrometry=True,
        merge_spin=True,
        merge_binary=True,
        merge_dm=True,
    ):
        """
        parameters
        ----------
        :pulsars:       dictionary with {'pta1': (pint_model, pint_toas),
                                         'pta2': t2_psr,
                                         'pta3': ...}
        :planets:       Whether to model SS planets
        :drop_t2pulsar: Whethet to delete the libstempo pulsar object
        :drop_pintpsr:  Whether to delete the pint objects
        :merge_astrometry:  Whether to merge astrometry parameters
        :merge_spin:        Whether to merge spin parameters
        :merge_binary:      Whether to merge binary parameters
        :merge_dm:          How/whether to merge DM modeling
        """
        self._sort = sort
        self.planets = planets

        # NOTE: essential that we do not sort the Enterprise Pulsars atm
        # TODO: remove the dependence on sorting by only using the Enterprise API
        # TODO: accept only Enterprise Pulsar objects. Can then also be a FilePulsar
        #       or something like that.

        # Creates self.name and self._epulsars
        self.create_enterprise_pulsars(
            pulsars,
            planets=planets,
            drop_t2pulsar=drop_t2pulsar,
            drop_pintpsr=drop_pintpsr,
        )

        self.set_parameters_from_meta_pulsars(
            merge_astrometry=merge_astrometry,
            merge_spin=merge_spin,
            merge_binary=merge_binary,
            merge_dm=merge_dm,
        )

        self.set_pulsar_attributes()

        self.check_pta_consistency()

        self.drop_pulsars(drop_t2pulsar, drop_pintpsr)

    def unpack_pulsar_dict(self, pulsars):
        """Unpack the pulsars dictionary into PINT and libstempo objects"""

        lt_pulsars = {}
        pint_models = {}
        pint_toas = {}

        for epname, psritem in pulsars.items():
            try:
                # If iterable of 2 items, then unpack as PINT objects
                pmodel, ptoas = psritem

                if not isinstance(pmodel, TimingModel) or not isinstance(ptoas, TOAs):
                    raise TypeError("Not valid PINT objects")

                pint_models.update({epname: pmodel})
                pint_toas.update({epname: ptoas})
            except (KeyError, TypeError, ValueError):
                # Not a pint pulsar, fail silently
                pass

            # Use duck typing to check if it's a tempopulsar-like object
            # Check for common tempopulsar methods/attributes
            is_tempopulsar = False
            try:
                # Check if it has the essential tempopulsar methods
                has_residuals = hasattr(psritem, "residuals") and callable(
                    getattr(psritem, "residuals")
                )
                has_toas = hasattr(psritem, "toas") and callable(
                    getattr(psritem, "toas")
                )
                has_designmatrix = hasattr(psritem, "designmatrix") and callable(
                    getattr(psritem, "designmatrix")
                )

                if has_residuals and has_toas and has_designmatrix:
                    is_tempopulsar = True
                    logger.debug(
                        f"Found tempopulsar-like object for {epname} (duck typing)"
                    )
            except Exception as e:
                logger.debug(f"Error checking tempopulsar methods for {epname}: {e}")

            if is_tempopulsar:
                lt_pulsars.update({epname: psritem})
            else:
                logger.debug(
                    f"Not a tempopulsar-like object for {epname}: {type(psritem)}"
                )

        return pint_models, pint_toas, lt_pulsars

    def check_for_pulsar(self, pint_toas, pint_models, lt_pulsars):
        """Check the objects for a single pulsar name, and return it"""

        pulsar_names = [
            os.path.basename(m.name.split(".")[0]) for m in pint_models.values()
        ] + [psr.name for psr in lt_pulsars.values()]

        if not all_equal(pulsar_names):
            raise ValueError(f"Not all the same pulsar: {pulsar_names}")

        if pint_toas.keys() != pint_models.keys():
            raise ValueError("PINT models and TOAs not provided for same PTAs")

        return pulsar_names[0]

    def create_enterprise_pulsars(
        self, pulsars, planets=True, drop_t2pulsar=True, drop_pintpsr=True
    ):
        """Create Enterprise pulsar objects from libstempo/PINT pulsars"""

        self._epulsars = {}

        pint_models, pint_toas, lt_pulsars = self.unpack_pulsar_dict(pulsars)
        self.name = self.check_for_pulsar(pint_toas, pint_models, lt_pulsars)

        for pta, model in pint_models.items():
            toas = pint_toas[pta]
            self._epulsars.update(
                {
                    pta: ep.PintPulsar(
                        toas,
                        model,
                        sort=False,
                        drop_pintpsr=drop_pintpsr,
                        planets=planets,
                    )
                }
            )

        for pta, psr in lt_pulsars.items():
            self._epulsars.update(
                {
                    pta: ep.Tempo2Pulsar(
                        psr, sort=False, drop_t2pulsar=drop_t2pulsar, planets=planets
                    )
                }
            )

        self._pint_models = pint_models
        self._pint_toas = pint_toas
        self._lt_pulsars = lt_pulsars

    def drop_pulsars(self, drop_t2pulsar=True, drop_pintpsr=True):
        """Drop the original pulsar objects if required"""

        if drop_t2pulsar:
            del self._lt_pulsars

        if drop_pintpsr:
            del self._pint_models
            del self._pint_toas

    def check_fitpar_works(self, psr, parname):
        """Check whether the parameter causes any observable delays"""

        par_index = psr.fitpars.index(parname)
        return np.sum(psr._designmatrix[:, par_index] ** 2) > 0

    def set_parameters_from_meta_pulsars(
        self, merge_astrometry=True, merge_spin=True, merge_binary=True, merge_dm=True
    ):
        """Create mapping from parameters to underlying objects"""
        # Figure out the BAS parameter names (remember ALIASING!)
        # Find out the set/fit/all parameters
        # Aside from BAS, all other parameters get a suffix

        # Need a layout for fitpars:
        # pta, parname  --> fullparname, index
        # fullparname (ordered dict)  -> {pta: parname}

        # Start with pta -> parnames (_epulsar[pta].fitpars)

        # NOTE: Do this better using PINT component parsing
        # The spin parameters (big range for now)

        merge_pars = [
            parname
            for (parnames, merge) in zip(
                [
                    spin_parameters,
                    astrometry_parameters,
                    binary_parameters,
                    dm_parameters,
                ],
                [merge_spin, merge_astrometry, merge_binary, merge_dm],
            )
            for parname in parnames
            if merge
        ]

        fitparameters = defaultdict(dict)
        setparameters = defaultdict(dict)
        for pta, epulsar in self._epulsars.items():
            for parname in epulsar.fitpars:
                meta_parname = parameter_aliases.get(parname, parname)

                if parname in merge_pars:
                    if not check_in_fitpars(parname, self._epulsars):
                        raise ValueError(f"Not all pulsars fit for {meta_parname}")

                    fitparameters[meta_parname].update({pta: parname})

                elif self.check_fitpar_works(epulsar, parname):
                    full_parname = f"{meta_parname}_{pta}"
                    fitparameters[full_parname].update({pta: parname})

            for parname in epulsar.setpars:
                meta_parname = parameter_aliases.get(parname)
                full_parname = f"{meta_parname}_{pta}"
                setparameters[full_parname].update({pta: parname})

        # If there is any overlap between fitparameters and setparameters, that
        # means that merged parameters are fit and frozen in different PTAs.
        # That is worthy of an exception
        if len(set(fitparameters) & set(setparameters)) > 0:
            join = set(fitparameters) & set(setparameters)
            raise ValueError(f"Parameters cannot be frozen & fit for: {join}")

        # Check whether the merge parameters have the same numerical value
        for parname in fitparameters:
            # Use the self._lt_pulsars, self._pint_models, self._pint_toas
            pass

        self._fitparameters = fitparameters
        self._setparameters = setparameters

        self.fitpars = list(fitparameters.keys())
        self.setpars = list(setparameters.keys())

    def get_pta_slices(self):
        """Get index slices for the different PTAs"""
        pta_obs_len = [len(epsr._toas) for epsr in self._epulsars.values()]
        ps = np.cumsum([0] + pta_obs_len)
        pta_slice = {
            pta: slice(ps[ii], ps[ii + 1])
            for (ii, pta) in enumerate(self._epulsars.keys())
        }

        return pta_slice

    def design_matrix_column(self, full_parname):
        """Get a single column of the combined design matrix"""

        # TODO: This needs to be dealt with differently by supporting libstempo units
        units_correction = {
            ("elong", "tempo2"): (1.0 * u.second / u.radian).to(u.second / u.deg).value,
            ("elong", "pint"): 1.0,
            ("elat", "tempo2"): (1.0 * u.second / u.radian).to(u.second / u.deg).value,
            ("elat", "pint"): 1.0,
            ("lambda", "tempo2"): (1.0 * u.second / u.radian)
            .to(u.second / u.deg)
            .value,
            ("lambda", "pint"): 1.0,
            ("beta", "tempo2"): (1.0 * u.second / u.radian).to(u.second / u.deg).value,
            ("beta", "pint"): 1.0,
            ("raj", "tempo2"): (1.0 * u.second / u.radian)
            .to(u.second / u.hourangle)
            .value,
            ("raj", "pint"): 1.0,
            ("decj", "tempo2"): (1.0 * u.second / u.radian).to(u.second / u.deg).value,
            ("decj", "pint"): 1.0,
        }

        colvec = np.zeros_like(self._toas)
        pta_slice = self.get_pta_slices()

        for pta, parname in self._fitparameters[full_parname].items():
            psr = self._epulsars[pta]
            par_index = psr.fitpars.index(parname)

            timing_package = get_timing_package(psr)

            key = (parname.lower(), timing_package.lower())
            units_mult = units_correction.get(key, 1.0)
            colvec[pta_slice[pta]] = psr._designmatrix[:, par_index] * units_mult

        return colvec

    def combine_all_flags(self):
        # NOTE: this uses the Enterprise API, so sorting takes place
        #       take note!
        # TODO: Check whether 128 characters is long enough in Tempo2 source code

        pta_slice = self.get_pta_slices()
        flags = defaultdict(lambda: np.zeros(len(self._toas), dtype="U128"))

        for pta, psr in self._epulsars.items():
            flag_pta = False
            for flag, flagvals in psr.flags.items():
                flags[flag][pta_slice[pta]] = flagvals

                # check whether 'pta' key exists
                if flag == "pta" and not np.any(flagvals == ""):
                    # for '\t' in EPTA data
                    flags[flag][pta_slice[pta]] = [
                        pta_flag.strip() for pta_flag in flagvals
                    ]
                    flag_pta = True

            timing_package = get_timing_package(psr)

            flags["pta_dataset"][pta_slice[pta]] = pta
            flags["timing_package"][pta_slice[pta]] = timing_package

            # if there is no pta flag, then use pta in dict
            if not flag_pta:
                flags["pta"][pta_slice[pta]] = pta

            # TODO: Add flag for how to do DM modeling?
            # TODO: Add flag for wideband data?

        # new-style storage of flags as a numpy record array (previously, psr._flags = flags)
        self._flags = np.zeros(
            len(self._toas), dtype=[(key, val.dtype) for key, val in flags.items()]
        )
        # self._flags = np.zeros(len(self._toas), dtype=[(key, '|S128') for key, val in flags.items()])

        for key, val in flags.items():
            self._flags[key] = val

    def set_position_and_planets(self):
        """Set the position, planet, and ssb quantities"""
        # NOTE: ignores potential sorting

        self._pdist = self._get_pdist()

        rajs_b = [psr._raj for psr in self._epulsars.values()]
        decjs_b = [psr._decj for psr in self._epulsars.values()]

        for psr in self._epulsars.values():
            if psr.name[0] == "B":
                coord_b = SkyCoord(
                    ra=psr._raj * u.rad,
                    dec=psr._decj * u.rad,
                    frame="fk5",
                    equinox="B1950",
                )
                coord_j = coord_b.transform_to(FK5(equinox="J2000"))
                psr._raj = coord_j.ra.rad
                psr._decj = coord_j.dec.rad

        rajs = [psr._raj for psr in self._epulsars.values()]
        decjs = [psr._decj for psr in self._epulsars.values()]

        # there is some confusion about J and B coordinate in parfile
        if (
            not np.allclose(rajs, rajs[0], atol=1e-7, rtol=1e-4)
            or not np.allclose(decjs, decjs[0], atol=1e-7, rtol=1e-3)
        ) and (
            not np.allclose(rajs_b, rajs_b[0], atol=1e-7, rtol=1e-4)
            or not np.allclose(decjs_b, decjs_b[0], atol=1e-7, rtol=1e-3)
        ):
            raise ValueError("Not all pulsar object have the same position")

        if np.allclose(rajs_b, rajs_b[0], atol=1e-7, rtol=1e-4):
            rajs = rajs_b
            decjs = decjs_b

        self._raj, self._decj = rajs[0], decjs[0]
        self._pos = self._get_pos()

        pta_slice = self.get_pta_slices()
        self._sunssb = None
        self._planetssb = None

        if self.planets:
            planetssb = np.empty((len(self._toas), 9, 6))
            sunssb = np.zeros((len(self._toas), 6))

            for pta, psr in self._epulsars.items():
                planetssb[pta_slice[pta]] = psr._planetssb
                sunssb[pta_slice[pta]] = psr._sunssb

            self._planetssb = planetssb
            self._sunssb = sunssb

        self._pos_t = np.zeros((len(self._toas), 3))
        for pta, psr in self._epulsars.items():
            self._pos_t[pta_slice[pta], :] = psr._pos_t

        # which_astrometry = (
        #    "AstrometryEquatorial" if "AstrometryEquatorial" in model.components else "AstrometryEcliptic"
        # )
        # self._pos_t = model.components[which_astrometry].ssb_to_psb_xyz_ICRS(model.get_barycentric_toas(toas)).value

    def set_pulsar_attributes(self):
        """Set all the attributes that Enterprise needs"""

        def concat(attribute):
            return np.concatenate(
                [getattr(epsr, attribute) for epsr in self._epulsars.values()]
            )

        self._toas = concat("_toas")
        self._stoas = concat("_stoas")
        self._residuals = concat("_residuals")
        self._toaerrs = concat("_toaerrs")
        self._ssbfreqs = concat("_ssbfreqs")
        self._telescope = concat("_telescope")

        # dm
        dms = np.unique([psr._dm for psr in self._epulsars.values()])[0]
        self._dm = dms
        self._dmx = []

        self.fitpars = list(self._fitparameters.keys())
        self.setpars = list(self._setparameters.keys())

        self._designmatrix = np.zeros((len(self._toas), len(self.fitpars)))
        for ii, full_parname in enumerate(self._fitparameters.keys()):
            self._designmatrix[:, ii] = self.design_matrix_column(full_parname)

        self.combine_all_flags()
        self.set_position_and_planets()

        # DM/DMX
        # TODO: Enterprise WidebandTimingModel relies on the DMXR1 / RMXR2
        #       parameters, and then compares the Pulsar.toas to those MJDs
        #       It is not clear how to join this approach with the multiple-
        #       pulsar case, so right now we cannot yet do a
        #       WidebandTimingModel.
        # self._dmx = None

        self.sort_data()

    def check_pta_consistency(self):
        """Check the consistency of the timing solutions of the PTAs"""

        # TODO: Create an Enterprise model for the PTAs, use a simple GP DM and
        #       a PL RN model, which is then used to fit for the timing model
        #       parameters. The new timing model parameters and the covariance
        #       can then be used for a consistency statistic.
        pass


def create_metapulsar(
    input_files,
    par_output_dir=None,
    return_metapulsar=True,
    planets=True,
    merge_astrometry=True,
    merge_spin=True,
    merge_binary=True,
    merge_dm=True,
):
    """Create a metapulsar object

    :param input_files: list of dictionaries
    :param par_output_dir:  If not None, where parfiles will be written
    :param return_metapulsar: Whether to return MetaPulsar or the parfile list
    :param planets: bool, defualt=True,
                    It is just requirement for ENTERPRISE object.
    :param merge_astrometry: bool, whether to merge astrometry parameters
    :param merge_spin: bool, whether to merge spin parameters
    :param merge_binary: bool, whether to merge binary parameters
    :param merge_dm:bool, whether to merge dm parameters

    :returns: Enterprise MetaPulsar object

    The input_files is a list of dictionaries, structured like:
    [
        {pta: ptaname, parfile: filepath, timfile: filepath, package: 'pint'},
        {pta: ptaname, parfile: filepath, timfile: filepath, package: 'tempo2'},
    ]

    NOTE: For now, the first par/tim combination will be used as the
          reference model
    """

    mpfs = MetaParfiles(
        input_files,
        merge_astrometry=merge_astrometry,
        merge_spin=merge_spin,
        merge_binary=merge_binary,
        merge_dm=merge_dm,
    )
    pulsar_dict = {}
    parfile_dict = mpfs.get_parfile_lines()

    # TODO: Make communication between mpfs object and here nicer
    for pfd in mpfs._parfiles:

        psr_name = pfd["name"]
        pta = pfd["pta"]
        timing_package = pfd["package"]
        parfile_lines = parfile_dict[pta]

        # We need to work with a temporary directory, because PINT determines
        # the pulsar name from the filename
        with tempfile.TemporaryDirectory() as temp_dir:

            # PINT gets pulsar name from filename
            parfile_name = f"{psr_name}.par"  # For PINT
            parfile_pta_name = f"{psr_name}_{pta}.par"  # To save

            temp_parfile_path = Path(temp_dir) / parfile_name
            with open(temp_parfile_path, "w+") as temp_parfile:

                temp_parfile.write("\n".join(parfile_lines))
                temp_parfile.flush()

                if timing_package in ["libstempo", "tempo2"]:
                    # use sandbox tempo2 to read the par/tim file
                    # TODO: check for maxobs

                    if sandbox_tempopulsar is not None:
                        # Use the new sandbox implementation
                        pulsar_dict[pfd["pta"]] = sandbox_tempopulsar(
                            parfile=temp_parfile.name,
                            timfile=pfd["timfile"],
                            dofit=False,
                        )
                    elif t2 is not None:
                        # Fallback to original libstempo
                        pulsar_dict[pfd["pta"]] = t2.sandbox.tempopulsar(
                            temp_parfile.name,
                            pfd["timfile"],
                            units=False,
                            maxobs=35000,
                            dofit=False,
                        )
                    else:
                        raise ImportError(
                            "Neither sandbox tempo2 nor libstempo available"
                        )

                elif timing_package == "pint":
                    # Use PINT to read the par/tim file

                    pulsar_dict[pfd["pta"]] = get_model_and_toas(
                        temp_parfile.name,
                        pfd["timfile"],
                        planets=planets,
                        allow_T2=True,
                    )

            if par_output_dir:
                # Move the parfile so it is not deleted

                save_parfile_path = Path(par_output_dir) / parfile_pta_name
                shutil.move(temp_parfile_path, save_parfile_path)

    if return_metapulsar:
        return MetaPulsar(
            pulsars=pulsar_dict,
            sort=True,
            planets=True,
            drop_t2pulsar=True,
            drop_pintpsr=True,
            merge_astrometry=merge_astrometry,
            merge_spin=merge_spin,
            merge_binary=merge_binary,
            merge_dm=merge_dm,
        )
    else:
        return pulsar_dict
