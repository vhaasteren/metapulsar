import os
from collections import defaultdict
import argparse
import metapulsar as mp


def get_ipta_release_definitions(ipta_data_dir="/data/IPTA-DR3"):
    """Get all the definitions and files of the IPTA data releases"""

    ipta_data = dict()

    # EPTA_DR2
    ipta_data["epta_dr2"] = dict(
        base_dir=f"{ipta_data_dir}/EPTA_DR2/",
        par_pattern=r"([BJ]\d{4}[+-]\d{2,4})/\1\.par",
        tim_pattern=r"([BJ]\d{4}[+-]\d{2,4})/\1_all\.tim",
        coordinates="ecliptical",
        timing_package="tempo2",
    )

    # PPTA DR2
    ipta_data["ppta_dr2"] = dict(
        base_dir=f"{ipta_data_dir}/PPTA_DR2/",
        par_pattern=r"([BJ]\d{4}[+-]\d{2,4})\.par",
        tim_pattern=r"([BJ]\d{4}[+-]\d{2,4})\.tim",
        coordinates="equatorial",
        timing_package="tempo2",
    )

    # PPTA DR3
    ipta_data["ppta_dr3"] = dict(
        base_dir=f"{ipta_data_dir}/PPTA_DR3/",
        par_pattern=r"([BJ]\d{4}[+-]\d{2,4})\.par",
        tim_pattern=r"([BJ]\d{4}[+-]\d{2,4})\.tim",
        coordinates="equatorial",
        timing_package="tempo2",
    )

    # InPTA_DR1
    ipta_data["inpta_dr1"] = dict(
        base_dir=f"{ipta_data_dir}/InPTA_DR1/",
        par_pattern=r"([BJ]\d{4}[+-]\d{2,4})\/\1\.par",
        tim_pattern=r"([BJ]\d{4}[+-]\d{2,4})\/\1_all\.tim",
        coordinates="equatorial",
        timing_package="tempo2",
    )

    # MPTA_DR1
    ipta_data["mpta_dr1"] = dict(
        base_dir=f"{ipta_data_dir}/MPTA_DR1/",
        par_pattern=r"MTMSP-([BJ]\d{4}[+-]\d{2,4})-\.par",
        tim_pattern=r"([BJ]\d{4}[+-]\d{2,4})_16ch\.tim",
        coordinates="equatorial",
        timing_package="tempo2",
    )

    # NANOGrav_12yr
    ipta_data["nanograv_12y"] = dict(
        base_dir=f"{ipta_data_dir}/NANOGrav_12y/",
        par_pattern=r"par/([BJ]\d{4}[+-]\d{2,4})(?!.*\.t2)_NANOGrav_12yv2\.gls\.par",
        tim_pattern=r"tim/([BJ]\d{4}[+-]\d{2,4})_NANOGrav_12yv2\.tim",
        coordinates="ecliptical",
        timing_package="pint",
    )

    # NANOGrav_15yr
    ipta_data["nanograv_15y"] = dict(
        base_dir=f"{ipta_data_dir}/NANOGrav_15y/",
        par_pattern=r"par/([BJ]\d{4}[+-]\d{2,4})(?!.*(ao|gbt)).*\.par",
        tim_pattern=r"tim/([BJ]\d{4}[+-]\d{2,4})(?!.*(ao|gbt)).*\.tim",
        coordinates="ecliptical",
        timing_package="pint",
    )

    for pta, pta_data in ipta_data.items():
        # Returns tuples in dict of {psrname: (parfile, timfile), ...}
        pta_data["par_and_tim_files"] = mp.get_pta_release_files(**pta_data)

    return ipta_data


def pta_map_to_psr_map(ipta_data):
    """Convert a dict per PTA to a dict per Pulsar"""

    # Convert to 'real' names (todo: do programmatically):
    epoch_map = {
        "J1857+0943": "B1855+09",
        "J1939+2134": "B1937+21",
        "J1955+2908": "B1953+29",
    }
    epoch_map_inv = {value: key for (key, value) in epoch_map.items()}

    all_pulsars = set(
        [
            epoch_map.get(psrname, psrname)
            for pta_data in ipta_data.values()
            for psrname in pta_data["par_and_tim_files"]
        ]
    )

    pulsar_dict = defaultdict(list)

    # Create the MetaPulsar input dict:
    for psrname in all_pulsars:
        for pta, pta_data in ipta_data.items():
            pta_psrnames = set(pta_data["par_and_tim_files"].keys())
            pta_psrnames_alt = {
                epoch_map_inv.get(psrname, psrname) for psrname in pta_psrnames
            }
            pta_psrnames_all = pta_psrnames | pta_psrnames_alt
            pta_psrname = (
                psrname
                if psrname in pta_psrnames
                else epoch_map_inv.get(psrname, psrname)
            )

            if psrname in pta_psrnames_all:
                parfile, timfile = pta_data["par_and_tim_files"][pta_psrname]

                pulsar_dict[psrname].append(
                    {
                        "pta": pta,
                        "parfile": parfile,
                        "timfile": timfile,
                        "package": pta_data["timing_package"],
                        "coordinates": pta_data["coordinates"],
                    }
                )

    return pulsar_dict


def main():

    # Initialize parser
    parser = argparse.ArgumentParser(description="Process Job ID")

    # Adding argument
    parser.add_argument(
        "--procid",
        metavar="N",
        type=int,  # nargs='+',
        help="An integer representing the Job ID",
    )

    parser.add_argument(
        "--ipta_data_dir",
        metavar="N",
        type=str,
        help="IPTA data release directory",
        default="/data/IPTA-DR3",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Whether to overwrite if HDF5 file exists",
    )

    parser.add_argument(
        "--output_dir",
        metavar="N",
        type=str,
        help="Output directory for hdf5 files",
        default="/data/hdf5-pulsars",
    )

    # Parse arguments
    args = parser.parse_args()

    # Access command-line arguments
    job_number = args.procid
    ipta_data_dir = args.ipta_data_dir
    h5dir = args.output_dir
    overwrite = args.overwrite

    # All the IPTA data releases and par/tim files
    ipta_data = get_ipta_release_definitions(ipta_data_dir=ipta_data_dir)

    # Select subset of PTAs:
    subset = ["epta_dr2", "ppta_dr3", "inpta_dr1", "mpta_dr1", "nanograv_15y"]
    ipta_dr3_data = {key: value for (key, value) in ipta_data.items() if key in subset}

    # Convert this to a dict per Pulsar so we can create MetaPulsar objects
    pulsar_dict = pta_map_to_psr_map(ipta_dr3_data)

    for procid, (psrname, pulsar_l) in enumerate(pulsar_dict.items()):
        outfile = os.path.join(h5dir, psrname + ".h5")

        if not os.path.isfile(outfile):
            print(f"Pulsar[{procid}] {psrname} is not done yet")


if __name__ == "__main__":
    main()
