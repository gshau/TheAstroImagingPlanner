import numpy as np
import pandas as pd
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

# from astroplan import FixedTarget
from astropy.wcs import WCS
from astropy.coordinates import Angle

from .stf import auto_stf
from .profile import cleanup_name
from .data_parser import get_data_info

from collections import defaultdict
import pandas_access as mdb


RA_KEY = "RA"
DEC_KEY = "DEC"
TARGET_KEY = "target"
DATA_DIR_KEY = "data_dir"
PROFILE_KEY = "profile_filename"

L_FILTER = "L"
R_FILTER = "R"
G_FILTER = "G"
B_FILTER = "B"
HA_FILTER = "Ha"
OIII_FILTER = "OIII"
SII_FILTER = "SII"

NARROWBAND_SUBEXPOSURE = 900
RGB_SUBEXPOSURE = 300
LUM_SUPEXPOSURE = 300
DEFAULT_SUBEXPOSURES = {
    L_FILTER: LUM_SUPEXPOSURE,
    R_FILTER: RGB_SUBEXPOSURE,
    G_FILTER: RGB_SUBEXPOSURE,
    B_FILTER: RGB_SUBEXPOSURE,
    HA_FILTER: NARROWBAND_SUBEXPOSURE,
    OIII_FILTER: NARROWBAND_SUBEXPOSURE,
    SII_FILTER: NARROWBAND_SUBEXPOSURE,
}


class Target:
    def __init__(self, name, ra=None, dec=None, notes=""):
        self.name = name
        if ra is None or dec is None:
            self.target = SkyCoord.from_name(name)
            self.ra = self.target.ra
            self.dec = self.target.dec
        else:
            self.target = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
            self.ra = ra
            self.dec = dec
        self.ra_string = Angle(self.ra, unit="hourangle").to_string(sep=" ")
        self.dec_string = Angle(self.dec).to_string(sep=" ")

        self.info = {}
        self.info[RA_KEY] = self.ra_string
        self.info[DEC_KEY] = self.dec_string
        self.info[TARGET_KEY] = self.name.lower()
        self.info["notes"] = notes


def clean_subframes(subframes, n_subs_name="subs_requested"):
    return dict((k, v) for k, v in subframes.items() if v[n_subs_name] > 0)


class ImagingTarget(Target):
    def __init__(self, name, subframes):
        super().__init__(name)

        self.subframes = clean_subframes(subframes)
        self.clean_name = cleanup_name(self.name)
        self.info.update({"subframes": self.subframes})

    def exposure_summary(self):

        df_requested = (
            pd.DataFrame.from_dict(self.info["subframes"], orient="index")
            .reset_index()
            .rename({"index": "filter"}, axis=1)
        )
        df_acquired = (
            pd.DataFrame.from_dict(self.exposure_on_file(), orient="index")
            .reset_index()
            .rename({"index": "filter"}, axis=1)
        )

        df_summary = pd.merge(df_requested, df_acquired, on=["filter", "bin"])
        df_summary["completed"] = (
            df_summary["exposure_acquired"] > df_summary["exposure_requested"]
        )

        df_summary["exposure_remaining"] = np.clip(
            df_summary["exposure_requested"] - df_summary["exposure_acquired"], 0, None
        )
        df_summary["subs_remaining"] = np.clip(
            df_summary["subs_requested"] - df_summary["subs_acquired"], 0, None
        )

        return df_summary

    def total_exposure_requested(self):
        exposure = 0
        for filter_name, data in self.subframes.items():
            exposure += data["sub_exposure"] * data["subs_requested"]
        return exposure

    def exposure_on_file(self):
        df = get_data_info()
        df0 = (
            df[df["target"] == self.clean_name]
            .groupby(["filter", "sub_exposure", "bin"])
            .count()["target"]
            .reset_index()
            .set_index("filter")
        )
        df0 = df0.rename({"target": "subs_acquired"}, axis=1)
        df0["exposure_acquired"] = (df0["sub_exposure"] * df0["subs_acquired"]).astype(
            int
        )
        df0 = df0.drop(["sub_exposure"], axis=1)

        df0
        subframes = df0.to_dict(orient="index")
        return clean_subframes(subframes, n_subs_name="subs_acquired")

    def remaining_exposure(self):
        df_request = pd.DataFrame(self.subframes).T
        df_request.index.name = "filter"
        df_request = df_request.reset_index().set_index(
            ["filter", "exposure", "binning"]
        )

        df_obtained = pd.DataFrame(self.exposure_on_file()).T
        df_obtained.index.name = "filter"
        df_obtained = df_obtained.reset_index().set_index(
            ["filter", "exposure", "binning"]
        )

        df_diff = df_request - df_obtained
        df_diff[df_diff < 0] = 0
        subframes = df_diff.reset_index().set_index("filter").to_dict(orient="index")
        return clean_subframes(subframes)

    def is_complete(self):
        return len(self.remaining_exposure().keys()) == 0


def get_roboclips(filename="/Volumes/Users/gshau/Dropbox/AstroBox/roboclip/VoyRC.mdb"):

    df_rc = mdb.read_table(filename, "RoboClip", converters_from_schema=False)
    target_list = defaultdict(dict)
    for row in df_rc.itertuples():
        profile = row.GRUPPO
        target = Target(
            row.TARGET,
            ra=row.RAJ2000 * u.hourangle,
            dec=row.DECJ2000 * u.deg,
            notes=row.NOTE,
        )
        target_list[profile][row.TARGET] = target

    return target_list


class Objects:
    def __init__(self):
        self.target_list = defaultdict(dict)
        self.profiles = []

    def process_objects(self, df_input):
        self.target_list = defaultdict(dict)
        for row in df_input.itertuples():
            profile = row.GROUP
            note = str(row.NOTE)
            if note == "nan":
                note = ""
            target = Target(
                row.TARGET,
                ra=row.RAJ2000 * u.hourangle,
                dec=row.DECJ2000 * u.deg,
                notes=note,
            )
            self.target_list[profile][row.TARGET] = target

    def load_from_df(self, df_input):
        self.df_objects = df_input
        self.process_objects(self.df_objects)
        self.profiles = sorted(list(self.target_list.keys()))


class RoboClipObjects(Objects):
    def __init__(self, filename):
        super().__init__()
        self.df_objects = mdb.read_table(
            filename, "RoboClip", converters_from_schema=False
        )
        self.df_objects.rename({"GRUPPO": "GROUP"}, axis=1, inplace=True)
        self.load_from_df(self.df_objects)


class SGPSequenceObjects(Objects):
    def __init__(self, filename):
        super().__init__()
        for self.filename in [filename]:
            with open(self.filename, "r") as f:
                self.data = json.load(f)
                self.df_objects = self.parse_data()
                self.process_objects(self.df_objects)

    def parse_data(self):
        self.sequence = {}
        root_name = ntpath.basename(self.filename)
        self.profiles.append(root_name)
        for sequence in self.data["arEventGroups"]:
            name = sequence["sName"]
            ref = sequence["siReference"]

            RA = ref["nRightAscension"]
            DEC = ref["nDeclination"]
            events = sequence["Events"]
            filters = []
            event_data = []
            note_string = ""
            for event in events:
                filters.append(event["sSuffix"])

                event_data.append(event)
                log.info(event_data)
                note_string += "<br> {filter} {exp}s ({ncomplete} / {ntotal}) exposure: {total_exposure:.1f}h".format(
                    filter=event["sSuffix"],
                    exp=event["nExposureTime"],
                    ncomplete=event["nNumComplete"],
                    ntotal=event["nRepeat"],
                    total_exposure=event["nNumComplete"]
                    * event["nExposureTime"]
                    / 3600,
                )
            notes = note_string
            self.sequence[name] = dict(
                RAJ2000=RA, DECJ2000=DEC, NOTE=notes, TARGET=name, GROUP=root_name
            )
        return pd.DataFrame.from_dict(self.sequence, orient="index").reset_index(
            drop=True
        )


def object_file_reader(filename):
    if ".mdb" in filename:
        return RoboClipObjects(filename)
    # if isinstance(filename, list):
    #     for file in
    elif ".sgf" in filename:
        return SGPSequenceObjects(filename)
