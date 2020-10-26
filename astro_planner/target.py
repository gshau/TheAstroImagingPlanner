import re


import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

from astropy.coordinates import Angle

from collections import defaultdict
import pandas_access as mdb

import json
import ntpath
from .logger import log


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

    def __repr__(self):
        return f"{self.name} at RA={self.ra} DEC={self.dec}"


def clean_subframes(subframes, n_subs_name="subs_requested"):
    return dict((k, v) for k, v in subframes.items() if v[n_subs_name] > 0)


def normalize_target_name(target):
    target = target.lower()
    target = target.replace("-", "_")
    target = target.replace(" ", "_")
    target = re.sub(r"^(?:sh2)_*", "sh2-", target)
    for catalog in ["ic", "vdb", "ngc", "ldn", "lbn", "arp", "abell"]:
        target = re.sub(f"^(?:{catalog})_*", f"{catalog}_", target)
    return target


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
            target_name = normalize_target_name(row.TARGET)
            target = Target(
                target_name,
                ra=row.RAJ2000 * u.hourangle,
                dec=row.DECJ2000 * u.deg,
                notes=note,
            )
            self.target_list[profile][target_name] = target

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
    elif ".sgf" in filename:
        return SGPSequenceObjects(filename)
