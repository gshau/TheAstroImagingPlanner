import re
import os

import time
import websocket
import base64


import sqlite3
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord

from astropy.coordinates import Angle

from collections import defaultdict
from xml.etree import ElementTree

import json
from .logger import log
from .globals import IS_WINDOWS
from .update_voyager_rating import (
    VoyagerConnectionManager,
    receive_message_callback,
)


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


def normalize_target_name(target):
    if target:
        target = target.lower()
        target = target.replace("-", "_")
        target = target.replace(" ", "_")
        target = re.sub(r"^(?:sh2)_*", "sh2-", target)
        for catalog in ["ic", "vdb", "ngc", "ldn", "lbn", "arp", "abell"]:
            target = re.sub(f"^(?:{catalog})_*", f"{catalog}_", target)
    return target


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


async def get_robotargets(connection_manager):
    # Get targets
    params = {}
    result = await connection_manager.send_command(
        "RemoteOpenRoboTargetGetTargetList", params, with_mac=True
    )
    robotarget_list = connection_manager.msg_list
    return robotarget_list


async def get_roboclip_targets(connection_manager):
    params = {}
    params["Order"] = 0
    params["FilterName"] = ""
    params["FilterGroup"] = ""
    params["FilterNote"] = ""

    result = await connection_manager.send_command(
        "RemoteRoboClipGetTargetList", params, with_mac=False
    )
    #     time.sleep(1)
    roboclip_list = connection_manager.msg_list

    return roboclip_list


async def get_robotargets_roboclip(server_url, server_port, auth_token):
    websocket.enableTrace(True)
    connection_manager = VoyagerConnectionManager(
        server_url=server_url, server_port=server_port
    )
    connection_manager.run_forever()
    connection_manager.receive_message_callback = receive_message_callback

    time.sleep(1)

    encoded_token = base64.urlsafe_b64encode(auth_token.encode("ascii"))
    result = await connection_manager.send_command(
        "AuthenticateUserBase", {"Base": encoded_token.decode("ascii")}
    )

    df_robotargets = pd.DataFrame(await get_robotargets(connection_manager))
    df_roboclip = pd.DataFrame(await get_roboclip_targets(connection_manager))

    connection_manager.ws.close()
    df_merged_targets = pd.merge(
        df_robotargets,
        df_roboclip,
        on="targetname",
        how="left",
        suffixes=["_robotarget", "_roboclip"],
    )
    df_merged_targets = df_merged_targets[
        [
            "guid_robotarget",
            "guid_roboclip",
            "targetname",
            "tag",
            "gruppo",
            "status",
            "statusop",
            "setname",
            "settag",
            "profilename",
            "raj2000",
            "decj2000",
            "pa",
            "dx",
            "dy",
            "note",
            "pixelsize",
            "focallen",
        ]
    ]
    return df_merged_targets


class Targets:
    def __init__(self):
        self.target_list = defaultdict(dict)
        self.profiles = []

    def process_targets(self, df_input):
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
        self.df_targets = df_input
        self.process_targets(self.df_targets)
        profiles = [
            profile for profile in self.target_list.keys() if isinstance(profile, str)
        ]
        self.profiles = sorted(profiles)


class RoboClipTargets(Targets):
    def __init__(self, filename, mdb_export_path=""):
        super().__init__()

        conn = sqlite3.connect(filename)
        df_targets = pd.read_sql("select * from RoboClip", conn)
        self.df_targets = df_targets

        self.df_targets.rename({"GRUPPO": "GROUP"}, axis=1, inplace=True)
        self.df_targets["GROUP"] = self.df_targets["GROUP"].fillna("UNLABELED")
        self.load_from_df(self.df_targets)


class SGPSequenceTargets(Targets):
    def __init__(self, filename):
        super().__init__()
        for self.filename in [filename]:
            with open(self.filename, "r") as f:
                self.data = json.load(f)
                self.df_targets = self.parse_data()
                self.process_targets(self.df_targets)

    def parse_data(self):
        self.sequence = {}
        root_name = os.path.basename(self.filename)
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
                log.debug(event_data)
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
            log.debug(self.sequence[name])
        return pd.DataFrame.from_dict(self.sequence, orient="index").reset_index(
            drop=True
        )


class NINASequenceTargets(Targets):
    def __init__(self, filename):
        super().__init__()
        for self.filename in [filename]:
            self.tree = ElementTree.parse(self.filename)
            self.df_targets = self.parse_data()
            self.process_targets(self.df_targets)

    def parse_data(self):
        self.sequence = {}
        root_name = os.path.basename(self.filename)
        self.profiles.append(root_name)
        r = self.tree.getroot()
        if r.tag != "CaptureSequenceList":
            e_list = self.tree.findall("CaptureSequenceList")
        else:
            e_list = [r]
        for e in e_list:
            target_name = e.get("TargetName")
            c = e.find("Coordinates")
            ra = float(c.find("RA").text)
            dec = float(c.find("Dec").text)

            note = f"NINA sequence file {root_name}"
            self.sequence[target_name] = dict(
                RAJ2000=ra, DECJ2000=dec, TARGET=target_name, GROUP=root_name, NOTE=note
            )
            log.debug(self.sequence[target_name])
        df = pd.DataFrame.from_dict(self.sequence, orient="index").reset_index(
            drop=True
        )

        return df


def target_file_reader(filename, **kwargs):
    if "VoyRC.db" in filename:
        return RoboClipTargets(filename, **kwargs)
    if ".sgf" in filename:
        return SGPSequenceTargets(filename, **kwargs)
    elif ".xml" in filename or ".ninaTargetSet" in filename:
        return NINASequenceTargets(filename, **kwargs)


class RoboTargets(Targets):
    def __init__(self, server_url, server_port, auth_token):
        super().__init__()
        self.server_url = server_url
        self.server_port = server_port
        self.auth_token = auth_token

    async def foo(self):
        df_targets = await get_robotargets_roboclip(
            self.server_url, self.server_port, self.auth_token
        )
        df_targets = df_targets.rename(
            {
                "note": "NOTE",
                "targetname": "TARGET",
                "raj2000": "RAJ2000",
                "decj2000": "DECJ2000",
                "gruppo": "GROUP",
            },
            axis=1,
        )
        df_targets = df_targets.dropna(
            subset=["RAJ2000", "DECJ2000", "GROUP", "TARGET", "NOTE"]
        )
        self.df_targets = df_targets
        # self.df_targets["GROUP"] = f"RoboTarget {self.server_url}"
        self.load_from_df(self.df_targets)


async def robotarget_reader(server_url, server_port, auth_token):
    rt = RoboTargets(server_url, server_port, auth_token)
    await rt.foo()
    df = rt.df_targets.drop("status", axis=1)
    return df
