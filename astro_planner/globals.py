import string
import random
import os
import getpass
import sys
from pathlib import Path
import shutil


IS_WINDOWS = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


path = Path(__file__).resolve().parents[1]
BASE_DIR = os.environ.get("BASE_DIR", path)
DATA_DIR = BASE_DIR

if IS_MAC:
    username = getpass.getuser()
    DATA_DIR = f"/Users/{username}/Library/ApplicationSupport/AstroImagingPlanner"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copytree(f"{BASE_DIR}/data", f"{DATA_DIR}/data")


EXPOSURE_COL = "Exposure"
INSTRUMENT_COL = "Instrument"
FOCALLENGTH_COL = "Focal Length"
BINNING_COL = "Binning"
PIXELSIZE_COL = "Pixel Size"


ENV = "primary"
EXC_INFO = True


ROUTE_PREFIX = (
    "/" + "".join(random.choice(string.ascii_lowercase) for i in range(8)) + "/"
)
ROUTE_PREFIX = "/"


TABLE_EXPORT_FORMAT = "csv"


# TODO: CHANGE TO ENUMS
L_FILTER = "L"
R_FILTER = "R"
G_FILTER = "G"
B_FILTER = "B"
HA_FILTER = "Ha"
OIII_FILTER = "OIII"
SII_FILTER = "SII"
BAYER = "OSC"
BAYER_ = "** BayerMatrix **"
BAYER__ = "__ BayerMatrix __"
NO_FILTER = "NO_FILTER"

FILTER_LIST = [
    L_FILTER,
    R_FILTER,
    G_FILTER,
    B_FILTER,
    HA_FILTER,
    OIII_FILTER,
    SII_FILTER,
    BAYER,
    BAYER_,
    BAYER__,
    NO_FILTER,
]


FILTER_MAP = {
    R_FILTER: ["Red"],
    G_FILTER: ["Green"],
    B_FILTER: ["Blue"],
    L_FILTER: ["Luminance"],
    HA_FILTER: ["HA"],
    OIII_FILTER: ["O3"],
    SII_FILTER: ["S2"],
    BAYER: [BAYER_, BAYER__],
}


COLORS = {
    L_FILTER: "black",
    R_FILTER: "red",
    G_FILTER: "green",
    B_FILTER: "blue",
    HA_FILTER: "crimson",
    SII_FILTER: "maroon",
    OIII_FILTER: "teal",
    BAYER: "gray",
    BAYER_: "gray",
    BAYER__: "gray",
    NO_FILTER: "gray",
}


BB_FILTERS = [L_FILTER, R_FILTER, G_FILTER, B_FILTER]
NB_FILTERS = [HA_FILTER, OIII_FILTER, SII_FILTER]

TRANSLATED_FILTERS = {
    "ha": ["ho", "sho", "hoo", "hos", "halpha", "h-alpha"],
    "oiii": ["ho", "sho", "hoo", "hos"],
    "sii": ["sho", "hos"],
    "nb": ["ha", "oiii", "sii", "sho", "ho", "hoo", "hos", "halpha", "h-alpha"],
    "bb": ["luminance", "lrgb", "lum"],
    "rgb": ["osc", "bayer", "dslr", "slr", "r ", " g ", " b ", "rgb"],
    "lum": ["luminance", "lum"],
}


IS_PROD = bool(os.getenv("IS_PROD", True))
# IS_PROD = False
