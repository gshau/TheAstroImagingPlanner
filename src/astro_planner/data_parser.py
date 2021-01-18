import os
import yaml


from .logger import log

DATA_DIR = os.getenv("DATA_DIR", "/data/")

FILTERS = ["L", "R", "G", "B", "Ha", "OIII", "SII", "OSC"]

with open("/app/conf/config.yml", "r") as f:
    CONFIG = yaml.load(f, Loader=yaml.BaseLoader)

SENSOR_MAP = CONFIG.get("sensor_map", {})
FITS_HEADER_MAP = CONFIG.get("fits_header_map", {})

EXPOSURE_COL = "Exposure"
INSTRUMENT_COL = "Instrument"
FOCALLENGTH_COL = "Focal Length"
BINNING_COL = "Binning"
PIXELSIZE_COL = "Pixel Size"

FITS_DF_COL_MAP = {}
for col in FITS_HEADER_MAP.get("exposure", ["EXPOSURE"]):
    FITS_DF_COL_MAP[col] = EXPOSURE_COL
for col in FITS_HEADER_MAP.get("instrument", ["INSTRUME"]):
    FITS_DF_COL_MAP[col] = INSTRUMENT_COL
for col in FITS_HEADER_MAP.get("focal_length", ["FOCALLEN"]):
    FITS_DF_COL_MAP[col] = FOCALLENGTH_COL
for col in FITS_HEADER_MAP.get("binning", ["XBINNING"]):
    FITS_DF_COL_MAP[col] = BINNING_COL
for col in FITS_HEADER_MAP.get("pixel_size", ["XPIXSZ"]):
    FITS_DF_COL_MAP[col] = PIXELSIZE_COL


def format_name(name):
    name = name.lower()
    name = name.replace(" ", "_")
    if "sh2" not in name:
        name = name.replace("-", "_")
    catalogs = ["ngc", "abell", "ic", "vdb", "ldn"]

    for catalog in catalogs:
        if catalog in name[: len(catalog)]:
            if f"{catalog}_" in name:
                continue
            number = name.replace(catalog, "")
            name = f"{catalog}_{number}"
    return name


def filter_map(filter_in):
    filters_to_replace = dict(
        Clear="L", Red="R", Green="G", Blue="B", SIII="SII", Luminance="L"
    )
    if filter_in in filters_to_replace:
        return filters_to_replace[filter_in]
    return filter_in


def equinox_ccdfname_parser(string):
    try:
        split_string = string.split(".")
        r = dict(zip(["OBJECT", "IMAGETYP"], split_string[:2]))
        exposure, remain = split_string[2].split("S", 1)
        temp, remain = remain.split("X")
        bin = remain[0]
        filter = remain[1:]
        r.update(
            {
                EXPOSURE_COL: int(exposure),
                "CCD-TEMP": int(temp),
                "XBINNING": int(bin),
                "YBINNING": int(bin),
            }
        )
        r.update({"FILTER": filter_map(filter)})
    except:
        log.warning(f"Error with {string}", exc_info=True)
        pass
    return r
