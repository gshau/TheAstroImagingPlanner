from .logger import log
from astro_planner.globals import EXPOSURE_COL, EXC_INFO


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
        log.warning(f"Error with {string}", exc_info=EXC_INFO)
        pass
    return r
