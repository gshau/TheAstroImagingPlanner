class ConfigDataBase:
    def __init__(self):
        pass

    def set_var(self, var):
        for key, value in var.items():
            setattr(self, key, value)

    def dict(self):
        return self.__dict__


class InspectorThresholds(ConfigDataBase):
    ECC_THR = "ecc_thr"
    STAR_FRAC_THR = "star_frac_thr"
    Z_SCORE = "z_score"
    IQR_SCALE = "iqr_scale"
    TRAIL_THR = "trail_thr"
    GRADIENT_THR = "gradient_thr"

    def __init__(self):
        self.ecc_thr = 0.6
        self.gradient_thr = 0.1
        self.iqr_scale = 1.5
        self.star_frac_thr = 0.5
        self.trail_thr = 8
        self.z_score = 5


class PlannerConfig(ConfigDataBase):
    LAT = "lat"
    LON = "lon"
    UTC_OFFSET = "utc_offset"
    MPSAS = "mpsas"
    MIN_FRAME_OVERLAP_FRACTION = "min_frame_overlap_fraction"
    MIN_MOON_DISTANCE = "min_moon_distance"
    TIME_RESOLUTION = "time_resolution"
    SOLAR_ALTITUDE_FOR_NIGHT = "solar_altitude_for_night"
    K_EXTINCTION = "k_extinction"
    PROFILES = "profiles"
    TARGET_PRIORITIES = "target_priorities"
    TARGET_STATUS = "target_status"
    COLOR_PALETTE = "color_palette"

    def __init__(self):
        self.mpsas = 22
        self.lat = 40
        self.lon = -90
        self.utc_offset = -6
        self.k_extinction = 0.16
        self.min_frame_overlap_fraction = 0.7
        self.min_moon_distance = 30
        self.time_resolution = 300
        self.solar_altitude_for_night = -14
        self.profiles = []
        self.target_priorities = []
        self.target_status = []
        self.color_palette = "base"


class VoyagerConfig(ConfigDataBase):
    HOSTNAME = "hostname"
    PASSWORD = "password"
    USER = "user"
    PORT = "port"

    def __init__(self):
        self.hostname = ""
        self.password = ""
        self.user = ""
        self.port = ""


class SwitchConfig(ConfigDataBase):
    SIRIL_SWITCH = "siril_switch"
    SILENCE_ALERTS_SWITCH = "silence_alerts_switch"
    VOYAGER_SWITCH = "voyager_switch"
    PLANNER_SWITCH = "planner_switch"
    INSPECTOR_SWITCH = "inspector_switch"
    CULL_DATA_SWITCH = "cull_data_switch"

    def __init__(self):
        self.silence_alerts_switch = False
        self.siril_switch = False
        self.voyager_switch = False
        self.planner_switch = True
        self.inspector_switch = True
        self.cull_data_switch = False
