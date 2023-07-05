import numpy as np


class Filter:
    def __init__(self, name, **kwargs):
        self.name = name
        self.bandpass = kwargs.get("bandpass", None)
        self.lam_range = kwargs.get("lam_range", None)
        self.lam_min = None
        self.lam_max = None
        if self.lam_range:
            self.lam_min = self.lam_range["min"]
            self.lam_max = self.lam_range["max"]
            self.bandpass = self.lam_max - self.lam_min

    def __repr__(self):
        if self.lam_range:
            return (
                f"{self.name} with wavelength range {self.lam_min} - {self.lam_max}nm"
            )
        elif self.bandpass:
            return f"{self.name} with bandpass {self.bandpass}nm"
        else:
            return self.name


class FilterWheel:
    def __init__(self, filters, n_pos=7, single_filter_transition_time=1):
        if len(filters) > n_pos:
            raise Exception("Filters outnumber positions!")
        self.filters = filters
        self.n_pos = n_pos
        self.filter_pos = dict(zip(self.filters, np.arange(len(self.filters))))
        self.single_filter_transition_time = single_filter_transition_time

    def time_to_filter_change(self, from_filter, to_filter):
        i_start = self.filter_pos[from_filter]
        i_end = self.filter_pos[to_filter]
        n_move = (i_end - i_start) % self.n_pos
        return n_move * self.single_filter_transition_time


class Sensor:
    def __init__(
        self,
        name,
        pixel_size,
        n_x,
        n_y,
        quantum_efficiency=None,
        gain=None,
        bit_depth=None,
        read_noise=None,
        full_well=None,
        download_time=None,
        filters=None,
        bias_level_adu=500,
        **kwargs,
    ):
        self.name = name
        self.pixel_size = float(pixel_size)
        self.n_x = int(n_x)
        self.n_y = int(n_y)

        self.read_noise = read_noise
        self.full_well = full_well
        self.download_time = download_time
        self.quantum_efficiency = quantum_efficiency
        self.gain = gain
        self.bit_depth = bit_depth
        self.bias_level_adu = bias_level_adu

        self.filters = dict(
            [
                [filter_name, Filter(filter_name, **filter_specs)]
                for filter_name, filter_specs in filters.items()
            ]
        )

    @property
    def dynamic_range(self):
        return 20 * np.log10(self.full_well / self.read_noise)

    @property
    def size(self):
        return np.array([self.n_x, self.n_y]) * self.pixel_size * 1e-3

    def __repr__(self):
        return f"{self.name} {self.n_x}x{self.n_y} pixels @ {self.pixel_size}um"
