import numpy as np


class Filter:
    def __init__(self, name, filter_type=None, bandwidth=None):
        self.name = name
        self.filter_type = filter_type
        self.bandwidth = bandwidth


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
        read_noise=None,
        full_well=None,
        download_time=None,
        filter_wheel=None,
        **kwargs,
    ):
        self.name = name
        self.pixel_size = float(pixel_size)
        self.n_x = int(n_x)
        self.n_y = int(n_y)
        self.size = np.array([self.n_x, self.n_y]) * self.pixel_size * 1e-3

        self.read_noise = read_noise
        self.full_well = full_well
        self.download_time = download_time

        self.filter_wheel = filter_wheel

    def dynamic_range(self):
        return 20 * np.log10(self.full_well / self.read_noise)

    def __repr__(self):
        return f"{self.name} {self.n_x}x{self.n_y} pixels @ {self.pixel_size}um"
