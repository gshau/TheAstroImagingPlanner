import numpy as np


FILTERS = ['L', 'R', 'G', 'B', 'Ha', 'OIII', 'SII']


class Filter:
    def __init__(self, name, filter_type=None, bandwidth=None):
        self.name = name
        self.filter_type = filter_type
        self.bandwidth = bandwidth


class FilterWheel:
    def __init__(self, filters, n_pos=7, single_filter_transition_time=1):
        if len(filters) > n_pos:
            raise Exception('Filters outnumber positions!')
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
    def __init__(self, name='template', pitch=5,
                 pix_x=1000,
                 pix_y=1000,
                 read_noise=10,
                 full_well=20000,
                 download_time=20,
                 filter_wheel=FilterWheel(FILTERS)):
        self.name = name
        self.pitch = pitch
        self.pix_x = pix_x
        self.pix_y = pix_y
        self.size = np.array([pix_x, pix_y]) * pitch * 1e-3

        self.read_noise = read_noise
        self.full_well = full_well
        self.download_time = download_time

        self.filter_wheel = filter_wheel

    def dynamic_range(self):
        return 20 * np.log10(self.full_well / self.read_noise)
