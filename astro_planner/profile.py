import pandas_access as mdb
from collections import defaultdict
import logging


from .camera import *
from .telescope import *

DEFAULT_DATA_PATH = r'C:\Users\Gabe\Dropbox\AstroBox\data'
DEFAULT_PROFILE_PATH = r'C:\Users\Gabe\Dropbox\AstroBox\Voyager Profiles'

DEGREES_PER_RADIAN = 180. / np.pi
ARCMIN_PER_DEGREE = 60


class Profile:
    def __init__(self, sensor, telescope, name=None, path=DEFAULT_PROFILE_PATH):
        if name:
            logging.warning('Profile name may not conform to standard!')
            self.name = name
        else:
            self.name = '_'.join(
                [cleanup_name(telescope.name), cleanup_name(sensor.name)])
        self.path = path
        self.filename = r'{}\{}'.format(self.path, self.name)
        self.sensor = sensor
        self.telescope = telescope

    def summary(self):
        print(self.name)
        print('Pixel scale:   {:.2f}'.format(self.pixel_scale()))
        print('Field of view: {:.1f} x {:.1f}'.format(*list(self.fov())))

    def pixel_scale(self):
        return self.sensor.pitch * 206.3 / self.telescope.focal_length

    def fov(self):
        return ARCMIN_PER_DEGREE * DEGREES_PER_RADIAN * np.arctan(self.sensor.size / self.telescope.focal_length)


def cleanup_name(name):
    new_name = name.lower()
    new_name = new_name.replace(' ', '_')
    new_name = new_name.replace('-', '_')
    new_name = new_name.replace('.', '')
    return new_name
