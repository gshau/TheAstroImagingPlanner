import rasterio
from rasterio.windows import Window

import numpy as np


class LightPollutionMap:
    # World Atlas 2015: Falchi, Fabio; Cinzano, Pierantonio; Duriscoe, Dan;
    # Kyba, Christopher C. M.; Elvidge, Christopher D.; Baugh, Kimberly;
    # Portnov, Boris; Rybnikova, Nataliya A.; Furgoni, Riccardo (2016):
    # Supplement to: The New World Atlas of Artificial Night Sky Brightness.
    # GFZ Data Services. http://doi.org/10.5880/GFZ.1.4.2016.001
    def __init__(self):
        filename = "/app/data/sky_atlas/World_Atlas_2015_compressed.tif"
        self.dataset = rasterio.open(filename)
        self.background_brightness = 0.174  # units of mcd/m^2
        self.background_mpsas = 22  # units of magnitude per square arc-second

    def brightness_for_location(self, latitude, longitude):
        col_offset, row_offset = self.dataset.index(longitude, latitude)
        self.data = self.dataset.read(1, window=Window(row_offset, col_offset, 1, 1))
        assert self.data.shape == (1, 1)
        brightness = self.data[0, 0]
        return brightness + self.background_brightness

    def mpsas_for_location(self, latitude, longitude):
        brightness = self.brightness_for_location(latitude, longitude)
        offset = np.log(brightness / self.background_brightness) / np.log(2.512)
        return self.background_mpsas - offset
