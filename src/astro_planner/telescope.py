import numpy as np


class Telescope:
    def __init__(
        self,
        name,
        aperture,
        focal_length,
        optical_efficiency=0.95,
        central_obstruction=0,
    ):
        self.name = name
        self.aperture = aperture
        self.focal_length = focal_length
        self.optical_efficiency = optical_efficiency
        self.central_obstruction = central_obstruction

    @property
    def focal_ratio(self):
        return self.focal_length / self.aperture

    @property
    def area(self):
        return ((self.aperture / 2) ** 2 - (self.central_obstruction / 2) ** 2) * np.pi

    @property
    def cfz(self):
        return 2 * self.focal_ratio * self.airy_disk_um

    @property
    def airy_disk_um(self):
        lam = 0.510  # in microns
        return 2.43932 * lam * self.focal_ratio

    @property
    def airy_disk_as(self):
        lam = 0.510e-3
        return 2 * np.arctan(1.21966 * lam / self.aperture) * 180 / np.pi * 3600

    def __repr__(self):
        return f"{self.name} {self.aperture}mm aperture @ f/{self.focal_ratio:.2f}"

