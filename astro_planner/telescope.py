class Telescope:
    def __init__(self, name, aperture, focal_length):
        self.name = name
        self.aperture = aperture
        self.focal_length = focal_length
        self.focal_ratio = focal_length / aperture

