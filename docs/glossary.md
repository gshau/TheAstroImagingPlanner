This glossary defines some of the data that's available either in table or graph format.  Generally, for data that's aggregated on a subexposure frame level, the available features are provided either from the FITs header, and are typically capitalized, or from an analysis of stars extracted from the subexposure frame.  

- filename: Name of file

- bkg_val: Background level of image in ADU

- bkg_rms: RMS background level of image

- frame_snr

- star_trail_strength: Metric to determine if star elongation is due to consistent trailing across the frame (i.e. wind, tracking issues)

- star_orientation_score: Metric to determine if star elongation is radial (metric is closer to 1), or azimuthal (metric is closer to 0)

- n_stars: Number of stars detected in image

- extract_thresh: internal threshold for star extraction


For most quantities, the aggregations performed are the average and standard deviation, which are labeled with a suffix `_mean` and `_std`, respectively.  These names are appended to each of the relevant 


- tnpix: Number of pixels associated with each star

- theta: Star elongation angle

- log_flux: log_10 of star's Flux

- fwhm: Full width, half max of stars, in pixels (arcsec for the suffix `_arcsec`)

- eccentricity: Eccentricity of stars

- chip_theta: Star angle with respect to the center of the chip


