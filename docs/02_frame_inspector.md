## Frame Inspector

[//]: # (src/assets/inspector_tab.png)
The Frame Inspector tab allows you to drill down into the data you have stored.  There are four primary options to filter the data: Date, Target, Focal Length, Pixel Size.  


- Monitor mode - monitor for new files. Acquired data and subframe graphs will update with the new data as it comes in. The indicator for this mode is at the top right of the page.
- Label Points - add target names to each point in the scatter plot.


### Frame Acceptance Criteria
The thresholds for accepting or rejecting frames allow you to adjust your tolerance for good/bad frames.  

- `Max Eccentricity` - maximum allowed eccentricity.  
- `Min Star Fraction` - minimum allowed star fraction.  The star fraction is calculated by the number of stars detected in a frame, divided by the maximum number of stars for the groupcombination.  Set this to zero to effectively turn this off.
- `z score` - maximum allowed [z-score](https://en.wikipedia.org/wiki/Standard_score#Calculation) for eccentricity and FWHM. Set this very large to effectively turn this off.
- `IQR scale` - the [Interquartile Range scale factor](https://en.wikipedia.org/wiki/Interquartile_range#Outliers), typically set to 1.5.  Set this very large to effectively turn this off.
- `Trail Threshold` - a score to determine how aligned the star directions are.  If the entire frame has star orientations in one direction, this value should be high.  Values above 5 are significant.  Note: this value can be high, even when eccentricity is low since it measures alignment of star elongation, not the strength.


## Acquired Data 
This graph is similar to the one on the Target Planning tab, but selected only for the filters.


## Subframe Data
The scatter graph allows you to compare correlations of different variables in your data.  The available variables are pulled from the FITs header, while some are derived from the star metrics the watchdog calculates when new files are added.  The derived star metric data are aggregated over the subframe.  You can select what variable to plot by the X and Y-axis dropdown options, along with the marker size.  

### Quick Options
These quick access options set the X and Y axes for the subframe data scatter graph

- `FWHM vs. Date` - default option once Monitor Mode is turned on.
- `Eccentricity vs. FWHM`
- `Altitude vs. Azimuth`
- `Star Count vs. Sky Background (ADU)` - useful to check for sky conditions.  Vastly lower star counts indicate either thin clouds, haze or fog.  
- `Focus Position vs. Temperature` - useful to check changes in focus position with different temperatures.  With this data, you can determine the temperature compensation coefficient for your setup.  Also, it can help determine filter offsets.  
- `Sky Background (ADU) vs. Altitude` - helpful to see the effect of different light pollution gradients in your skies.
- `Spacing Metric vs. Star Trailing`
- `Eccentricity vs. Star Trailing`


### Summary Table
This table is an aggregation table over targets, filters, binning, focal length and pixel size.  Total exposure, the number of subframes and star orientation scores are provided.  Additionally, the `CCD-TEMP Dispersion` is given, which is the standard deviation of the CCD temperature.  This can be helpful when preprocessing your data to take care you're subtracting the right darks from your lights.

### Subexposure data
This table shows data at the subframe level, and can include all data available in the scatter graph variables, but also textual data stored in the FITs headers.  Add additional fields in the `Show FITs Header Cols` dropdown.
