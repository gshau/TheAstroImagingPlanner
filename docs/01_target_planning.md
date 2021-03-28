## Target Planning


[//]: # (src/assets/planner_tab.png)
### Target Graph
The main target graph shows the data over the course of a night.  This graph, like others in the app, is interactive, and allows you to deselect targets by clicking them in the legend.  Double-click to show only that target.  You can also zoom in/out.  Each target curve has a popup of information.  This information includes the date, target name, moon distance and approximate sky brightness at the target.

### Acquired Data Graph
This graph shows how much data has been acquried so far for each target, and partitioned by filter used.  The data is also auto-graded based on the settings in the Frame Inspector.  Auto-rejected data based on those settings show up in the negative side of the graph, while accepted data shows up in the positive side of the graph.  The total bar length indicates the total amount of data for that target and filter.  

### Settings
-  **Date & Location**: Once a date or location has changed, all target data is updated, including altitude/azimuth over time, local sky brightness, contrast, etc.  Location can be set by clicking a location on the map.  Local sky brightness (SQM) and Bortle Scale data are also updated


-  **Group**: This defines what targets you'd like to view.  For Voyager, this is the group in the RoboClip database.  For SGP or NINA, it is the name of the sequence files.


-  **Quantity to plot**: Options here include:
    - `Altitude` - Target altitude
    - `Airmass` - Target airmass - how thick the atmosphere, relative to zenith, when viewing target.
    - `Sky Brightness (Experimental)` - Estimated sky background in magnitudes per square arcsecond at the target's location in the sky.  
    - `Relative Contrast (Experimental)` - Contrast reduction from ideal, given the best dark skies at your location.  The contrast reduction is a combination of increased sky brightness due to the moon or additional light pollution away from zenith, and of decreased SNR as the airmass increases with lower altitudes.  This is not a perfect measure of contrast, but I think it does encapsulate some of the effects going into it.


-  **Matching Filters in Notes**: This section allows you to filter the targets shown in the main target graph.  For Voyager RoboClip databases, the data pulled from the stored Notes are checked with filter strings.  Filtering options with their matching strings include the following using case insensitive string matching
    - `Narrowband`: "ha", "oiii", "sii", "sho", "ho", "hoo", "hos", "halpha", "h-alpha"
    - `Broadband`: "luminance", "lrgb", "lum"
    - `Luminance`: "luminance", "lum"
    - `RGB`: "osc", "bayer", "dslr", "slr", r", "g", "b", "rgb"
    - `Ha`: "ho", "sho", "hoo", "hos", "halpha", "h-alpha"
    - `OIII`: "ho", "sho", "hoo", "hos"
    - `SII`: "sho", "hos"
    

-  **Selected Target Status**: This option allows you to filter the targets shown in both the target graph and acquired data bar graph based on their status.  The status options are defined as:
    - `Pending`: targets not yet imaged
    - `Active`: targets currently being worked on
    - `Acquired`: targets with enough data to process
    - `Closed`: targets with finished final images


-  **Display Only Seasonal Targets**: This option allows you to only show seasonal targets in the main target graph.  Seasonal targets are defined as having a transit occur, or the maximum altitude of the target is above 60 degrees during night hours.
 

-  **Change Target Status**: This dropdown allows you to select the status for various targets.  You can specify multiple targets and change their status with the radio buttons below the dropdown.  

                                                
- **Other settings**
    - `Minimum Frame Overlap Fraction` - this controls how close the RA/Dec of the target is to the center of the frame for stored data.  The fraction is the distance from the target RA/Dec to the center of the frame, relative to the half the vertical height of the frame.  
    - `Minimum Moon Distance` - minimum distance from target to the Moon.  Useful for filtering targets that would have strong gradients and light pollution from the Moon
    - `Extinction Coefficient` - atmospheric extinction coefficient.  Used in the calculation of contrast and sky brightness.  Values less than 0.2 indicate exceptionally transparent skies.  
    - `Sky Brightness (mpsas)` - local sky brightness in magnitudes per square arc-second (mpsas).  This is extracted from [The new world atlas of artificial night sky brightness](https://doi.org/10.1126/sciadv.1600377), and is updated on a location change.  This value can also be overridden.

### Show Weather Forecast
Clicking on this button displays a modal that shows weather forecasts for the location.  Sources currently are [ClearOutside](http://clearoutside.com/) and [NWS](https://www.weather.gov/).

[//]: # (src/assets/weather_forecast_modal.png)