                                                

## Directory Settings
- `Target Directory` - where all target data is located - either Voyager Roboclip database, or SGP/NINA sequence files
- `Raw FITs Directory` - where all raw light subframes are located
- `Calibration FITs Directory` - where all calibration subframes are located
- `Preprocessed Output Directory` - where preprocessing output will be stored

## FITs File Processing Settings
- `Thread Count` - number of CPU threads used to process subframes
- `Save Settings` - saves current settings into config file

## Profile List
A list of Voyager profiles or SGP/NINA files to show in planning tab
## Connect With Voyager 
If on, allows for syncing accpet/reject status of subframes with Voyager Advanced.  
- `Voyager Connection` - list of inputs required to connect with Voyager
  - `Voyager Hostname` - ip address for Voyager
  - `Voyager Port` - port for Voyager connection, typically 5950
  - `Voyager User` - username
  - `Voyager Password` - password
## Silence Alerts
If on, prevents notification box showing newly processed/removed files in the app's database.


## Utilities
[//]: # (src/assets/utilities_tab.png)
The utilities tab gives some control over some portions of the app.  
 - `Download App Log` - download the log
 - `Download Target Data` - downloads target data
 - `Download Target Status` - downloads target status
 - `Download FITs Data Tables` - downloads all FITs data 
 
 
Note: The FITs header must contain the following entries to be processed correctly:
 ```
  "OBJECT"
  "DATE-OBS"
  "CCD-TEMP"
  "FILTER"
  "OBJCTRA"
  "OBJCTDEC"
  "OBJCTALT"
  "INSTRUME"
  "FOCALLEN"
  "EXPOSURE"
  "XBINNING"
  "XPIXSZ"
```
