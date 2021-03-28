                                                
## Utilities
[//]: # (src/assets/utilities_tab.png)
The utilities tab gives some control over some portions of the app.  
 - `Clear All Tables` - clears all tables (restart from scratch)
 - `Clear Targets Table` - clears only the target table
 - `Clear Header Tables` - clears FITs header tables including status
 - `Clear Star Tables` - clears all star metrics data
 - `Download Planner Log` - download the log for the planner
 - `Download Watchdog Log` - download the log for all backend work the watchdog does, including fits header reading, star metric eval, etc.
 - `Restart App` - restart the main app
 - `Restart Watchdog` - restart the watchdog
 - `Show File Skiplist` - Show a list of files the watchdog skipped due to an error (most likely a key missing FITs header - see below for a list of required header entries). 
 - `Clear File Skiplist` - Clear the skiplist to allow for those files to be reprocessed

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
