; -- 64Bit.iss --
; Demonstrates installation of a program built for the x64 (a.k.a. AMD64)
; architecture.
; To successfully run this installation and the program it installs,
; you must have a "x64" edition of Windows.

; SEE THE DOCUMENTATION FOR DETAILS ON CREATING .ISS SCRIPT FILES!

#define AppName "AstroImaging Planner"
#define AppExeName "AstroImagingPlanner.exe"
#define AppVersion GetVersionNumbersString("..\..\dist\AIP\AstroImagingPlanner.exe")
#define AppId "AIP"
#define InstallerMode "Install"

[Setup]
AppId={#AppId}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
VersionInfoVersion={#AppVersion}
WizardStyle=modern
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
; UninstallDisplayIcon={app}\{#AppName}.exe
UninstallDisplayIcon={uninstallexe}
Compression=lzma2
SolidCompression=yes 
; OutputDir=userdocs:Inno Setup Examples Output
; "ArchitecturesAllowed=x64" specifies that Setup cannot run on
; anything but x64.
ArchitecturesAllowed=x64
; "ArchitecturesInstallIn64BitMode=x64" requests that the install be
; done in "64-bit mode" on x64, meaning it should use the native
; 64-bit Program Files directory and the 64-bit view of the registry.
ArchitecturesInstallIn64BitMode=x64
OutputBaseFilename="{#InstallerMode} {#AppName}"
OutputDir="./installers/"
SetupIconFile="..\..\assets\windows\favicon.ico"

[Types]
Name: "onlyAIP"; Description: "{#InstallerMode} The AstroImaging Planner"
;Name: "onlyAccess"; Description: "Install only MS Access Drivers"
;Name: "full"; Description: "{#InstallerMode} The AstroImaging Planner and MS Access Drivers"

[Components]
Name: "aip"; Description: "AstroImaging Planner component"; Types: onlyAIP
;Name: "access"; Description: "Access driver component"; Types: full onlyAccess

[Files]
Source: "..\..\dist\AIP\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs; Components: aip
;Source: "..\..\driver\AccessDatabaseEngine.exe"; DestDir: "{app}\driver"; DestName: access_engine.exe; Check: not IsWin64; Flags: ignoreversion; Components: access
;Source: "..\..\driver\AccessDatabaseEngine_X64.exe"; DestDir: "{app}\driver"; DestName: access_engine.exe; Check: IsWin64; Flags: ignoreversion; Components: access


[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
  

[Icons]
Name: "{group}\My Program"; Filename: "{app}\{#AppName}.exe"; IconFilename: "{app}\assets\windows\favicon.ico"
Name: "{commonprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"; IconFilename: "{app}\assets\windows\favicon.ico"
Name: "{commondesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon; IconFilename: "{app}\assets\windows\favicon.ico"

[Dirs]
Name: {app}; Permissions: users-full

[Run]
;Filename: "{app}\driver\access_engine.exe"; Components: access
