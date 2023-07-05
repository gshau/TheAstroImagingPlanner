# -*- mode: python ; coding: utf-8 -*-
import os, glob

datas = []
datas += [("venv\Lib\site-packages\dash_leaflet", "dash_leaflet")]
datas += [("venv\Lib\site-packages\pysiril", "pysiril")]
datas += [("venv\Lib\site-packages\dash_daq", "dash_daq")]
#datas += [("venv\Lib\site-packages\GDAL-3.3.3.dist-info", "GDAL-3.3.3.dist-info")]
#datas += [("venv\Lib\site-packages\rasterio", "rasterio")]
datas += [("venv\Lib\site-packages\dash_iconify", './dash_iconify')]
datas += [("venv\Lib\site-packages\dash_mantine_components", './dash_mantine_components')]
datas += [("assets", "assets")]
#datas += [("data", "data")]

datas += [('./data/_template', './data/_template')]
datas += [('./data/banner.txt', './data/')]
datas += [('./data/license', './data/license')]
datas += [('./data/logs', './data/logs')]
#datas += [('./data/sky_atlas', './data/sky_atlas')]

datas += [("docs", "docs")]
datas += [("./git.hash", "./")]
datas += [("./metadata.yml", "./")]

hiddenimports = []
hiddenimports += ["sep"]
hiddenimports += ["pandas_access"]
hiddenimports += ["rsa"]
hiddenimports += ["timezonefinder"]
hiddenimports += ["astro_planner"]
hiddenimports += ["fast_ephemeris"]
hiddenimports += ["image_grading"]
hiddenimports += ["rasterio._shim"]

rasterio_imports_paths = glob.glob(r'venv\Lib\site-packages\rasterio\*.py')
for item in rasterio_imports_paths:
    current_module_filename = os.path.split(item)[-1]
    current_module_filename = 'rasterio.'+current_module_filename.replace('.py', '')
    hiddenimports.append(current_module_filename)


binaries = []

pathex = ["./venv/lib/python3.8/site-packages"]

a = Analysis(["main.py"],
             pathex=pathex,
             binaries=binaries,
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             #hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(pyz,
          a.scripts,
          [],
          version="file_version_info.txt",
          exclude_binaries=True,
          name="AstroImagingPlanner",
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          console=True,
          icon='assets\\windows\\favicon.ico',
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None )

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas, 
               strip=False,
               upx=True,
               upx_exclude=[],
               name="AIP")


app = BUNDLE(coll,
             name="AstroImagingPlanner.exe",
             icon="assets/icons/AIP.icns",
             #bundle_identifier="com.astroimagingplanner",
             )

