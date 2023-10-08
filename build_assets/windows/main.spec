# -*- mode: python ; coding: utf-8 -*-
import os, glob

dir = '..\..'

datas = []
datas += [(f"{dir}/venv/Lib/site-packages/dash_leaflet", "dash_leaflet")]
datas += [(f'{dir}/venv/Lib/site-packages/rasterio', './rasterio')]
datas += [(f"{dir}/venv/Lib/site-packages/pysiril", "pysiril")]
datas += [(f"{dir}/venv/Lib/site-packages/dash_daq", "dash_daq")]
datas += [(f"{dir}/venv/Lib/site-packages/dash_iconify", './dash_iconify')]
datas += [(f"{dir}/venv/Lib/site-packages/dash_mantine_components", './dash_mantine_components')]
datas += [(f"{dir}/assets", "assets")]

datas += [(f'{dir}/data/_template', './data/_template')]
datas += [(f'{dir}/data/banner.txt', './data/')]
datas += [(f'{dir}/data/logs', './data/logs')]

datas += [(f'./{dir}/docs', './docs')]
datas += [(f'./{dir}/git.hash', './')]
datas += [(f'./{dir}/metadata.yml', './')]

hiddenimports = []
hiddenimports += ["sep"]
hiddenimports += ["timezonefinder"]
hiddenimports += ["astro_planner"]
hiddenimports += ["fast_ephemeris"]
hiddenimports += ["image_grading"]
hiddenimports += ["rasterio._shim"]

rasterio_imports_paths = glob.glob(f'{dir}/venv/Lib/site-packages/rasterio/*.py')
for item in rasterio_imports_paths:
    current_module_filename = os.path.split(item)[-1]
    current_module_filename = 'rasterio.'+current_module_filename.replace('.py', '')
    hiddenimports.append(current_module_filename)


binaries = []

pathex = [f"./{dir}/venv/lib/python3.11/site-packages"]

a = Analysis([f"{dir}/main.py"],
             pathex=pathex,
             binaries=binaries,
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             #hooksconfig={},
             runtime_hooks=[],
             excludes=['tkinter'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(pyz,
          a.scripts,
          [],
          version=f"{dir}/file_version_info.txt",
          exclude_binaries=True,
          name="AstroImagingPlanner",
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          console=True,
          icon=f'{dir}/assets/windows/favicon.ico',
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
             icon=f"{dir}/assets/icons/AIP.icns",
             #bundle_identifier="com.astroimagingplanner",
             )

