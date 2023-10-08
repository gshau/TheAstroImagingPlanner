# -*- mode: python ; coding: utf-8 -*-
import yaml

dir = '../..'
venv = 'venv'

datas = []
datas += [(f'./{dir}/{venv}/lib/python3.11/site-packages/dash_leaflet', './dash_leaflet')]
datas += [(f'./{dir}/{venv}/lib/python3.11/site-packages/rasterio', './rasterio')]
datas += [(f'./{dir}/{venv}/lib/python3.11/site-packages/dash_daq', './dash_daq')]
datas += [(f'./{dir}/{venv}/lib/python3.11/site-packages/pysiril', './pysiril')]
datas += [(f'./{dir}/{venv}/lib/python3.11/site-packages/dash_iconify', './dash_iconify')]
datas += [(f'./{dir}/{venv}/lib/python3.11/site-packages/dash_mantine_components', './dash_mantine_components')]
datas += [(f'./{dir}/assets', './assets')]


datas += [(f'./{dir}/data/_template', './data/_template')]
datas += [(f'./{dir}/data/banner.txt', './data/')]
datas += [(f'./{dir}/data/logs', './data/logs')]

datas += [(f'./{dir}/docs', './docs')]
datas += [(f'./{dir}/git.hash', './')]
datas += [(f'./{dir}/metadata.yml', './')]

hiddenimports = []
hiddenimports += ['sep']
hiddenimports += ['timezonefinder']
hiddenimports += ['astro_planner']
hiddenimports += ['fast_ephemeris']
hiddenimports += ['image_grading']
hiddenimports += ['rasterio._shim']

binaries = []
binaries += [('/usr/local/opt/glib/lib/libglib-2.0.0.dylib', '.')]
binaries += [('/usr/local/opt/gettext/lib/libintl.8.dylib', '.')]


pathex = [f'./{dir}/{venv}/lib/python3.11/site-packages']

a = Analysis([f'{dir}/main.py'],
             pathex=pathex,
             binaries=binaries,
             datas=datas,
             hiddenimports=hiddenimports,
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data)

with open(f'./metadata.yml', 'r') as f:
    metadata = yaml.safe_load(f)
    version = metadata.get('Version')

exe = EXE(pyz,
          a.scripts,
          [],
          version=version,
          exclude_binaries=True,
          name='AstroImagingPlanner',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          upx_exclude=[],
          console=True,
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
               name='AIP')


info_plist = {
            'NSPrincipalClass': 'NSApplication',
            'NSAppleScriptEnabled': False,
            'CFBundleShortVersionString': version,
            'LSUIElement': False,
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'My File Format',
                    'CFBundleTypeIconFile': 'MyFileIcon.icns',
                    'LSItemContentTypes': ['com.example.myformat'],
                    'LSHandlerRank': 'Owner'
                    }
                ]
            }

app = BUNDLE(coll,
             name='AstroImagingPlanner.app',
             icon=f'{dir}/assets/icons/AIP.icns',
             bundle_identifier='com.astroimagingplanner',
             info_plist=info_plist)
