# -*- mode: python ; coding: utf-8 -*-
import yaml

venv = 'venv'

datas = []
datas += [(f'./{venv}/lib/python3.11/site-packages/click', './click')]
datas += [(f'./{venv}/lib/python3.11/site-packages/dash_leaflet', './dash_leaflet')]
datas += [(f'./{venv}/lib/python3.11/site-packages/rasterio', './rasterio')]
datas += [(f'./{venv}/lib/python3.11/site-packages/dash_daq', './dash_daq')]
datas += [(f'./{venv}/lib/python3.11/site-packages/pysiril', './pysiril')]
datas += [(f'./{venv}/lib/python3.11/site-packages/dash_iconify', './dash_iconify')]
datas += [(f'./{venv}/lib/python3.11/site-packages/dash_mantine_components', './dash_mantine_components')]
datas += [('./assets', './assets')]

datas += [('./data/_template', './data/_template')]
datas += [('./data/banner.txt', './data/')]
datas += [('./data/license', './data/license')]
datas += [('./data/logs', './data/logs')]

datas += [('./docs', './docs')]
datas += [('./git.hash', './')]
datas += [('./metadata.yml', './')]

hiddenimports = []
hiddenimports += ['sep']
hiddenimports += ['pandas_access']
hiddenimports += ['rsa']
hiddenimports += ['timezonefinder']
hiddenimports += ['astro_planner']
hiddenimports += ['fast_ephemeris']
hiddenimports += ['image_grading']
hiddenimports += ['rasterio._shim']

binaries = [('/usr/local/bin/mdb-*', '.')]
binaries += [('/usr/local/lib/*mdb*.dylib', '.')]
binaries += [('/usr/local/opt/glib/lib/libglib-2.0.0.dylib', '.')]
binaries += [('/usr/local/opt/gettext/lib/libintl.8.dylib', '.')]
#binaries += [(f'./{venv}/lib/python3.11/site-packages/pyodbc.cpython-311-darwin.so', '.')]


pathex = [f'./{venv}/lib/python3.11/site-packages']

a = Analysis(['main.py'],
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

with open('./metadata.yml', 'r') as f:
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
             icon='assets/icons/AIP.icns',
             bundle_identifier='com.astroimagingplanner',
             info_plist=info_plist)
