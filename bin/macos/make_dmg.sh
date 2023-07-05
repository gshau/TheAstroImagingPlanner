#!/bin/bash


VERSION=$(grep 'Version' metadata.yml| awk '{print $2}' )
FINAL_DMG=./AstroImagingPlanner.dmg
TMP_DMG=./tmp.dmg

rm -f $FINAL_DMG

APP_FILE="./dist/AstroImagingPlanner.app"
xattr -cr $APP_FILE
hdiutil create $TMP_DMG -ov -volname "AstroImagingPlanner" -fs HFS+ -srcfolder $APP_FILE
hdiutil convert $TMP_DMG -format UDZO -o $FINAL_DMG
mv $FINAL_DMG ./dist/

# https://github.com/create-dmg/create-dmg