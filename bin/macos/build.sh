#!/bin/bash

# x86_64
export LDFLAGS="-L/usr/local/Homebrew/opt/unixodbc/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/Homebrew/opt/unixodbc/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/Homebrew/opt/unixodbc/lib/pkgconfig $PKG_CONFIG_PATH"


# arm64
# export LDFLAGS="-L/opt/homebrew/Cellar/unixodbc/2.3.11/lib $LDFLAGS"
# export CPPFLAGS="-I/opt/homebrew/Cellar/unixodbc/2.3.11/include $CPPFLAGS"
# export PKG_CONFIG_PATH="/opt/homebrew/Cellar/unixodbc/2.3.11/lib/pkgconfig $PKG_CONFIG_PATH"

VENV='venv'

pyinstaller=./$VENV/bin/pyinstaller
git log -n 1 --pretty=format:"%H" > git.hash
$pyinstaller --distpath ./dist --noconfirm main.spec
xattr -cr ./dist/AstroImagingPlanner.app
# /Users/gshau/astronomy/planner_data/test/roboclip