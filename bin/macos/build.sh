#!/bin/bash

# x86_64
export LDFLAGS="-L/usr/local/Homebrew/opt/unixodbc/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/Homebrew/opt/unixodbc/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/Homebrew/opt/unixodbc/lib/pkgconfig $PKG_CONFIG_PATH"

VENV='venv'

pyinstaller=./$VENV/bin/pyinstaller
git log -n 1 --pretty=format:"%H" > git.hash
$pyinstaller --distpath ./dist --noconfirm ./build_assets/macos/main.spec
xattr -cr ./dist/AstroImagingPlanner.app