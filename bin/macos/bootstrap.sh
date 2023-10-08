#!/bin/bash

# x86_64
export LDFLAGS="-L/usr/local/Homebrew/opt/unixodbc/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/Homebrew/opt/unixodbc/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/Homebrew/opt/unixodbc/lib/pkgconfig $PKG_CONFIG_PATH"


/usr/local/homebrew/bin/python3.11 -m venv venv
# /opt/homebrew/bin/python3.10 -m venv venv
venv/bin/pip3 install pip==23.1.2
venv/bin/pip3 install -r requirements.txt
arch -x86_64 /usr/local/homebrew/bin/brew install mdbtools unixodbc
# arch -arm64 brew install mdbtools unixodbc