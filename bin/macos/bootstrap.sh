#!/bin/bash

# x86_64
export LDFLAGS="-L/usr/local/Homebrew/opt/unixodbc/lib $LDFLAGS"
export CPPFLAGS="-I/usr/local/Homebrew/opt/unixodbc/include $CPPFLAGS"
export PKG_CONFIG_PATH="/usr/local/Homebrew/opt/unixodbc/lib/pkgconfig $PKG_CONFIG_PATH"


/usr/local/Homebrew/bin/python3.11 -m venv venv
venv/bin/pip3 install pip==23.1.2
venv/bin/pip3 install -r requirements.txt
# venv/bin/pip3 uninstall pyodbc
# venv/bin/pip3 install --no-binary :all: pyodbc==4.0.34
arch -x86_64 brew install mdbtools unixodbc