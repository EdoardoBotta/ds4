#!/bin/bash
# Builds DS4MacApp.app, a proper app bundle (with the DS4 icon) around the
# swift build output, so the app has a Dock/Finder icon instead of running
# as a bare `swift run` executable.
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${1:-release}"
swift build -c "$CONFIG"

APP="DS4MacApp.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cp ".build/$CONFIG/DS4MacApp" "$APP/Contents/MacOS/DS4MacApp"
cp "Info.plist" "$APP/Contents/Info.plist"
cp "AppIcon.icns" "$APP/Contents/Resources/AppIcon.icns"

touch "$APP"
echo "Built $APP"
