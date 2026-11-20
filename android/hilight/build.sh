#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
ANDROID_JAR="$SDK/platforms/android-37.0/android.jar"
D8="$SDK/build-tools/37.0.0/d8"

javac --release 11 -cp "$ANDROID_JAR" HiLightBridge.java
"$D8" --min-api 31 --output hilight.jar HiLightBridge.class
rm -f HiLightBridge.class
echo "Built $(pwd)/hilight.jar ($(wc -c < hilight.jar) bytes)"
