#!/usr/bin/env bash
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
SDK="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
BT="$SDK/build-tools/$(ls "$SDK/build-tools" | sort -V | tail -n1)"
PLAT="$SDK/platforms/$(ls "$SDK/platforms" | sort -V | tail -n1)/android.jar"

BUILD="$DIR/build"
rm -rf "$BUILD/gen" "$BUILD/obj" "$BUILD/res_flat" "$BUILD/dex"
mkdir -p "$BUILD/gen" "$BUILD/obj" "$BUILD/res_flat" "$BUILD/dex"

echo "1/5 Compiling resources (aapt2)..."
"$BT/aapt2" compile --dir "$DIR/res" -o "$BUILD/res_flat/res.zip"
"$BT/aapt2" link \
  -o "$BUILD/unsigned.apk" \
  -I "$PLAT" \
  --manifest "$DIR/AndroidManifest.xml" \
  --java "$BUILD/gen" \
  "$BUILD/res_flat/res.zip"

echo "2/5 Compiling Java (javac)..."
javac --release 11 \
  -classpath "$PLAT" \
  -d "$BUILD/obj" \
  "$DIR/src/com/osmo/face/MainActivity.java" \
  $(find "$BUILD/gen" -name '*.java')

echo "3/5 Converting bytecode to DEX (d8)..."
"$BT/d8" --lib "$PLAT" --min-api 30 --output "$BUILD/dex" $(find "$BUILD/obj" -name '*.class')

echo "4/5 Packaging & aligning APK..."
cp "$BUILD/unsigned.apk" "$BUILD/unaligned.apk"
(cd "$BUILD/dex" && zip -q -u "$BUILD/unaligned.apk" classes.dex)
"$BT/zipalign" -f 4 "$BUILD/unaligned.apk" "$BUILD/aligned.apk"

KEYSTORE="$BUILD/debug.keystore"
if [ ! -f "$KEYSTORE" ]; then
  keytool -genkeypair -v \
    -keystore "$KEYSTORE" \
    -storepass android -keypass android \
    -alias osmo -keyalg RSA -keysize 2048 -validity 10000 \
    -dname "CN=OsmoFace,O=Hackathon,C=FR" >/dev/null 2>&1
fi

echo "5/5 Signing APK..."
"$BT/apksigner" sign \
  --ks "$KEYSTORE" \
  --ks-pass pass:android \
  --key-pass pass:android \
  --out "$BUILD/OsmoFace.apk" \
  "$BUILD/aligned.apk"

echo "Built: $BUILD/OsmoFace.apk ($(du -h "$BUILD/OsmoFace.apk" | awk '{print $1}'))"
