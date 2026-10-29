#!/bin/bash
# Builds "Lamin Mesh Analyzer.app" (self-contained: the app code is copied inside the bundle).
# Usage: ./make_app.sh [destination folder]   (default: this folder)
# Re-run after editing app.py / lamin_core.py so the icon uses the new code.
set -e
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="${1:-$SRC}"
APP="$DEST/Lamin Mesh Analyzer.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/app"
cp "$SRC/app.py" "$SRC/lamin_core.py" "$SRC/meshwork.py" "$SRC/ui.py" "$SRC/lifecycle.py" "$SRC/serve.py" "$SRC/theme.py" "$SRC/requirements.txt" "$APP/Contents/Resources/app/"
cp -R "$SRC/example" "$SRC/assets" "$SRC/.streamlit" "$APP/Contents/Resources/app/"
rm -f "$APP/Contents/Resources/app/example/"*.pdf

# icon: the author's own (assets/icon1024.png, not in the public repository) or the public one
ICON_SRC="$SRC/assets/icon1024.png"; [ -f "$ICON_SRC" ] || ICON_SRC="$SRC/assets/icon_public.png"
ICONSET="$(mktemp -d)/AppIcon.iconset"; mkdir -p "$ICONSET"
for s in 16 32 128 256 512; do
  sips -z $s $s "$ICON_SRC" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) "$ICON_SRC" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Lamin Mesh Analyzer</string>
  <key>CFBundleDisplayName</key><string>Lamin Mesh Analyzer</string>
  <key>CFBundleIdentifier</key><string>local.lamin-mesh-analyzer</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>launcher</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>11.0</string>
</dict></plist>
PLIST

cat > "$APP/Contents/MacOS/launcher" <<'LAUNCH'
#!/bin/bash
# Starts the analyzer server (if not already running) and opens it in the default browser.
# The app quits by itself when its browser tab is closed (lifecycle.py); this script then ends and the icon
# leaves the Dock. If this script is quit or force-quit from the Dock, the server notices and stops too.
APPDIR="$(cd "$(dirname "$0")/../Resources/app" && pwd)"
PORT=8599
URL="http://localhost:$PORT"
LOG="$HOME/Library/Logs/LaminMeshAnalyzer.log"
alive() { curl -s -m 1 "$URL/_stcore/health" | grep -q ok; }

if alive; then open "$URL"; exit 0; fi

CANDIDATES=(/opt/anaconda3/bin/python3 "$HOME/anaconda3/bin/python3" "$HOME/opt/anaconda3/bin/python3"
            "$HOME/miniconda3/bin/python3" "$HOME/miniforge3/bin/python3" /opt/homebrew/bin/python3
            /usr/local/bin/python3 /Library/Frameworks/Python.framework/Versions/Current/bin/python3)
xcode-select -p >/dev/null 2>&1 && CANDIDATES+=(/usr/bin/python3)

PY=""
for p in "${CANDIDATES[@]}"; do
  [ -x "$p" ] && "$p" -c "import streamlit, skimage, tifffile, plotly, matplotlib, pandas, scipy" 2>/dev/null && { PY="$p"; break; }
done
if [ -z "$PY" ]; then
  BASE=""; for p in "${CANDIDATES[@]}"; do [ -x "$p" ] && { BASE="$p"; break; }; done
  if [ -z "$BASE" ]; then
    osascript -e 'display alert "Python 3 not found" message "Lamin Mesh Analyzer needs Python 3. Install Anaconda (anaconda.com) or Python from python.org, then open the app again."'
    exit 1
  fi
  ans=$(osascript -e 'button returned of (display dialog "Lamin Mesh Analyzer needs a few Python packages (streamlit, scikit-image, …). Install them now? This takes 1–2 minutes." buttons {"Cancel", "Install"} default button "Install" with title "Lamin Mesh Analyzer")' 2>/dev/null)
  [ "$ans" = "Install" ] || exit 0
  osascript -e 'display notification "Installing Python packages…" with title "Lamin Mesh Analyzer"'
  if ! "$BASE" -m pip install --user -r "$APPDIR/requirements.txt" > "$LOG" 2>&1; then
    osascript -e 'display alert "Package installation failed" message "Details: ~/Library/Logs/LaminMeshAnalyzer.log"'
    exit 1
  fi
  PY="$BASE"
fi

osascript -e 'display notification "Starting… your browser will open in a few seconds." with title "Lamin Mesh Analyzer"'
cd "$APPDIR"
echo "[$(date '+%F %T')] starting with $PY" >> "$LOG"
# The server watches this launcher (by pid): quitting the app from the Dock stops the server as well.
LAMIN_MESH_LAUNCHER_PID=$$ "$PY" -W ignore serve.py --port $PORT --headless >> "$LOG" 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null' EXIT INT TERM
for i in $(seq 1 120); do
  if alive; then
    open "$URL"
    wait $PID          # the icon stays in the Dock while the app runs, and leaves when it quits
    exit 0
  fi
  kill -0 $PID 2>/dev/null || break
  sleep 0.5
done
osascript -e 'display alert "Lamin Mesh Analyzer did not start" message "Details: ~/Library/Logs/LaminMeshAnalyzer.log"'
exit 1
LAUNCH
chmod +x "$APP/Contents/MacOS/launcher"
touch "$APP"
echo "Built: $APP"
