#!/bin/bash
# Game Analytics Platform — launcher
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JAR="$SCRIPT_DIR/backend/target/game-analytics-0.0.1-SNAPSHOT.jar"

if [ ! -f "$JAR" ]; then
    echo "❌ JAR not found. Run: python install.py first."
    exit 1
fi

echo "🎮 Starting Game Analytics Platform..."
echo "   → Open http://localhost:8080 in your browser"
echo "   → Press Ctrl+C to stop"
echo ""
java -jar "$JAR"

EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Java backend crashed with exit code $EXIT_CODE."
fi
