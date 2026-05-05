@echo off
REM Game Analytics Platform — launcher
SET SCRIPT_DIR=%~dp0
SET JAR=%SCRIPT_DIR%backend\target\game-analytics-0.0.1-SNAPSHOT.jar
echo Starting Game Analytics Platform...
echo Open http://localhost:8080 in your browser
echo Press Ctrl+C to stop
java -jar "%JAR%"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Java backend crashed with exit code %ERRORLEVEL%
)
pause
