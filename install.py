#!/usr/bin/env python3
"""
Game Analytics Platform — Cross-Platform Installer
Usage: python install.py
Tested on: Linux, macOS, Windows
"""

import os
import sys
import shutil
import platform
import subprocess

PLATFORM = platform.system()
ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Colour & Symbol helpers (ASCII fallback on Windows) ────────────────────
def green(s):  return f"\033[92m{s}\033[0m" if PLATFORM != "Windows" else s
def red(s):    return f"\033[91m{s}\033[0m" if PLATFORM != "Windows" else s
def yellow(s): return f"\033[93m{s}\033[0m" if PLATFORM != "Windows" else s
def bold(s):   return f"\033[1m{s}\033[0m"  if PLATFORM != "Windows" else s

SYMBOL_OK   = green("✓") if PLATFORM != "Windows" else "[OK]"
SYMBOL_FAIL = red("✗")   if PLATFORM != "Windows" else "[FAIL]"
SYMBOL_WARN = yellow("⚠") if PLATFORM != "Windows" else "[WARN]"
SYMBOL_INFO = "→"        if PLATFORM != "Windows" else "->"
SYMBOL_DONE = "✅"       if PLATFORM != "Windows" else "OK"

def ok(msg):   print(f"  {SYMBOL_OK} {msg}")
def fail(msg): print(f"  {SYMBOL_FAIL} {msg}"); sys.exit(1)
def warn(msg): print(f"  {SYMBOL_WARN} {msg}")
def info(msg): print(f"  {SYMBOL_INFO} {msg}")

def run(cmd, cwd=None, check=True):
    """Run a shell command, streaming output live."""
    info(cmd if isinstance(cmd, str) else " ".join(cmd))
    result = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, check=check)
    return result.returncode == 0

def remove_directory(path):
    """Robustly remove a directory, handling Windows read-only/permission issues."""
    if not os.path.exists(path):
        return

    def handle_errors(func, path, exc_info):
        import stat
        # If the error is due to read-only, change permission and retry
        try:
            os.chmod(path, stat.S_IWUSR)
            func(path)
        except:
            # If it still fails, just ignore it and let the main loop handle it
            pass

    try:
        shutil.rmtree(path, onerror=handle_errors)
    except Exception as e:
        # On Windows, sometimes files are locked. Try renaming as a fallback.
        try:
            temp_path = path + "_old_" + str(os.getpid())
            os.rename(path, temp_path)
            warn(f"Directory {path} was locked; renamed to {temp_path}. Please delete it manually later.")
        except:
            fail(f"Permission denied: Could not remove {path}. Ensure no other process (like the Java backend) is using it.")


# ── Step helpers ──────────────────────────────────────────────────────────
def check_java():
    print(bold("\n[1/6] Checking Java 17+"))
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT).decode()
        # Extract version number
        import re
        match = re.search(r'version "(\d+)', out)
        version = int(match.group(1)) if match else 0
        if version < 17:
            fail(f"Java {version} found but Java 17+ is required. Install JDK 17 and retry.")
        ok(f"Java {version} found")
    except FileNotFoundError:
        fail("Java not found. Install JDK 17+ from https://adoptium.net/ and retry.")


def check_node():
    print(bold("\n[2/6] Checking Node.js"))
    try:
        out = subprocess.check_output(["node", "--version"]).decode().strip()
        ok(f"Node.js {out} found")
    except FileNotFoundError:
        fail("Node.js not found. Install from https://nodejs.org/ and retry.")


def setup_python_venv():
    print(bold("\n[3/6] Setting up Python virtual environment"))
    venv_dir = os.path.join(ROOT, "venv")

    if os.path.exists(venv_dir):
        ok("venv already exists — skipping creation")
    else:
        run([sys.executable, "-m", "venv", "venv"], cwd=ROOT)
        ok("venv created")

    # Determine pip path
    if PLATFORM == "Windows":
        pip = os.path.join(ROOT, "venv", "Scripts", "pip")
    else:
        pip = os.path.join(ROOT, "venv", "bin", "pip")

    req_file = os.path.join(ROOT, "requirements.txt")
    if os.path.exists(req_file):
        run([pip, "install", "-r", req_file], cwd=ROOT)
        ok("Python dependencies installed")
    else:
        warn("requirements.txt not found — skipping pip install")


def build_frontend():
    print(bold("\n[4/6] Building React frontend"))
    frontend_dir = os.path.join(ROOT, "frontend")

    if not os.path.exists(frontend_dir):
        fail("frontend/ directory not found. Repo may be incomplete.")

    # Install npm deps if needed
    if not os.path.exists(os.path.join(frontend_dir, "node_modules")):
        run("npm install", cwd=frontend_dir)
        ok("npm dependencies installed")
    else:
        ok("node_modules already present")

    # Build
    run("npm run build", cwd=frontend_dir)
    ok("React production build complete (frontend/dist/)")

    # Copy dist → Spring Boot static
    dist_dir    = os.path.join(frontend_dir, "dist")
    static_dir  = os.path.join(ROOT, "backend", "src", "main", "resources", "static")

    remove_directory(static_dir)
    shutil.copytree(dist_dir, static_dir)
    ok("React build copied -> backend/src/main/resources/static/")


def build_backend():
    print(bold("\n[5/6] Building Spring Boot JAR"))
    backend_dir = os.path.join(ROOT, "backend")

    if PLATFORM == "Windows":
        mvnw = os.path.join(backend_dir, "mvnw.cmd")
        cmd  = f'"{mvnw}" clean package -DskipTests'
    else:
        mvnw = os.path.join(backend_dir, "mvnw")
        os.chmod(mvnw, 0o755)
        cmd  = f'"{mvnw}" clean package -DskipTests'

    # Fallback to system mvn if mvnw not present
    if not os.path.exists(mvnw.replace('"', '')):
        warn("mvnw not found — falling back to system mvn")
        cmd = "mvn clean package -DskipTests"

    run(cmd, cwd=backend_dir)
    ok("Spring Boot JAR built -> backend/target/game-analytics-0.0.1-SNAPSHOT.jar")


def create_launchers():
    print(bold("\n[6/6] Creating launcher scripts"))
    jar_path = os.path.join("backend", "target", "game-analytics-0.0.1-SNAPSHOT.jar")
    x_mark = "X" if PLATFORM == "Windows" else "❌"

    # Linux / macOS
    sh_path = os.path.join(ROOT, "start.sh")
    with open(sh_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write("# Game Analytics Platform — launcher\n")
        f.write('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n')
        f.write(f'JAR="$SCRIPT_DIR/{jar_path}"\n\n')
        f.write('if [ ! -f "$JAR" ]; then\n')
        f.write(f'    echo "{x_mark} JAR not found. Run: python install.py first."\n')
        f.write('    exit 1\n')
        f.write('fi\n\n')
        f.write('echo "Starting Game Analytics Platform..."\n')
        f.write('echo "   -> Open http://localhost:8080 in your browser"\n')
        f.write('echo "   -> Press Ctrl+C to stop"\n')
        f.write('echo ""\n')
        f.write('java -jar "$JAR"\n\n')
        f.write('EXIT_CODE=$?\n')
        f.write('if [ $EXIT_CODE -ne 0 ]; then\n')
        f.write(f'    echo "{x_mark} Java backend crashed with exit code $EXIT_CODE."\n')
        f.write('fi\n')
    os.chmod(sh_path, 0o755)
    ok("start.sh created")

    # Windows
    bat_path = os.path.join(ROOT, "start.bat")
    with open(bat_path, "w", newline="\r\n", encoding="utf-8") as f:
        f.write("@echo off\n")
        f.write("REM Game Analytics Platform — launcher\n")
        f.write('SET SCRIPT_DIR=%~dp0\n')
        f.write(f'SET JAR=%SCRIPT_DIR%{jar_path.replace("/", chr(92))}\n')
        f.write('echo Starting Game Analytics Platform...\n')
        f.write('echo Open http://localhost:8080 in your browser\n')
        f.write('echo Press Ctrl+C to stop\n')
        f.write('java -jar "%JAR%"\n')
        f.write('if %ERRORLEVEL% NEQ 0 (\n')
        f.write(f'    echo {x_mark} Java backend crashed with exit code %ERRORLEVEL%\n')
        f.write(')\n')
        f.write('pause\n')
    ok("start.bat created")


def setup_properties():
    print(bold("\n[7/7] Configuring application properties"))
    props_path = os.path.join(ROOT, "backend", "src", "main", "resources", "application.properties")
    if os.path.exists(props_path):
        with open(props_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Inject absolute path for project.root
        # Use forward slashes even on Windows for Java compatibility
        safe_root = ROOT.replace("\\", "/")
        import re
        content = re.sub(r'^project\.root=.*$', f'project.root={safe_root}', content, flags=re.MULTILINE)
        
        with open(props_path, "w", encoding="utf-8") as f:
            f.write(content)
        ok(f"Injected project.root = {safe_root}")
    else:
        warn("application.properties not found")


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(bold("=" * 55))
    print(bold("  Game Analytics Platform — Installer"))
    print(bold("=" * 55))

    check_java()
    check_node()
    setup_python_venv()
    build_frontend()
    build_backend()
    create_launchers()
    setup_properties()

    print(bold("\n" + "=" * 55))
    print(green(f"  {SYMBOL_DONE}  Installation complete!"))
    print(bold("=" * 55))
    print()
    if PLATFORM == "Windows":
        print("  Run:  start.bat")
    else:
        print("  Run:  ./start.sh")
    print("  Then open:  http://localhost:8080")
    print()
