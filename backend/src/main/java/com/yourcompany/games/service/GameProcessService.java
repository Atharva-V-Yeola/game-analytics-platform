package com.yourcompany.games.service;

import com.yourcompany.games.model.Game;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

@Service
public class GameProcessService {

    // Python executable path from application.properties
    @Value("${python.path:python}")
    private String pythonPath;

    // Project root directory (where games/ and configs/ live)
    @Value("${project.root:}")
    private String projectRoot;

    // Run Python scripts without OpenCV windows (true = safe default for server/web use)
    // Set game.headless=false in application.properties only if running on a desktop
    // with a display and you want to see the camera windows.
    @Value("${game.headless:true}")
    private boolean gameHeadless;

    // Registry of running processes: gameId -> Process
    private final Map<String, Process> runningProcesses = new ConcurrentHashMap<>();

    // Watchdog executor to auto-kill hung processes
    private final ScheduledExecutorService watchdog = Executors.newScheduledThreadPool(1);

    /**
     * Start a game by launching its Python script.
     */
    public synchronized boolean startGame(Game game) {
        if (runningProcesses.containsKey(game.getId())) {
            return false; // Already running
        }

        try {
            // Resolve working directory
            File workingDir = resolveWorkingDirectory();

            // Resolve OS-specific python executable
            String resolvedPythonPath = resolvePythonPath(workingDir);

            // Derive config path from script path (e.g., games/exe2_balance.py -> configs/exe2_balance.json)
            String configPath = game.getScriptPath()
                .replace("games/", "configs/")
                .replace(".py", ".json");

            // Build command: python games/exe1_yolo_ball.py --game-id exe1 --camera 0 ...
            // --headless prevents cv2.imshow() windows and skips interactive setup screens.
            // This is required for reliable server operation; set game.headless=false in
            // application.properties only when running on a local desktop with a display.
            java.util.List<String> cmd = new java.util.ArrayList<>(java.util.Arrays.asList(
                resolvedPythonPath,
                game.getScriptPath(),
                "--game-id", game.getId(),
                "--camera", "0",
                "--output", "./data",
                "--config", configPath
            ));
            if (gameHeadless) {
                cmd.add("--headless");
            }
            ProcessBuilder pb = new ProcessBuilder(cmd);

            pb.directory(workingDir);
            pb.redirectErrorStream(true); // Merge stderr into stdout

            log("[" + game.getId() + "] Starting: " + String.join(" ", pb.command()));

            Process process = pb.start();
            runningProcesses.put(game.getId(), process);

            // Start stdout reader thread
            Thread reader = new Thread(() -> readProcessOutput(game.getId(), process));
            reader.setDaemon(true);
            reader.start();

            // Schedule watchdog: kill after 30 minutes max
            watchdog.schedule(() -> {
                if (process.isAlive()) {
                    log("[" + game.getId() + "] Watchdog timeout, forcing stop");
                    process.destroyForcibly();
                }
            }, 30, TimeUnit.MINUTES);

            return true;

        } catch (IOException e) {
            log("[" + game.getId() + "] Failed to start: " + e.getMessage());
            return false;
        }
    }

    /**
     * Stop a game by destroying its process.
     */
    public synchronized boolean stopGame(String gameId) {
        Process process = runningProcesses.get(gameId);
        if (process == null) {
            return false;
        }

        log("[" + gameId + "] Stopping process...");

        // Try graceful first (SIGTERM)
        process.destroy();

        // Wait up to 5 seconds for graceful exit
        boolean exited = false;
        try {
            exited = process.waitFor(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Force kill if still alive
        if (!exited && process.isAlive()) {
            log("[" + gameId + "] Graceful stop failed, forcing...");
            process.destroyForcibly();
        }

        runningProcesses.remove(gameId);
        return true;
    }

    /**
     * Check if a game is currently running.
     */
    public boolean isRunning(String gameId) {
        Process process = runningProcesses.get(gameId);
        return process != null && process.isAlive();
    }

    /**
     * Read and log Python process stdout.
     */
    private void readProcessOutput(String gameId, Process process) {
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(process.getInputStream()))) {

            String line;
            while ((line = reader.readLine()) != null) {
                log("[" + gameId + "] " + line);

                // Optional: Parse JSON logs for structured monitoring
                if (line.contains("\"event\":\"ERROR\"")) {
                    log("[" + gameId + "] PYTHON ERROR DETECTED");
                }
            }

        } catch (IOException e) {
            log("[" + gameId + "] Output read error: " + e.getMessage());
        } finally {
            // Process ended (gracefully or crashed)
            // Remove ONLY if this exact process is still mapped to this gameId
            runningProcesses.remove(gameId, process);
            try {
                process.waitFor(2, TimeUnit.SECONDS);
                log("[" + gameId + "] Process ended with exit code: " + process.exitValue());
            } catch (Exception e) {
                log("[" + gameId + "] Process ended. Exit code unavailable.");
            }
        }
    }

    /**
     * Resolve the working directory for Python subprocess.
     */
    public File resolveWorkingDirectory() {
        if (projectRoot != null && !projectRoot.isEmpty()) {
            File dir = new File(projectRoot);
            if (dir.exists() && dir.isDirectory()) {
                return dir;
            }
        }

        // Start with the current working directory
        File currentDir = new File(System.getProperty("user.dir"));

        // Check if current directory has the necessary folders (games, configs)
        if (new File(currentDir, "games").exists() && new File(currentDir, "configs").exists()) {
            return currentDir;
        }

        // If running from IDE (e.g., inside 'backend/' folder), go up one level
        File parentDir = currentDir.getParentFile();
        if (parentDir != null && new File(parentDir, "games").exists() && new File(parentDir, "configs").exists()) {
            return parentDir;
        }

        // Fallback
        return currentDir;
    }

    /**
     * Dynamically resolve python path based on OS and venv presence.
     */
    private String resolvePythonPath(File workingDir) {
        if (pythonPath != null && !pythonPath.isEmpty() && !pythonPath.equals("python")) {
            // Check if it's the hardcoded absolute linux path, ignore it if we are on windows
            if (pythonPath.startsWith("/") && System.getProperty("os.name").toLowerCase().contains("win")) {
                // Ignore hardcoded linux path
            } else {
                File configuredPath = new File(pythonPath);
                if (configuredPath.exists()) {
                    return pythonPath;
                }
            }
        }

        // Check for local venv
        boolean isWindows = System.getProperty("os.name").toLowerCase().contains("win");
        File venvPython;
        if (isWindows) {
            venvPython = new File(workingDir, "venv\\Scripts\\python.exe");
        } else {
            venvPython = new File(workingDir, "venv/bin/python");
        }

        if (venvPython.exists()) {
            return venvPython.getAbsolutePath();
        }

        // Fallback to system python
        return "python";
    }

    private void log(String message) {
        System.out.println("[GameProcessService] " + message);
    }
}