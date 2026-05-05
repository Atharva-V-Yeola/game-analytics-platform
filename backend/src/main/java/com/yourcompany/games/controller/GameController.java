package com.yourcompany.games.controller;

import com.yourcompany.games.model.Game;
import com.yourcompany.games.service.GameProcessService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.stream.Collectors;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import jakarta.annotation.PostConstruct;

@RestController
@RequestMapping("/api/games")
@CrossOrigin(origins = "http://localhost:3000") // For React dev server
public class GameController {

    @Autowired
    private GameProcessService processService;

    private List<Game> availableGames = new ArrayList<>();

    @PostConstruct
    public void init() {
        try {
            ObjectMapper mapper = new ObjectMapper();
            File projectRoot = processService.resolveWorkingDirectory();
            File registryFile = new File(projectRoot, "configs/games_registry.json");
            
            if (registryFile.exists()) {
                availableGames = mapper.readValue(registryFile, new TypeReference<List<Game>>() {});
            } else {
                System.err.println("Warning: " + registryFile.getAbsolutePath() + " not found!");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    /**
     * GET /api/games
     * Returns all games with their current running status.
     */
    @GetMapping
    public List<Game> getGames() {
        return availableGames.stream()
            .peek(game -> {
                boolean running = processService.isRunning(game.getId());
                game.setRunning(running);
                game.setStatusMessage(running ? "Running" : "Idle");
            })
            .collect(Collectors.toList());
    }

    /**
     * POST /api/games/{id}/start
     */
    @PostMapping("/{id}/start")
    public ResponseEntity<?> startGame(@PathVariable String id) {
        Game game = findGame(id);
        if (game == null) {
            return ResponseEntity.notFound().build();
        }

        boolean started = processService.startGame(game);
        if (started) {
            return ResponseEntity.ok().body("{\"status\":\"started\"}");
        } else {
            return ResponseEntity.badRequest().body("{\"error\":\"Already running or failed to start\"}");
        }
    }

    /**
     * POST /api/games/{id}/stop
     */
    @PostMapping("/{id}/stop")
    public ResponseEntity<?> stopGame(@PathVariable String id) {
        boolean stopped = processService.stopGame(id);
        if (stopped) {
            return ResponseEntity.ok().body("{\"status\":\"stopped\"}");
        } else {
            return ResponseEntity.badRequest().body("{\"error\":\"Not running\"}");
        }
    }

    /**
     * GET /api/games/data
     * Lists CSV files in the data/ directory.
     */
    @GetMapping("/data")
    public List<String> getDataFiles() {
        File projectRoot = processService.resolveWorkingDirectory();
        File dataDir = new File(projectRoot, "data");
        
        if (!dataDir.exists() || !dataDir.isDirectory()) {
            return Collections.emptyList();
        }

        File[] files = dataDir.listFiles((dir, name) -> name.endsWith(".csv"));
        if (files == null) {
            return Collections.emptyList();
        }

        return Arrays.stream(files)
            .sorted((a, b) -> Long.compare(b.lastModified(), a.lastModified())) // Newest first
            .map(File::getName)
            .collect(Collectors.toList());
    }

    private Game findGame(String id) {
        return availableGames.stream()
            .filter(g -> g.getId().equals(id))
            .findFirst()
            .orElse(null);
    }
}