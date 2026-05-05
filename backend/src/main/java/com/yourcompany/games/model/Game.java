package com.yourcompany.games.model;

public class Game {
    private String id;
    private String name;
    private String description;
    private String scriptPath;
    private boolean running;
    private String statusMessage;

    public Game() {}

    public Game(String id, String name, String description, String scriptPath) {
        this.id = id;
        this.name = name;
        this.description = description;
        this.scriptPath = scriptPath;
        this.running = false;
        this.statusMessage = "Idle";
    }

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public String getScriptPath() { return scriptPath; }
    public void setScriptPath(String scriptPath) { this.scriptPath = scriptPath; }

    public boolean isRunning() { return running; }
    public void setRunning(boolean running) { this.running = running; }

    public String getStatusMessage() { return statusMessage; }
    public void setStatusMessage(String statusMessage) { this.statusMessage = statusMessage; }
}