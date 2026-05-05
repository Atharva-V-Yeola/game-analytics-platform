# Game Analytics Platform

Welcome to the Game Analytics Platform! This software uses your webcam and advanced AI computer vision to track your movements across 16 different fitness exercises.

This guide is written for **non-technical users**. If this is a brand-new computer, you will need to install a few foundational programs before the game can run. Don't worry, it's just like installing any normal app!

---

## Step 1: Download the Project
1. Go to the GitHub page where this code is hosted.
2. Click the green **"<> Code"** button.
3. Click **"Download ZIP"**.
4. Once downloaded, **extract (unzip)** the folder to your Desktop.

---

## Step 2: Install Prerequisites
Your computer needs three pieces of standard software to run the AI, the backend server, and the dashboard. 

> **Important during installation:** If any installer asks to "Add to PATH", make sure you **check that box!**

### 1. Install Python (Runs the AI Vision)
* **Windows & Mac:** Go to [python.org/downloads](https://www.python.org/downloads/) and click the big yellow download button (Python 3.10 or higher is recommended). 
* *Crucial Windows Step:* When you open the Python installer, check the box at the very bottom that says **"Add python.exe to PATH"** before clicking Install.

### 2. Install Java 17 (Runs the Server)
* Go to [Adoptium.net](https://adoptium.net/temurin/releases/?version=17).
* Select your Operating System (Windows, macOS, or Linux).
* Download the `.msi` (Windows) or `.pkg` (Mac) installer and run it. Keep hitting "Next" until it finishes.

### 3. Install Node.js (Runs the Dashboard)
* Go to [nodejs.org](https://nodejs.org/).
* Download the **LTS (Long Term Support)** version.
* Run the installer and keep hitting "Next" until it finishes.

---

## Step 3: Run the Auto-Installer
Now that your computer has the right tools, the project can build itself!

### On Windows:
1. Open the folder you extracted on your desktop.
2. Click on the address bar at the very top of the folder window.
3. Type `cmd` and press **Enter**. (This opens a black terminal window).
4. Type the following command and press **Enter**:
   ```
   python install.py
   ```
5. Wait. The script will automatically download the AI models, build the website, and configure the server. It might take 5-10 minutes. 

### On Mac/Linux:
1. Open the **Terminal** app.
2. Type `cd ` (with a space after it).
3. Drag the unzipped project folder from your desktop into the Terminal window and press **Enter**.
4. Type the following command and press **Enter**:
   ```
   python3 install.py
   ```
5. Wait for the installation to complete.

---

## Step 4: Play!

Once the installer finishes successfully, you will never need to run `install.py` again.

### How to start the game every time:
* **Windows:** Open the project folder and double-click the **`start.bat`** file.
* **Mac/Linux:** Open the terminal in the folder and type `./start.sh`

A terminal window will pop up to start the server. Leave this window open in the background!

### Open the Dashboard
1. Open Google Chrome or Safari.
2. Type `http://localhost:8080` into the web address bar and press Enter.
3. You will see the Game Analytics Dashboard! Click on any of the 16 exercises to begin your workout.

### How to close the game:
When you are done playing, go back to the black terminal window running the server and press **Ctrl + C** on your keyboard to shut it down.
