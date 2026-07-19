import os
import time
import base64
import sqlite3
import threading
import urllib.parse
from playwright.sync_api import sync_playwright
import eel

class BrowserNode:
    def __init__(self):
        self.browser = None
        self.context = None
        self.page = None
        self.playwright = None
        self.is_active = False
        self.stream_thread = None
        self.db_path = "skills/browser_history.db"
        
        # Initialize historical ledger tracking
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT NOT NULL,
                    title TEXT,
                    timestamp REAL NOT NULL
                )
            """)

    def launch(self):
        """Spins up the isolated browser context and hooks global listeners."""
        if self.is_active:
            return
            
        print("[Browser Node]: Engaging native Chromium sandbox...")
        self.is_active = True
        
        # Dynamic window adjustment executed inside Python container limits
        try:
            # Resize application deck to fit layout expansions
            eel.set_window_size(1920, 920)
        except Exception:
            pass

        # Boot playwright inside dedicated thread wrapper
        threading.Thread(target=self._browser_lifecycle_loop, daemon=True).start()

    def _browser_lifecycle_loop(self):
        with sync_playwright() as p:
            self.playwright = p
            self.browser = p.chromium.launch(
                headless=True,  # Set False so you can see the local instance if desired
                args=["--disable-blink-features=AutomationControlled"]
            )
            self.context = self.browser.new_context(
                viewport={"width": 1280, "height": 720},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            self.page = self.context.new_page()
            
            # Register navigation tracking events natively
            self.page.on("framenavigated", self._log_navigation)
            
            # Load landing baseline
            self.page.goto("https://www.google.com")
            
            # Spin up the frame-by-frame base64 UI pipeline
            self.stream_thread = threading.Thread(target=self._screencast_pipeline, daemon=True)
            self.stream_thread.start()
            
            # Keep the lifecycle running until aborted
            while self.is_active:
                time.sleep(0.5)
                
            self.browser.close()

    def _screencast_pipeline(self):
        """Continuously pipes encoded frame captures down to the graphical window layer."""
        while self.is_active and self.page:
            try:
                # Capture target active surface layout
                screenshot_bytes = self.page.screenshot(type="jpeg", quality=60)
                b64_string = base64.b64encode(screenshot_bytes).decode("utf-8")
                
                # Pipe base64 string downstream to Eel graphics hook
                eel.update_browser_viewport(b64_string)()
            except Exception:
                pass
            time.sleep(0.2)  # Max ~5hz telemetry stream overhead protection

    def _log_navigation(self, frame):
        if frame == self.page.main_frame:
            url = self.page.url
            try:
                title = self.page.title()
            except Exception:
                title = "Unknown Page Source"
                
            print(f"[Browser Ledger Nav]: {title} -> {url}")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO history (url, title, timestamp) VALUES (?, ?, ?)",
                    (url, title, time.time())
                )
            
            # Notify UI to update history module instantly
            try:
                eel.trigger_history_sync()()
            except Exception:
                pass

    # --- ACTION ABSTRACTIONS FOR COGNITIVE LOOP ENTRY ---
    def execute_navigate(self, url):
        if self.page:
            self.page.goto(url, wait_until="load")
            return self.scrape_page_text()
        return "Browser node context uninitialized."

    def execute_search(self, query):
        if self.page:
            search_url = f"https://www.google.com/search?q={urllib.parse.quote_plus(query)}"
            self.page.goto(search_url, wait_until="load")
            return self.scrape_page_text()
        return "Browser node context uninitialized."

    def scrape_page_text(self):
        """Extracts text for Ada to ingest as short-term structural awareness context."""
        if self.page:
            try:
                return self.page.evaluate("() => document.body.innerText")
            except Exception as e:
                return f"Failed contextual text collection parse: {e}"
        return ""

    def close(self):
        self.is_active = False
        try:
            eel.set_window_size(1500, 920)  # Re-scale back down seamlessly
        except Exception:
            pass