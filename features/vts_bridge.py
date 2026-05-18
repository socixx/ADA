import asyncio
import threading
import pyvts
import os

class VTSBridge:
    def __init__(self, port=8001):
        print(f"[Model-Engine] Initializing VTube Studio Bridge on port {port}...")
        
        # Ensure the memory directory exists for the auth token
        os.makedirs("./memory", exist_ok=True)
        
        plugin_info = {
            "plugin_name": "Ada Core AI",
            "developer": "Ada Core Pipeline",
            "authentication_token_path": "./memory/vts_token.txt"
        }
        
        self.vts = pyvts.vts(plugin_info=plugin_info, port=port)
        self.connected = False
        self.hotkeys = []
        
        # Spawn dedicated thread for the VTS asyncio loop
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._start_loop, daemon=True)
        self.thread.start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.connect_and_auth())
            self.loop.run_forever()
        except Exception as e:
            print(f"[Model-Engine] Disconnected or failed to connect: {e}")

    async def connect_and_auth(self):
        try:
            await self.vts.connect()
            
            # 1. Try to read the token and authenticate silently
            try:
                await self.vts.read_token()
                await self.vts.request_authenticate()
            except Exception:
                # 2. If it fails (no file, or invalid token), request a new one
                print("[Model-Engine] Token missing or invalid. Please click 'Allow' inside VTube Studio...")
                
                # This line triggers the actual popup in the VTube Studio UI
                await self.vts.request_authenticate_token()
                
                # Save the new token for next time
                await self.vts.write_token()
                
                # Authenticate with the newly granted token
                await self.vts.request_authenticate()
            
            self.connected = True
            print("[Model-Engine] ✅ Successfully authenticated with VTube Studio!")
            
            # 3. Fetch all available hotkeys/expressions from the currently loaded model
            response = await self.vts.request(self.vts.vts_request.requestHotKeyList())
            
            # Safely extract hotkeys to prevent future KeyErrors
            self.hotkeys = response.get('data', {}).get('availableHotkeys', [])
            print(f"[Model-Engine] Loaded {len(self.hotkeys)} hotkeys from current model.")
            
        except Exception as e:
            print(f"[Model-Engine] 🛑 Authentication failed: {e}")

    def trigger_action(self, action_text: str):
        """
        Synchronous method called by main.py.
        Fires an async websocket request to VTS.
        """
        if not self.connected:
            return
            
        clean_action = action_text.lower().strip()
        print(f"\n[Model-Engine] ✨ Triggering expression: *{clean_action}*")
        
        asyncio.run_coroutine_threadsafe(self._async_trigger(clean_action), self.loop)

    async def _async_trigger(self, action_text: str):
        target_hotkey = None
        # Naive matching: if the LLM action word is in the VTS hotkey name, or vice versa
        for hk in self.hotkeys:
            hk_name = hk['name'].lower()
            if action_text in hk_name or hk_name in action_text:
                target_hotkey = hk['hotkeyID']
                break
                
        if target_hotkey:
            await self.vts.request(self.vts.vts_request.requestTriggerHotKey(target_hotkey))
        else:
            pass # Fail silently if she hallucinates an action you don't have a hotkey for