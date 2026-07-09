import asyncio
import threading
import pyvts
import os

class VTSBridge:
    def __init__(self, port=8006):
        print(f"[Model-Engine] Initializing VTube Studio Bridge on port {port}...")
        
        # Ensure the directory exists
        os.makedirs("./character_files", exist_ok=True)
        
        # CHANGED: Target location redirected explicitly into character_files/
        token_path = "./character_files/vts_token.txt"
        
        plugin_info = {
            "plugin_name": "Ada Intelligence Engine",
            "developer": "Ada Core",
            "authentication_token_path": token_path
        }
        
        self.vts = pyvts.vts(plugin_info=plugin_info, port=port)
        self.connected = False
        self.hotkeys = []
        
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
            
            # Enforce hardcoded token route inside pyvts handler references
            token_path = "./character_files/vts_token.txt"
            self.vts.vts_request.authentication_token_path = token_path
            
            # 1. Look for your custom pasted token file
            if os.path.exists(token_path):
                print(f"[Model-Engine] Found manual token file at {token_path}. Authenticating...")
                await self.vts.read_token()
                try:
                    await self.vts.request_authenticate()
                    self.connected = True
                except Exception:
                    print("[Model-Engine] Token rejected by VTS (403). Resetting cache target...")
                    os.remove(token_path)
                    self.connected = False
            
            # 2. Fallback request prompt if token gets invalidated or file is absent
            if not self.connected:
                print("\n" + "="*50)
                print("[Model-Engine] REQUESTING FRESH ACCESS PAIR!")
                print("[Model-Engine] PLEASE CLICK 'ALLOW' INSIDE VTUBE STUDIO NOW.")
                print("="*50 + "\n")
                
                await self.vts.request_authenticate_token()
                await self.vts.write_token()
                await self.vts.request_authenticate()
                self.connected = True
            
            print("[Model-Engine] ✅ Successfully authenticated with VTube Studio!")
            
            response = await self.vts.request(self.vts.vts_request.requestHotKeyList())
            self.hotkeys = response.get('data', {}).get('availableHotkeys', [])
            print(f"[Model-Engine] Loaded {len(self.hotkeys)} hotkeys from current model.")
            
        except Exception as e:
            print(f"[Model-Engine] 🛑 Authentication failed: {e}")

    def trigger_action(self, action_text: str):
        if not self.connected: return
        clean_action = action_text.lower().strip()
        print(f"\n[Model-Engine] ✨ Triggering expression: *{clean_action}*")
        asyncio.run_coroutine_threadsafe(self._async_trigger(clean_action), self.loop)

    async def _async_trigger(self, action_text: str):
        target_hotkey = None
        for hk in self.hotkeys:
            hk_name = hk['name'].lower()
            if action_text in hk_name or hk_name in action_text:
                target_hotkey = hk['hotkeyID']
                break
                
        if target_hotkey:
            await self.vts.request(self.vts.vts_request.requestTriggerHotKey(target_hotkey))