import pynvml
import psutil
import time
import os

def get_process_name(pid):
    try:
        proc = psutil.Process(pid)
        return proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return "Unknown/Docker"

def track_vram():
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    if device_count == 0:
        print("[Error] No NVIDIA GPU found.")
        return

    # Assuming RTX 3090 is GPU 0
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    
    try:
        while True:
            os.system('cls' if os.name == 'nt' else 'clear')
            
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_vram = memory_info.total / (1024 ** 2)
            used_vram = memory_info.used / (1024 ** 2)
            free_vram = memory_info.free / (1024 ** 2)
            
            print("="*60)
            print(f" VRAM X-RAY LOG | GPU: {gpu_name}")
            print(f" TOTAL: {total_vram:.0f} MB | USED: {used_vram:.0f} MB | FREE: {free_vram:.0f} MB")
            print("="*60)
            print(f"{'PID':<10} | {'PROCESS NAME':<20} | {'VRAM USED':<10}")
            print("-" * 60)
            
            # Fetch all processes currently using the GPU
            compute_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
            
            all_procs = {p.pid: p.usedGpuMemory for p in (compute_procs + graphics_procs)}
            
            # Sort by highest memory usage
            sorted_procs = sorted(all_procs.items(), key=lambda item: item[1] if item[1] else 0, reverse=True)
            
            for pid, mem in sorted_procs:
                name = get_process_name(pid)
                mem_mb = (mem / (1024 ** 2)) if mem else 0
                
                # WSL2/Docker processes usually show up as 'wsl' or 'Unknown' depending on Windows version
                if "vmmem" in name.lower() or "wsl" in name.lower():
                    name = f"[Docker/WSL] {name}"
                
                print(f"{pid:<10} | {name:<20} | {mem_mb:>7.0f} MB")
            
            print("\nPress Ctrl+C to exit tracker...")
            time.sleep(2)
            
    except KeyboardInterrupt:
        print("\nShutting down VRAM tracker.")
    finally:
        pynvml.nvmlShutdown()

if __name__ == "__main__":
    track_vram()