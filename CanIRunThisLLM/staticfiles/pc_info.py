import platform, psutil, GPUtil, cpuinfo, requests, json, sys, time, os, datetime
from selenium import webdriver
import time
from multiprocessing import Process, freeze_support

API_URL = "http://127.0.0.1:8000/"
_already_executed = False

def log(msg):
    print(f"{datetime.datetime.now()} [PID:{os.getpid()}] {msg}")

def get_system_info():
    log("Get CPU information")
    try:
        cpu_info = cpuinfo.get_cpu_info()
    except Exception as e:
        log(f"Error getting CPU info: {e}")
        cpu_info = {}
    info = {
        "os": platform.system(),
        "cpu_name": cpu_info.get('brand_raw', 'Unknown'),
        "cpu_cores": psutil.cpu_count(logical=False) or 0,
        "logical_cpus": psutil.cpu_count(logical=True) or 0,
        "ram": f"{round(psutil.virtual_memory().total / (1024 ** 3), 2)} GB",
    }
    
    log("Get GPU information")
    try:
        gpus = GPUtil.getGPUs()
    except Exception as e:
        log(f"Error getting GPU info: {e}")
        gpus = []
    if gpus:
        gpu_list = []
        for gpu in gpus:
            gpu_list.append({
                "name": gpu.name,
                "memory_total": f"{gpu.memoryTotal/1000} GB",
            })
        info["gpus"] = gpu_list
    else:
        info["gpus"] = None

    log("Get DISK information")
    partitions = psutil.disk_partitions()
    storage_list = []
    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            storage_list.append({
                "device": partition.device,
                "total": f"{usage.total / (1024 ** 3):.2f} GB",
                "free": f"{usage.free / (1024 ** 3):.2f} GB",
            })
        except Exception as e:
            storage_list.append({
                "device": partition.device,
                "error": str(e)
            })
    info["storage"] = storage_list

    return info

def send_data():
    global _already_executed
    if _already_executed:
        log("send_data() already executed. Exiting.")
        sys.exit(0)
    _already_executed = True

    log("Collecting system info...")
    system_info = get_system_info()
    log("System info collected:")
    log(json.dumps(system_info, indent=2))
    
    try:
        log(f"Sending POST request to: {API_URL}")
        response = requests.post(API_URL, json=system_info, headers={'Content-Type': 'application/json'}, timeout=10)
        log(f"HTTP response status code: {response.status_code}")
    except Exception as e:
        log(f"Error during POST request: {e}")
        sys.exit(1)

    if response.status_code == 200:
        log("Data sent successfully!")

    else:
        log(f"Failed to send data: {response.status_code} {response.text}")
    
    time.sleep(1)
    sys.exit(0)

if __name__ == "__main__":
    freeze_support()
    send_data()