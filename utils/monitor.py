import time
import threading
import pandas as pd
import psutil
import GPUtil

class ResourceMonitor:
    def __init__(self, report_dir):
        self.report_dir = report_dir
        self.running = False
        self.data = []
        self.lock = threading.Lock()

    def start_monitoring(self, interval=1):
        """Continuously record resource usage at every `interval` seconds."""
        self.running = True
        while self.running:
            ts = time.time()
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            # Total disk I/O (read + write)
            disk_counters = psutil.disk_io_counters()
            disk_io = disk_counters.read_bytes + disk_counters.write_bytes
            # GPU usage (if available)
            try:
                gpus = GPUtil.getGPUs()
                gpu = gpus[0].load * 100 if gpus else 0
            except Exception:
                gpu = 0
            with self.lock:
                self.data.append({
                    "timestamp": ts,
                    "cpu_percent": cpu,
                    "memory_percent": mem,
                    "gpu_percent": gpu,
                    "disk_io": disk_io
                })
            time.sleep(interval)

    def stop_monitoring(self):
        """Stop the monitoring loop."""
        self.running = False

    def get_data(self):
        """Return the recorded data as a pandas DataFrame."""
        with self.lock:
            df = pd.DataFrame(self.data)
        # Convert UNIX timestamp to readable datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
