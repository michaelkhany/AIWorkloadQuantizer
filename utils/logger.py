import json
import numpy as np

def log_console_output(text, filepath):
    """Append a line of text to the console log file."""
    with open(filepath, "a") as f:
        f.write(text + "\n")

def log_metrics(metrics_dict, filepath):
    """Save the metrics dictionary to a JSON file, converting non-serializable types."""
    def convert(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        if isinstance(o, (np.float64, np.float32)):
            return float(o)
        raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
    
    with open(filepath, "w") as f:
        json.dump(metrics_dict, f, indent=4, default=convert)
