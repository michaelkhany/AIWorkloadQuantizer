import subprocess

def run_script(script_path, args_list, output_callback):
    """
    Run a Python script with the provided arguments.
    Captures stdout and stderr in real time, calling output_callback for each line.
    """
    cmd = ["python", script_path] + args_list
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Poll for new output until the process terminates
    while True:
        output = process.stdout.readline()
        error = process.stderr.readline()
        if output:
            output_callback(output.strip())
        if error:
            output_callback(error.strip())
        if output == "" and error == "" and process.poll() is not None:
            break
    return process.returncode
