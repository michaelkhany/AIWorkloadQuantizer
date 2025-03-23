import math

def calculate_workload(resource_df, execution_time):
    """
    Calculate the AI Workload metric using a differential approach.
    
    The method records the initial resource usage, then determines the maximum usage 
    during execution, calculates the difference, and computes the workload metric based on that difference.
    
    AI Workload = (log(1 + CompRes_diff) / log(1 + 10^18)) * (1 / Execution Time)
    
    where:
        CompRes_diff = (delta_CPU_GIPS + delta_RAM_GTps + delta_GPU_GIPS + delta_Storage_IO_GBs)
    """
    # Check if data is available
    if resource_df.empty:
        return {"ai_workload": 0, "error": "No resource data recorded."}

    # Initial (baseline) values
    initial = resource_df.iloc[0]
    # Maximum values observed during the run
    maximum = resource_df.max()

    # Calculate the differences (ensuring they are non-negative)
    delta_cpu = max(0, maximum["cpu_percent"] - initial["cpu_percent"])
    delta_mem = max(0, maximum["memory_percent"] - initial["memory_percent"])
    delta_gpu = max(0, maximum["gpu_percent"] - initial["gpu_percent"])
    delta_disk_io = max(0, maximum["disk_io"] - initial["disk_io"])

    # Scaling factors (demonstration purposes)
    CPU_factor = 1e9     # Assume 1 GIPS at 100% CPU usage
    MEM_factor = 1e9     # Assume 1 GTps at 100% Memory usage
    GPU_factor = 1e9     # Assume 1 GIPS at 100% GPU usage
    IO_factor = 1e-9     # Convert bytes to GigaBytes

    CPU_GIPS = (delta_cpu / 100) * CPU_factor
    RAM_GTps = (delta_mem / 100) * MEM_factor
    GPU_GIPS = (delta_gpu / 100) * GPU_factor
    Storage_IO_GBs = delta_disk_io * IO_factor

    CompRes_diff = CPU_GIPS + RAM_GTps + GPU_GIPS + Storage_IO_GBs

    if execution_time <= 0:
        ai_workload = 0
    else:
        ai_workload = (math.log(1 + CompRes_diff) / math.log(1 + 1e18)) * (1 / execution_time)

    return {
        "initial_cpu": initial["cpu_percent"],
        "initial_memory": initial["memory_percent"],
        "initial_gpu": initial["gpu_percent"],
        "initial_disk_io": initial["disk_io"],
        "max_cpu": maximum["cpu_percent"],
        "max_memory": maximum["memory_percent"],
        "max_gpu": maximum["gpu_percent"],
        "max_disk_io": maximum["disk_io"],
        "delta_cpu": delta_cpu,
        "delta_memory": delta_mem,
        "delta_gpu": delta_gpu,
        "delta_disk_io": delta_disk_io,
        "CompRes_diff": CompRes_diff,
        "execution_time": execution_time,
        "ai_workload": ai_workload
    }
