# AIWorkloadQuantizer

## Overview

**AIWorkloadQuantizer** is an offline, browser-based application built with **Streamlit** that allows you to run any Python script, monitor its hardware resource usage in real-time, and compute a quantized AI workload metric based on system performance and execution characteristics.

## Features

- **Script Selection UI:** Choose a directory containing Python scripts and select which one to run, with optional command-line parameters.
- **Live Execution & Monitoring:** Track CPU, GPU, RAM, and disk I/O usage in real time while the script executes.
- **AI Workload Computation:** Calculate an AI workload metric using a standardized equation derived from resource usage and runtime.
- **Results & Downloads:** Visualize performance charts, view logs and metrics, and download CSV/JSON reports.

## Running the app using streamlit
```
streamlit run app.py
```

You can now view your Streamlit app in your browser.
```
Local URL: http://localhost:8501
Network URL: http://192.168.0.141:8501
```
## Reference
- A. K. Sharma, M. Bidollahkhani, and J. M. Kunkel, “AI Work Quantization Model: Closed-System AI Computational Effort Metric,” arXiv preprint arXiv:2503.14515, Mar. 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2503.14515
