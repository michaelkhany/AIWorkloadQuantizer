# ==============================================================================
# Project      : AIWorkloadQuantizer
# File         : app.py
# Created on   : 2022-02-15
# Last Modified: 2022-03-25
# Author       : Michael B. Khani
# Reviewer     : —
# ==============================================================================
# Description  :
# ------------------------------------------------------------------------------
# AIWorkloadQuantizer is an offline, browser-based monitoring tool developed 
# using Streamlit. It enables users to execute any Python script, monitor 
# system resource usage in real-time (CPU, memory, GPU, and disk I/O), and 
# quantify the AI workload based on performance and execution characteristics.
#
# Key Features:
#   - Interactive script selection from a local directory.
#   - Parameter injection to support custom command-line arguments.
#   - Real-time visualization of hardware resource utilization.
#   - AI workload metric computed via a differential system model.
#   - Exportable logs, performance charts, and metric reports (JSON format).
#
# Underlying Methods:
#   - `ResourceMonitor` (in utils/monitor.py) tracks time-series data for 
#     CPU, memory, GPU, and disk activity during script execution.
#   - `calculate_workload` (in utils/metrics.py) computes a composite metric 
#     that quantifies computational effort, tailored for AI workloads.
#
# Citation:
#   A. K. Sharma, M. Bidollahkhani, and J. M. Kunkel, 
#   “AI Work Quantization Model: Closed-System AI Computational Effort Metric,” 
#   arXiv preprint arXiv:2503.14515, Mar. 2025. [Online]. 
#   Available: https://doi.org/10.48550/arXiv.2503.14515
#
# Usage:
#   Install Streamlit (if not already installed):
#       $ pip install streamlit
#
#   Run the application:
#       $ streamlit run app.py
#
# Notes:
#   - Designed for offline use.
#   - All outputs (logs, metrics, plots) are saved in session-specific folders.
#   - Compatible with Python 3.7+, Streamlit, version 1.42.1+.

# ==============================================================================

import os
import time
import datetime
import threading
import streamlit as st
import pandas as pd

# Import our helper modules
from utils import file_utils, monitor, executor, logger, metrics

st.title("AIWorkloadQuantizer")
st.sidebar.header("Script Selection")

# Initialize session state flag if not already defined
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False

# Input for scripts directory
scripts_dir = st.sidebar.text_input("Enter the scripts directory:", value="scripts")
if not os.path.exists(scripts_dir):
    st.error("Directory does not exist. Please create it and add your Python scripts.")
else:
    scripts = file_utils.list_scripts(scripts_dir)
    if not scripts:
        st.warning("No Python scripts found in the selected directory.")
    else:
        selected_script = st.sidebar.selectbox("Select a Python script:", scripts)
        params = st.sidebar.text_input("Enter command-line parameters (space-separated):", "")
        
        # The run button; note that if a report has already been generated,
        # we continue showing the previous results.
        run_button = st.sidebar.button("Run Script")
        
        # If run button is clicked or a report is already generated, display output
        if run_button or st.session_state.report_generated:
            # Run the script only if we haven't generated a report yet
            if not st.session_state.report_generated:
                # Create a unique report folder based on timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                report_dir = os.path.join("data", "reports", f"script_{timestamp}")
                os.makedirs(report_dir, exist_ok=True)
                st.session_state.report_dir = report_dir

                st.subheader("Console Output")
                console_output_area = st.empty()

                # Start resource monitoring in a background thread
                resource_monitor = monitor.ResourceMonitor(report_dir)
                monitor_thread = threading.Thread(target=resource_monitor.start_monitoring, args=(1,))
                monitor_thread.start()

                # Build the script path and parse parameters
                script_path = os.path.join(scripts_dir, selected_script)
                args_list = file_utils.parse_args(params)

                # Define callback to update console output in real time
                def output_callback(line):
                    console_output_area.text(line)

                st.info("Running script...")
                start_time = time.time()
                exit_code = executor.run_script(script_path, args_list, output_callback)
                execution_time = time.time() - start_time

                # Stop monitoring and wait for the thread to finish
                resource_monitor.stop_monitoring()
                monitor_thread.join()

                # Retrieve resource usage data and save to CSV
                resource_df = resource_monitor.get_data()
                resources_csv_path = os.path.join(report_dir, "resources.csv")
                resource_df.to_csv(resources_csv_path, index=False)

                # Save final console log
                console_log_path = os.path.join(report_dir, "console.log")
                logger.log_console_output(f"Script finished with exit code: {exit_code}", console_log_path)

                # Calculate AI workload metrics and save as JSON
                metrics_json_path = os.path.join(report_dir, "metrics.json")
                metrics_result = metrics.calculate_workload(resource_df, execution_time)
                logger.log_metrics(metrics_result, metrics_json_path)

                st.success("Script execution completed!")
                st.write("**Execution Time:** {:.2f} seconds".format(execution_time))
                st.write("**AI Workload:** {:.6f}".format(metrics_result["ai_workload"]))

                # Plot resource usage using Plotly
                import plotly.express as px
                fig = px.line(resource_df,
                              x="timestamp",
                              y=["cpu_percent", "memory_percent", "gpu_percent", "disk_io"],
                              labels={"value": "Usage", "timestamp": "Time (UTC)"},
                              title="System Resource Usage Over Time")
                st.plotly_chart(fig)

                # Cache file contents in session state to preserve data across re-runs
                with open(resources_csv_path, "rb") as f:
                    st.session_state.resources_csv_data = f.read()
                with open(console_log_path, "rb") as f:
                    st.session_state.console_log_data = f.read()
                with open(metrics_json_path, "rb") as f:
                    st.session_state.metrics_json_data = f.read()

                st.session_state.execution_time = execution_time
                st.session_state.metrics_result = metrics_result
                st.session_state.report_generated = True

            # Display outputs from session state
            st.write("**Execution Time:** {:.2f} seconds".format(st.session_state.execution_time))
            st.write("**AI Workload:** {:.6f}".format(st.session_state.metrics_result["ai_workload"]))

            # Download buttons using cached file contents
            st.download_button(
                label="Download Resource CSV",
                data=st.session_state.resources_csv_data,
                file_name="resources.csv"
            )
            st.download_button(
                label="Download Console Log",
                data=st.session_state.console_log_data,
                file_name="console.log"
            )
            st.download_button(
                label="Download Metrics JSON",
                data=st.session_state.metrics_json_data,
                file_name="metrics.json"
            )
