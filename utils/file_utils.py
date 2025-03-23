import os
import shlex

def list_scripts(directory):
    """Return a list of Python script filenames in the given directory."""
    try:
        return [f for f in os.listdir(directory) if f.endswith(".py")]
    except Exception:
        return []

def parse_args(params_str):
    """
    Parse a string of command-line parameters into a list.
    For example, the string '--flag1 value1 --flag2 "value with spaces"'
    will be converted to a list of arguments.
    """
    if not params_str.strip():
        return []
    return shlex.split(params_str)
