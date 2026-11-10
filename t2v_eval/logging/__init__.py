import os
import subprocess
from enum import Enum

from .utils import (
    LOG_DIR,
    reset_log_dir,
    setup_log_dir,
    set_variable,
    get_variable,
    log_data,
    compact_log_dir,
    iter_series_records,
    SERIES_JSONL_NAME,
)
from .matplotlib_logging import activate_matplotlib_logging
from .seaborn_logging import activate_seaborn_logging
from .geopandas_logging import activate_geopandas_logging

__all__ = [
    "LOG_DIR",
    "reset_log_dir",
    "setup_log_dir",
    "set_variable",
    "get_variable",
    "log_data",
    "compact_log_dir",
    "iter_series_records",
    "SERIES_JSONL_NAME",
    "activate_matplotlib_logging",
    "activate_seaborn_logging",
    "activate_geopandas_logging",
    "LoggingState",
    "CodeCompletionState"
]

class LoggingState(Enum):
    BAD_INPUT = "BAD_INPUT"
    HAVING_DATA = "HAVING_DATA"
    NO_DATA = "NO_DATA"
    
class CodeCompletionState(Enum):
    CONTEXT_ERROR = "context-error"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"

LOGGING_ACTIVATION_CODE = """from t2v_eval.logging import activate_t2v_logging; activate_t2v_logging({log_dir!r})"""
IGNORE_WARNING_CODE = "import warnings; warnings.filterwarnings('ignore')"


_PATCHES_APPLIED = False


def activate_t2v_logging(log_dir=None):
    """
    Activate T2V logging for matplotlib and seaborn.

    Monkey-patches are applied only on the first call.  Subsequent calls only
    update the log directory and re-enable ``T2V_ISLOG``, so repeated
    invocations (e.g. in a test suite) never double-wrap any method.
    """
    global _PATCHES_APPLIED
    try:
        if log_dir is None:
            reset_log_dir()
        else:
            setup_log_dir(log_dir)
        set_variable("T2V_ISLOG", True)
        if not _PATCHES_APPLIED:
            activate_matplotlib_logging()
            activate_seaborn_logging()
            activate_geopandas_logging()
            _PATCHES_APPLIED = True
        return True
    except Exception as e:
        print(f"Error activating T2V logging: {e}")
        return False


def prepare_py_file_for_logging(py_path_in, 
                                py_path_out,
                                log_dir):
    """
    Prepare a Python file for logging by inserting logging activation code.
    Args:
        py_path_in (str): Path to the input Python file.
        py_path_out (str): Path to save the output Python file with logging code.
    Returns:
        str: Path to the output Python file with logging code.
    """ 
    
    # load python file
    source = open(py_path_in, 'r').read()
    
    # find line startswith "##START VISUALISATION CODE"
    output_idx = None
    lines = source.split("\n")
    for i in range(len(lines)):
        line_lower = lines[i].lower()
        if "##start visualisation code" in line_lower:
            output_idx = i
            break
    if output_idx is None:
        print(f"Cannot find the output visualization code line in the py file: {py_path_in}")
        return None
    
    # insert activate logging code before that line
    if output_idx is not None:
        lines.insert(output_idx, LOGGING_ACTIVATION_CODE.format(log_dir=log_dir))
        
    # insert ignore warning code at the beginning of the file
    lines.insert(0, IGNORE_WARNING_CODE) 
    
    # join lines to source
    source = "\n".join(lines)
    
    # save to output_path
    with open(py_path_out, 'w') as f:
        f.write(source)

    return py_path_out


def is_valid_log_dir(log_dir, verbose=False):
    """
    Check if the log directory contains valid logging data.
    A valid log directory should exist and contain more than 1 file.
    1 default file ("variable.json") is created even if no data is logged.
    
    Args:
        log_dir (str): Path to the log directory.
        verbose (bool): If True, print detailed information.
    Returns:
        bool: True if the log directory is valid, False otherwise.
    """
    
    if not os.path.exists(log_dir):
        if verbose:
            print(f"Log directory does not exist: {log_dir}")
        return False
    
    num_files = sum([len(files) for r, d, files in os.walk(log_dir)])
    if num_files <= 1:
        if verbose:
            print(f"Log directory has no data files: {log_dir}")
        return False

    if verbose:
        print(f"Log directory is valid: {log_dir} with {num_files} files.")
    return True


def execute_and_log_data_py_path(py_path,
                                 py_env="python3",
                                 timeout=300,
                                 use_cache=False,
                                 verbose=False):
    assert "/t2v_pred/" in py_path or "/t2v_gt/" in py_path, f"Python path does not contain /t2v_pred/ or /t2v_gt/: {py_path}"
    
    # check if the log directory already exists, if use_cache is True
    py_log_dir = py_path.rsplit(".", maxsplit=1)[0] + "_log"
    if use_cache and is_valid_log_dir(py_log_dir, verbose=verbose):
        return LoggingState.HAVING_DATA, py_log_dir
    
    # setup logging directory
    setup_log_dir(py_log_dir)
    
    # prepare py file with logging code inserted
    py_path_out = prepare_py_file_for_logging(py_path_in=py_path, 
                                              py_path_out=f"{py_log_dir}.py",
                                              log_dir=py_log_dir)
    if py_path_out is None:
        return LoggingState.BAD_INPUT, py_log_dir
    
    # execute the py file
    try:
        result = subprocess.run([py_env, py_path_out], 
                                cwd=os.path.dirname(py_path_out),
                                capture_output=True, 
                                text=True,
                                timeout=timeout)  # timeout in seconds
    except subprocess.TimeoutExpired:
        print(f"Execution of {py_path_out} timed out.")
        return LoggingState.NO_DATA, py_log_dir
    except Exception as e:
        print(f"Error executing {py_path_out}: {e}")
        return LoggingState.NO_DATA, py_log_dir
        
    if verbose:
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

    # check if logging data is valid
    if is_valid_log_dir(py_log_dir, verbose=verbose):
        if verbose:
            print(f"Logging data saved to: {py_log_dir}")
        return LoggingState.HAVING_DATA, py_log_dir
    else:
        return LoggingState.NO_DATA, py_log_dir
