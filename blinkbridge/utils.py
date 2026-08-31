"""Utility functions for process and file system operations.

Provides functions to interact with Linux /proc filesystem for monitoring
process file descriptors and waiting for specific file operations.
"""
import logging
import os
import time
from pathlib import Path
from typing import List, Union


log = logging.getLogger(__name__)


def get_pids_by_name(process_name: str) -> List[int]:
    """Get all process IDs with the given process name.
    
    Searches /proc filesystem for processes matching the specified name.
    
    Args:
        process_name: Name of the process to search for (from /proc/[pid]/comm)
        
    Returns:
        List of process IDs matching the name. Empty list if none found.
        
    Example:
        >>> pids = get_pids_by_name('ffmpeg')
        >>> print(pids)
        [1234, 5678]
    """
    pids = []
    
    try:
        proc_path = Path('/proc')
        if not proc_path.exists():
            log.error("/proc filesystem not available")
            return pids
    except Exception as e:
        log.error(f"Error accessing /proc: {e}")
        return pids

    try:
        for pid_dir in proc_path.iterdir():
            try:
                if not (pid_dir.is_dir() and pid_dir.name.isdigit()):
                    continue
            except (OSError, PermissionError):
                continue
                
            try:
                comm_file = pid_dir / 'comm'
                with open(comm_file, 'r') as f:
                    comm = f.read().strip()
                    if comm == process_name:
                        pids.append(int(pid_dir.name))
            except FileNotFoundError:
                # Process may have terminated between directory listing and file read
                continue
            except PermissionError:
                # No permission to read this process
                continue
            except (ValueError, OSError) as e:
                # Invalid PID or other OS error
                log.debug(f"Error reading process {pid_dir.name}: {e}")
                continue
    except Exception as e:
        log.error(f"Error scanning /proc directory: {e}")

    return pids

def get_open_files(pid: int) -> List[Path]:
    """Get all files currently opened by a process.
    
    Reads the /proc/[pid]/fd directory to find all open file descriptors.
    
    Args:
        pid: Process ID to check
        
    Returns:
        List of Path objects for open files. Empty list if process not found
        or no files are open.
        
    Note:
        Symbolic links in /proc/[pid]/fd point to the actual file paths.
        This function resolves those links to get the real paths.
    """
    file_names = []
    
    try:
        fd_dir = Path(f'/proc/{pid}/fd')

        if not fd_dir.exists() or not fd_dir.is_dir():
            log.debug(f"Process {pid} not found or fd directory not accessible")
            return file_names
    except OSError as e:
        log.debug(f"Error accessing /proc/{pid}/fd: {e}")
        return file_names
    
    try:
        for fd in fd_dir.iterdir():
            try:
                resolved_path = fd.resolve()
                file_names.append(resolved_path)
            except FileNotFoundError:
                # FD may have been closed between listing and resolving
                continue
            except PermissionError:
                # No permission to read this FD
                continue
            except (OSError, RuntimeError) as e:
                # Other errors (e.g., broken symlink, too many levels)
                log.debug(f"Error resolving fd {fd}: {e}")
                continue
    except PermissionError:
        log.debug(f"No permission to list open files for process {pid}")
    except OSError as e:
        log.debug(f"Error listing open files for process {pid}: {e}")
        
    return file_names
 
def is_file_open(process_name: str, file_name: Union[str, Path]) -> bool:
    """Check if a file is currently open by any process with the given name.
    
    Useful for verifying that a media file is actively being read/written.
    
    Args:
        process_name: Name of the process(es) to check
        file_name: Path to the file to check
        
    Returns:
        True if the file is open by any matching process, False otherwise.
        
    Example:
        >>> is_file_open('ffmpeg', '/tmp/video.mp4')
        True
    """
    try:
        file_name = Path(file_name).resolve()
    except (OSError, RuntimeError) as e:
        log.error(f"Error resolving file path {file_name}: {e}")
        return False
    
    try:
        pids = get_pids_by_name(process_name)
    except Exception as e:
        log.error(f"Error getting PIDs for process {process_name}: {e}")
        return False

    for pid in pids:
        try:
            open_files = get_open_files(pid)
            if file_name in open_files:
                return True
        except Exception as e:
            log.debug(f"Error checking open files for PID {pid}: {e}")
            continue
                
    return False

def get_socket_states(pid: int) -> List[str]:
    """TCP states of the sockets a process holds, as /proc/net/tcp spells them.

    Returns hex state codes ('01' = ESTABLISHED, '08' = CLOSE_WAIT, ...). An
    empty list means "nothing could be determined" -- no sockets yet, or /proc
    unreadable -- which callers must distinguish from "sockets exist and none
    is healthy". A publisher that has not connected yet also has none.
    """
    inodes = set()
    try:
        fd_dir = Path(f'/proc/{pid}/fd')
        for fd in fd_dir.iterdir():
            try:
                target = os.readlink(fd)
            except (OSError, PermissionError):
                continue
            if target.startswith('socket:['):
                inodes.add(target[len('socket:['):-1])
    except (OSError, PermissionError) as e:
        log.debug(f"Could not list sockets for process {pid}: {e}")
        return []

    if not inodes:
        return []

    states = []
    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            with open(table) as f:
                next(f, None)  # header
                for line in f:
                    parts = line.split()
                    if len(parts) > 9 and parts[9] in inodes:
                        states.append(parts[3])
        except (OSError, StopIteration) as e:
            log.debug(f"Could not read {table}: {e}")

    return states


def wait_until_file_open(file_path: Union[str, Path], pid: int, timeout: float=10.0, poll_interval: float=0.1) -> float:
    """Wait until a file is opened by a specific process.
    
    Polls the process's open file descriptors until the specified file appears.
    Used to synchronize video streaming operations.
    
    Args:
        file_path: Path to the file to monitor
        pid: Process ID to check
        timeout: Maximum time to wait in seconds (default: 10.0)
        poll_interval: How often to check in seconds (default: 0.1)
        
    Returns:
        Time elapsed in seconds until file was opened.
        
    Raises:
        TimeoutError: If timeout is reached before file is opened by the process.
        ValueError: If timeout or poll_interval are invalid
        
    Example:
        >>> elapsed = wait_until_file_open('/tmp/video.mp4', 1234, timeout=5.0)
        >>> print(f"File opened after {elapsed:.2f} seconds")
        File opened after 0.47 seconds
    """
    if timeout <= 0:
        raise ValueError(f"Invalid timeout: {timeout} (must be > 0)")
    if poll_interval <= 0:
        raise ValueError(f"Invalid poll_interval: {poll_interval} (must be > 0)")
    
    try:
        file_path = Path(file_path).resolve()
    except (OSError, RuntimeError) as e:
        log.error(f"Error resolving file path {file_path}: {e}")
        raise ValueError(f"Invalid file path: {e}")
    
    start_time = time.time()

    while time.time() - start_time <= timeout:
        try:
            open_files = get_open_files(pid)
            
            if file_path in open_files:
                elapsed = time.time() - start_time
                log.debug(f"File {file_path} opened by process {pid} after {elapsed:.2f}s")
                return elapsed
        except Exception as e:
            log.debug(f"Error checking if file is open (will retry): {e}")

        try:
            time.sleep(poll_interval)
        except Exception as e:
            log.error(f"Error during sleep: {e}")
            break

    # Debug, not error: the only caller treats expiry as a normal outcome and
    # logs it at warning with the camera name attached. Logging it as an error
    # here says "something broke" about a wait that is expected to expire.
    log.debug(f"Timeout waiting for process {pid} to open {file_path} after {timeout}s")
    raise TimeoutError(
        f"process {pid} did not open {file_path} within {timeout}s -- "
        f"the publisher plays the concat in real time, so it only opens a newly "
        f"queued file once the current one has finished; a clip longer than "
        f"{timeout}s still playing will exceed this"
    )