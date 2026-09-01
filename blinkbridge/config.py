"""Configuration management for BlinkBridge.

Handles loading and parsing of configuration files, and provides global
configuration variables used throughout the application.
"""
import json
import logging
import os
import sys
from pathlib import Path
from datetime import timedelta
from typing import Union


log = logging.getLogger(__name__)


__all__ = [
    'COMMON_FFMPEG_ARGS',
    'CONFIG',
    'DELAY_RESTART',
    'RTSP_URL',
    'PATH_VIDEOS',
    'PATH_CONCAT',
    'PATH_CONFIG'
]

# Common FFmpeg arguments used across all FFmpeg operations
COMMON_FFMPEG_ARGS = ['-hide_banner', '-loglevel', 'error', '-y']

# Global configuration dictionary loaded from JSON file
CONFIG = None
# Delay between stream restart attempts (timedelta)
DELAY_RESTART = None
# RTSP server URL string
RTSP_URL = None
# Path to videos directory
PATH_VIDEOS = None
# Path to FFmpeg concat files directory
PATH_CONCAT = None
# Path to configuration files directory
PATH_CONFIG = None

def load_config_file(file_name: Union[str, Path]) -> None:
    """Load configuration from a JSON file and initialize global config variables.
    
    Args:
        file_name: Path to the configuration JSON file
        
    Raises:
        FileNotFoundError: If the config file doesn't exist
        json.JSONDecodeError: If the config file contains invalid JSON
        KeyError: If required configuration keys are missing
        ValueError: If configuration values are invalid
    """
    global CONFIG, DELAY_RESTART, RTSP_URL, PATH_VIDEOS, PATH_CONCAT, PATH_CONFIG

    file_name = Path(file_name)
    
    # Check if config file exists
    try:
        if not file_name.exists():
            print(f"ERROR: Configuration file not found: {file_name}", file=sys.stderr)
            print(f"Please create a configuration file at {file_name.resolve()}", file=sys.stderr)
            raise FileNotFoundError(f"Configuration file not found: {file_name}")
    except OSError as e:
        print(f"ERROR: Cannot access configuration file: {e}", file=sys.stderr)
        raise
    
    # Read and parse JSON
    try:
        with open(file_name, 'r') as f:
            CONFIG = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in configuration file: {e}", file=sys.stderr)
        print(f"Please check the syntax in {file_name}", file=sys.stderr)
        raise
    except IOError as e:
        print(f"ERROR: Cannot read configuration file: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error loading configuration: {e}", file=sys.stderr)
        raise
    
    # Validate and extract required configuration
    try:
        # Validate camera configuration
        if 'cameras' not in CONFIG:
            raise KeyError("Missing 'cameras' section in configuration")
        if 'restart_delay_seconds' not in CONFIG['cameras']:
            raise KeyError("Missing 'cameras.restart_delay_seconds' in configuration")
        
        DELAY_RESTART = timedelta(seconds=CONFIG['cameras']['restart_delay_seconds'])
        
        # Validate RTSP server configuration
        if 'rtsp_server' not in CONFIG:
            raise KeyError("Missing 'rtsp_server' section in configuration")
        if 'address' not in CONFIG['rtsp_server'] or 'port' not in CONFIG['rtsp_server']:
            raise KeyError("Missing RTSP server address or port in configuration")
        
        RTSP_URL = f"rtsp://{CONFIG['rtsp_server']['address']}:{CONFIG['rtsp_server']['port']}"

        # Validate paths configuration
        if 'paths' not in CONFIG:
            raise KeyError("Missing 'paths' section in configuration")
        if 'videos' not in CONFIG['paths']:
            raise KeyError("Missing 'paths.videos' in configuration")
        if 'concat' not in CONFIG['paths']:
            raise KeyError("Missing 'paths.concat' in configuration")
        if 'config' not in CONFIG['paths']:
            raise KeyError("Missing 'paths.config' in configuration")
        
        PATH_VIDEOS = Path(CONFIG['paths']['videos'])
        PATH_CONCAT = Path(CONFIG['paths']['concat'])
        PATH_CONFIG = Path(CONFIG['paths']['config'])
        
        # Create directories if they don't exist
        try:
            PATH_VIDEOS.mkdir(parents=True, exist_ok=True)
            PATH_CONCAT.mkdir(parents=True, exist_ok=True)
            PATH_CONFIG.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"ERROR: Failed to create required directories: {e}", file=sys.stderr)
            raise
        
        # Validate other required settings
        if 'blink' not in CONFIG:
            raise KeyError("Missing 'blink' section in configuration")
        # login credentials are optional — they can be supplied at runtime via
        # the web UI instead of being stored in the config file.
        CONFIG['blink'].setdefault('login', {})
        CONFIG['blink']['login'].setdefault('username', '')
        CONFIG['blink']['login'].setdefault('password', '')
        
        # Set defaults for optional settings
        CONFIG.setdefault('log_level', 'INFO')
        CONFIG.setdefault('still_video_duration', 0.5)
        # How many times a motion clip plays before the still takes over. 1 is
        # the historical behaviour: the still replaced the clip about a second
        # after it was queued, so the clip played exactly once. A detector
        # downstream gets only that single pass to recognise anything, which
        # measured too short here -- Frigate produced no event from a 15 s clip
        # while producing them from the same clip left looping.
        CONFIG.setdefault('clip_repeats', 3)
        # Frame rate to encode the looping still at. None inherits the clip's
        # rate. Forcing it low decouples the still's length from its encode
        # cost, which matters because a longer still is what keeps the
        # publisher from crossing a concat boundary several times a second.
        CONFIG.setdefault('still_video_fps', None)
        CONFIG['cameras'].setdefault('enabled', [])
        CONFIG['cameras'].setdefault('disabled', [])
        CONFIG['cameras'].setdefault('max_failures', 3)
        # Per-camera publish downscale, {camera_name: max_height}. A camera
        # whose native resolution is too high to sustain through the
        # mediamtx->reader pipe (1440p stalls to ~0 fps on this host) is
        # re-encoded down to the given height by the publisher; cameras absent
        # here keep -c:v copy. See StreamServer._video_publish_args.
        CONFIG['cameras'].setdefault('transcode_max_height', {})
        CONFIG['blink'].setdefault('history_days', 90)
        CONFIG['blink'].setdefault('poll_interval', 1)
        CONFIG['blink'].setdefault('metadata_pages', 10)

        # Optional export of Frigate cameras block (snippet only)
        CONFIG.setdefault('frigate_export', {})
        CONFIG['frigate_export'].setdefault('enabled', False)
        CONFIG['frigate_export'].setdefault('output_path', str(PATH_CONFIG / 'frigate_cameras.yml'))
        CONFIG['frigate_export'].setdefault('rtsp_host', CONFIG['rtsp_server']['address'])
        CONFIG['frigate_export'].setdefault('rtsp_port', CONFIG['rtsp_server']['port'])
        CONFIG['frigate_export'].setdefault('roles', ['detect', 'record'])
        CONFIG['frigate_export'].setdefault('detect_defaults', {})
        # No width/height defaults on purpose: the export measures each camera's
        # real resolution and only falls back to these. Injecting 1280x720 here
        # would turn "unset" into a guess that is wrong for any other model.
        CONFIG['frigate_export']['detect_defaults'].setdefault('fps', 1)

        # Optional periodic refresh of the looping still from a fresh camera
        # snapshot. Each refresh wakes the camera, which costs battery on the
        # battery-powered models, so nothing runs unless an interval is set:
        # with the section absent, default_interval_minutes is 0 and per_camera
        # is empty, which disables it for every camera on its own. 'enabled'
        # therefore defaults to True rather than False -- it is a master switch
        # for turning a configured setup off, and defaulting it to False would
        # only mean that a config setting nothing but intervals silently does
        # nothing.
        CONFIG.setdefault('snapshot_refresh', {})
        CONFIG['snapshot_refresh'].setdefault('enabled', True)
        CONFIG['snapshot_refresh'].setdefault('default_interval_minutes', 0)
        CONFIG['snapshot_refresh'].setdefault('per_camera', {})

        # Optional burn-in of the clip's Blink-cloud recording time (top-left).
        # Off by default: 'still' is nearly free (the still is re-encoded
        # anyway), but 'clip' forces a re-encode of every motion clip, which
        # otherwise streams with -c:v copy. 'format' is a strftime pattern
        # applied to the recording time converted to the container's local TZ.
        CONFIG.setdefault('timestamp_overlay', {})
        CONFIG['timestamp_overlay'].setdefault('still', False)
        CONFIG['timestamp_overlay'].setdefault('clip', False)
        CONFIG['timestamp_overlay'].setdefault('format', '%d.%m.%Y %H:%M:%S')

        # Optional BlinkBridge utility web server
        CONFIG.setdefault('web', {})
        CONFIG['web'].setdefault('enabled', False)
        CONFIG['web'].setdefault('host', '0.0.0.0')
        CONFIG['web'].setdefault('port', 8765)

    except KeyError as e:
        print(f"ERROR: Configuration validation failed: {e}", file=sys.stderr)
        print(f"Please check your configuration file at {file_name}", file=sys.stderr)
        raise
    except (TypeError, ValueError) as e:
        print(f"ERROR: Invalid configuration value: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error processing configuration: {e}", file=sys.stderr)
        raise

# Load configuration from environment variable or default location
try:
    config_file = os.getenv('BLINKBRIDGE_CONFIG', 'config.json')
    load_config_file(config_file)
except Exception as e:
    print(f"\nFATAL ERROR: Failed to load configuration: {e}\n", file=sys.stderr)
    sys.exit(1)
 
