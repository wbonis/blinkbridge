"""Main application for BlinkBridge RTSP streaming service.

Manages the lifecycle of camera streams, monitors for motion detection,
and handles stream failures and restarts. Provides graceful shutdown handling.
"""
import asyncio
import logging
import signal
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from rich.highlighter import JSONHighlighter, NullHighlighter
from rich.logging import RichHandler

from blinkbridge.blink import CameraManager
from blinkbridge.config import *
from blinkbridge.ffmpeg import probe_stream_shape
from blinkbridge.stream_server import StreamServer
from blinkbridge.web import BlinkBridgeWebServer


log = logging.getLogger(__name__)

# Minimum poll interval enforced by BlinkPy API (seconds)
MIN_BLINK_THROTTLE = 2
# How often to log summary status at INFO level (seconds)
LOG_INTERVAL_SECONDS = 30
# Grace period for FFmpeg processes to shutdown cleanly (seconds)
SHUTDOWN_GRACE_PERIOD = 0.2


class CameraState(Enum):
    """Operational state of a single camera stream."""
    STARTING = "starting"  # Grey screen — stream up, waiting for first clip
    LIVE     = "live"      # Streaming real clip(s)
    OFFLINE  = "offline"   # Camera / sync module unreachable
    ERROR    = "error"     # Unknown / inconsistent state


class Application:
    """Main application that manages camera streams and monitors for motion.
    
    Coordinates CameraManager and StreamServer instances for each camera,
    handles motion detection polling, stream failures, and restarts.
    
    Attributes:
        stream_servers: Dict mapping camera names to StreamServer instances
        cam_manager: CameraManager instance for Blink integration
        running: Boolean flag indicating if application should continue running
    """
    
    def __init__(self) -> None:
        self.stream_servers: Dict[str, StreamServer] = {}
        self.cam_manager: Optional[CameraManager] = None
        self.running: bool = False
        self.web_server: Optional[BlinkBridgeWebServer] = None
        self._monitor_task: Optional[asyncio.Task] = None
        # Per-camera operational state and starting-poll counter
        self.camera_states: Dict[str, CameraState] = {}
        self.camera_starting_polls: Dict[str, int] = defaultdict(int)

    async def start_stream(self, camera_name: str, redownload: bool = False) -> Optional[StreamServer]:
        """Start a stream server for a camera using the Starting placeholder.

        The stream always begins with the grey "Starting..." screen.  The
        monitoring loop drives the transition to LIVE, OFFLINE, or ERROR once
        Blink state has been refreshed.

        Args:
            camera_name: Name of the camera.
            redownload: Unused — kept for call-site compatibility during restarts.

        Returns:
            StreamServer instance if the stream started successfully, else None.
        """
        if not self.running:
            log.debug(f"{camera_name}: skipping stream start (shutdown in progress)")
            return None

        # Fetch a clip before opening the stream. FFmpeg publishes with -c copy,
        # so it writes the RTSP SDP once from the very first file it plays and
        # never revises it; whatever shape that file has is what readers are
        # told the stream is, for as long as it runs. Having a clip on disk
        # first lets get_placeholder() build a Starting screen at this camera's
        # own resolution, frame rate and audio layout, so the announced stream
        # still matches once the camera goes LIVE. The monitoring loop would
        # fetch this clip a poll later anyway.
        try:
            await self.cam_manager.save_latest_clip(camera_name)
        except Exception as e:
            log.debug(f"{camera_name}: no clip available before stream start: {e}")

        starting_video = self.cam_manager.get_placeholder('starting', camera_name)
        if starting_video is None:
            log.error(f"{camera_name}: starting placeholder not available, cannot start stream")
            return None

        log.info(f"{camera_name}: starting stream (Starting screen)")
        try:
            stream_server = StreamServer(camera_name)
            stream_server.start_server(starting_video)
            self.stream_servers[camera_name] = stream_server
            self.camera_states[camera_name] = CameraState.STARTING
            self.camera_starting_polls[camera_name] = 0
            return stream_server
        except Exception as e:
            log.error(f"{camera_name}: failed to start stream server: {e}")
            return None

    async def check_for_motion(self, camera_name: str) -> bool:
        """Check for motion on a camera and add new clip to stream if detected.

        Returns True if a new clip was fetched and added to the stream.
        This method is now a thin wrapper kept for compatibility; the state
        machine in _update_camera_state drives all stream transitions.
        """
        try:
            ss = self.stream_servers.get(camera_name)
            if not ss or not ss.is_running():
                return False

            file_name_new_clip = await self.cam_manager.check_for_motion(camera_name)
            if not file_name_new_clip:
                return False

            log.debug(f"{ss.stream_name}: adding new clip to stream")
            ss.add_video(file_name_new_clip)
            return True
        except Exception as e:
            log.error(f"{camera_name}: error in check_for_motion: {e}", exc_info=True)
            return False
    
    async def start(self) -> None:
        """Start the application, initialize cameras, and begin monitoring.
        
        Raises:
            LoginError: If Blink authentication fails
            TokenRefreshFailed: If Blink token refresh fails
            Exception: For other critical initialization errors
        """
        self.running = True

        # Start the web server before camera login so the /2fa endpoint is
        # reachable if Blink requires two-factor authentication.
        try:
            await self._start_web_server()
        except Exception as e:
            log.warning(f"Failed to start web server: {e}")

        try:
            self.cam_manager = CameraManager()
            if self.web_server is not None:
                self.cam_manager.twofa_provider = self.web_server.request_2fa_code
                self.cam_manager.credentials_provider = self.web_server.request_credentials
            await self.cam_manager.start()
        except Exception as e:
            log.error(f"Failed to initialize camera manager: {e}")
            raise

        try:
            enabled_cameras = self._get_enabled_cameras()
            log.info(f"enabled cameras: {enabled_cameras}")
        except Exception as e:
            log.error(f"Failed to get enabled cameras: {e}")
            raise

        try:
            await self._initialize_camera_streams(enabled_cameras)
        except Exception as e:
            log.error(f"Error during camera stream initialization: {e}")
            # Continue even if some streams fail to initialize

        try:
            self._export_frigate_camera_block()
        except Exception as e:
            log.warning(f"Failed to export Frigate camera block: {e}")

        if self.running:
            try:
                self._monitor_task = asyncio.create_task(self._monitor_cameras())
                await self._monitor_task
            except asyncio.CancelledError:
                log.debug("Monitor task cancelled")
            except Exception as e:
                log.error(f"Error in camera monitoring loop: {e}")
                raise
    
    def _get_enabled_cameras(self) -> set:
        """Get the set of enabled cameras from config.
        
        Returns:
            Set of camera names that should be monitored
            
        Note:
            If CONFIG['cameras']['enabled'] is empty, enables all discovered cameras.
            Always excludes cameras in CONFIG['cameras']['disabled'].
        """
        if CONFIG['cameras']['enabled']:
            enabled_cameras = set(CONFIG['cameras']['enabled'])
        else:
            # Union in recently-known cameras from the clip cache, not just
            # Blink's live snapshot -- a camera whose whole sync module is
            # temporarily unreachable would otherwise never get a stream
            # server and could never be flagged OFFLINE (see
            # recently_known_cameras() docstring).
            enabled_cameras = set(self.cam_manager.get_cameras()) | self.cam_manager.recently_known_cameras()
        
        return enabled_cameras - set(CONFIG['cameras']['disabled'])
    
    async def _initialize_camera_streams(self, enabled_cameras: set) -> None:
        """Create stream servers for all enabled cameras.
        
        Args:
            enabled_cameras: Set of camera names to initialize
            
        Note:
            All cameras start with the grey "Starting..." placeholder.  The
            monitoring loop fetches clips and drives state transitions after
            all streams are up.

            Iterates enabled_cameras directly rather than filtering
            self.cam_manager.get_cameras() -- enabled_cameras can include
            cameras recently_known_cameras() recovered from the clip cache
            that are absent from Blink's live snapshot (e.g. a sync module
            that's down right now), and those still need a stream_servers
            entry at startup, not just on the first _discover_new_cameras()
            pass a poll cycle later.
        """
        for camera in sorted(enabled_cameras):
            if not self.running:
                log.info("Shutdown requested during startup, stopping stream creation")
                break

            ss = await self.start_stream(camera)
            if ss is None:
                log.warning(f"{camera}: failed to start stream")
                continue

            ss.failure_count = 0
            ss.datetime_started = datetime.now()
            await asyncio.sleep(0)

    def _export_frigate_camera_block(self) -> None:
        """Export a Frigate cameras YAML block for manual inclusion.

        This does not integrate with or control Frigate runtime. It only writes
        a snippet file users can paste/merge into their own Frigate config.
        """
        export_cfg = CONFIG.get('frigate_export', {})
        if not export_cfg.get('enabled', False):
            return

        if not self.cam_manager:
            raise RuntimeError("Camera manager not initialized")

        camera_names = sorted(set(self.cam_manager.get_cameras()))
        roles = list(export_cfg.get('roles', ['detect', 'record']))
        rtsp_host = str(export_cfg.get('rtsp_host', CONFIG['rtsp_server']['address']))
        rtsp_port = int(export_cfg.get('rtsp_port', CONFIG['rtsp_server']['port']))
        detect_defaults = dict(export_cfg.get('detect_defaults', {}))
        # No built-in fallback dimensions: an unset default means "let Frigate
        # work it out", which is safer than a guess that happens to be wrong
        # for this camera. See the per-camera loop below.
        width = detect_defaults.get('width')
        height = detect_defaults.get('height')
        width = int(width) if width else None
        height = int(height) if height else None
        fps = int(detect_defaults.get('fps', 1))

        lines = [
            "# Auto-generated by BlinkBridge.",
            "# Paste this block into your Frigate config under the top-level 'cameras:' key.",
            "cameras:",
        ]

        for camera_name in camera_names:
            camera_key = camera_name.replace(' ', '_').lower()

            # Detect geometry: measured beats configured beats omitted. Blink
            # models differ in resolution (720p through 1440p), so writing the
            # configured detect_defaults for every camera would hand Frigate
            # the wrong frame geometry for any camera that isn't that size --
            # and silently mask the very mismatch this export exists to
            # surface. Measure it from the camera's own most recent clip; fall
            # back to the configured defaults only when there is no clip to
            # measure; and when there is neither, write no dimensions at all so
            # Frigate reads them off the stream itself rather than trusting a
            # guess. detect fps stays configured: that's Frigate's analysis
            # rate, not the stream's.
            shape = probe_stream_shape(PATH_VIDEOS / f"{camera_key}_latest.mp4")
            if shape is not None:
                cam_width, cam_height = shape['width'], shape['height']
            else:
                cam_width, cam_height = width, height

            lines.append(f"  {camera_key}:")
            lines.append("    ffmpeg:")
            lines.append("      inputs:")
            lines.append(f"        - path: rtsp://{rtsp_host}:{rtsp_port}/{camera_key}")
            lines.append("          roles:")
            for role in roles:
                lines.append(f"            - {role}")
            lines.append("    detect:")
            lines.append("      enabled: true")
            if cam_width and cam_height:
                lines.append(f"      width: {cam_width}")
                lines.append(f"      height: {cam_height}")
            else:
                lines.append("      # width/height omitted -- Frigate reads them from the stream")
            lines.append(f"      fps: {fps}")

        output_path = Path(str(export_cfg.get('output_path', PATH_CONFIG / 'frigate_cameras.yml')))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n")
        log.info(f"Exported Frigate camera block for {len(camera_names)} cameras to {output_path}")

    async def _start_web_server(self) -> None:
        """Create and start the optional utility web server."""
        web_cfg = CONFIG.get('web', {})
        if not web_cfg.get('enabled', False):
            return

        export_cfg = CONFIG.get('frigate_export', {})
        export_path = Path(str(export_cfg.get('output_path', PATH_CONFIG / 'frigate_cameras.yml')))

        self.web_server = BlinkBridgeWebServer(
            host=str(web_cfg.get('host', '0.0.0.0')),
            port=int(web_cfg.get('port', 8765)),
            frigate_export_path=export_path,
        )
        self.web_server.restart_callback = self.restart
        await self.web_server.start()
        log.info(f"Web server enabled at http://{web_cfg.get('host', '0.0.0.0')}:{web_cfg.get('port', 8765)}")
    
    async def _monitor_cameras(self) -> None:
        """Main monitoring loop for camera motion detection.
        
        Continuously polls cameras for motion and manages stream health.
        Logs periodic status summaries at configured intervals.
        
        Note:
            Warns if poll_interval is less than BlinkPy's API throttle limit.
        """
        log.info(f"monitoring cameras for motion (poll interval: {CONFIG['blink']['poll_interval']}s)")
        
        if CONFIG['blink']['poll_interval'] < MIN_BLINK_THROTTLE:
            log.warning(
                f"poll_interval ({CONFIG['blink']['poll_interval']}s) is less than "
                f"BlinkPy's minimum throttle time ({MIN_BLINK_THROTTLE}s). "
                f"Effective poll rate will be ~{MIN_BLINK_THROTTLE}s due to API throttling."
            )
        
        poll_count = 0
        last_log_time = datetime.now()
        log_interval = timedelta(seconds=LOG_INTERVAL_SECONDS)
        
        while self.running:
            poll_count += 1
            log.debug(f"Poll #{poll_count}: checking {len(self.stream_servers)} cameras...")
            
            if datetime.now() - last_log_time >= log_interval:
                self._log_camera_status(poll_count)
                last_log_time = datetime.now()
            
            await self._check_cameras_for_updates()
            await self._discover_new_cameras()
            await self._restart_failed_streams()
            await asyncio.sleep(CONFIG['blink']['poll_interval'])
    
    def _log_camera_status(self, poll_count: int) -> None:
        """Log periodic status summary of cameras.
        
        Args:
            poll_count: Current poll iteration number
        """
        log.debug(
            f"Poll #{poll_count}: {len(self.stream_servers)} cameras active"
        )
    
    async def _discover_new_cameras(self) -> None:
        """Start streams for any enabled cameras that have appeared since startup.

        self.blink.cameras only reflects whatever Blink reported as of the
        last refresh() call, and the initial camera list is captured once
        at startup (_initialize_camera_streams) -- so a camera, or an
        entire sync module, that's down when the container starts is
        silently invisible for the rest of this process's life, even after
        it comes back online. This periodically re-checks for newly
        enabled-and-discovered cameras and starts a stream for each one,
        the same way _initialize_camera_streams does at startup.
        """
        try:
            enabled_cameras = self._get_enabled_cameras()
        except Exception as e:
            log.error(f"Error computing enabled cameras during discovery: {e}")
            return

        new_cameras = enabled_cameras - set(self.stream_servers.keys())
        if not new_cameras:
            return

        for camera_name in sorted(new_cameras):
            if not self.running:
                break
            log.info(f"{camera_name}: newly discovered (or returned) -- starting stream")
            ss = await self.start_stream(camera_name)
            if ss is None:
                log.warning(f"{camera_name}: failed to start stream for newly discovered camera")
                continue
            ss.failure_count = 0
            ss.datetime_started = datetime.now()

    async def _check_cameras_for_updates(self) -> None:
        """Run the state machine for every active camera stream.

        For each camera:
        1. Refresh Blink state and check for new motion clips.
        2. Check online/offline status.
        3. Transition the camera to STARTING / LIVE / OFFLINE / ERROR as
           appropriate and swap the stream content to the matching video.
        """
        for camera_name in list(self.stream_servers.keys()):
            if not self.running:
                break
            try:
                await self._update_camera_state(camera_name)
            except Exception as e:
                log.error(f"{camera_name}: critical error in state update: {e}", exc_info=True)
                try:
                    ss = self.stream_servers.get(camera_name)
                    if ss:
                        ss.close()
                except Exception as close_err:
                    log.error(f"{camera_name}: error closing stream after update failure: {close_err}")

    async def _update_camera_state(self, camera_name: str) -> None:
        """Drive the state machine for a single camera.

        State transition table:

        Current     offline   new_clip   Action
        -------     -------   --------   ------
        STARTING    yes       –          → OFFLINE (show offline screen)
        STARTING    no        yes        → LIVE    (add_video)
        STARTING    no        no         try historical clip; if found → LIVE
                                         else increment poll count;
                                         if count >= max_failures → ERROR
        LIVE        yes       –          → OFFLINE (show offline screen)
        LIVE        no        yes        stay LIVE (add_video)
        LIVE        no        no         stay LIVE (no change)
        OFFLINE     yes       –          stay OFFLINE (no change)
        OFFLINE     no        yes        → LIVE    (add_video)
        OFFLINE     no        no         → STARTING (show starting screen)
        ERROR       yes       –          → OFFLINE (show offline screen)
        ERROR       no        yes        → LIVE    (add_video)
        ERROR       no        no         try historical clip; if found → LIVE
                                         else stay ERROR
        """
        ss = self.stream_servers.get(camera_name)
        if not ss or not ss.is_running():
            return

        current_state = self.camera_states.get(camera_name, CameraState.STARTING)

        # --- Step 1: refresh Blink data and check for new motion clip ---
        new_clip: Optional[Path] = None
        refresh_ok = True
        try:
            new_clip = await self.cam_manager.check_for_motion(camera_name)
        except Exception as e:
            log.error(f"{camera_name}: error refreshing Blink data: {e}", exc_info=True)
            refresh_ok = False

        if not refresh_ok:
            # We can't make reliable state decisions without a fresh refresh.
            return

        # --- Step 2: read online status from the freshly-refreshed camera object ---
        is_offline = self.cam_manager.is_camera_offline(camera_name)

        # --- Step 3: state transitions ---

        # Helper shorthands. Resolved per camera so the placeholder matches
        # this camera's stream shape and swapping to it needs no publisher
        # restart (see CameraManager.get_placeholder).
        offline_video  = self.cam_manager.get_placeholder('offline', camera_name)
        starting_video = self.cam_manager.get_placeholder('starting', camera_name)
        error_video    = self.cam_manager.get_placeholder('error', camera_name)

        if is_offline:
            if current_state != CameraState.OFFLINE:
                log.info(f"{camera_name}: camera offline — showing OFFLINE screen (was {current_state.value})")
                if offline_video:
                    ss.swap_to_placeholder(offline_video)
                self.camera_states[camera_name] = CameraState.OFFLINE
                self.camera_starting_polls[camera_name] = 0
            return

        # Camera is online from here.

        if new_clip:
            if current_state != CameraState.LIVE:
                log.info(f"{camera_name}: clip received — going LIVE (was {current_state.value})")
            ss.add_video(new_clip)
            self.camera_states[camera_name] = CameraState.LIVE
            self.camera_starting_polls[camera_name] = 0
            return

        # Online, no new motion clip.

        if current_state == CameraState.LIVE:
            # Still online and streaming — nothing to do.
            return

        if current_state == CameraState.OFFLINE:
            # Just came back online; return to STARTING to wait for a clip.
            log.info(f"{camera_name}: back online — returning to STARTING")
            if starting_video:
                ss.swap_to_placeholder(starting_video)
            self.camera_states[camera_name] = CameraState.STARTING
            self.camera_starting_polls[camera_name] = 0
            return

        if current_state == CameraState.STARTING:
            # Try to pick up any historical clip from the cached metadata.
            try:
                clip = await self.cam_manager.save_latest_clip(camera_name)
                if clip is not None:
                    log.info(f"{camera_name}: found historical clip — going LIVE")
                    ss.add_video(clip)
                    self.camera_states[camera_name] = CameraState.LIVE
                    self.camera_starting_polls[camera_name] = 0
                    return
            except Exception as e:
                log.warning(f"{camera_name}: error checking for historical clip: {e}")

            # Still no clip — increment poll counter and check for ERROR threshold.
            count = self.camera_starting_polls[camera_name] + 1
            self.camera_starting_polls[camera_name] = count
            max_polls = CONFIG['cameras']['max_failures']
            if count >= max_polls:
                log.warning(
                    f"{camera_name}: online for {count} polls with no clip — going to ERROR"
                )
                if error_video:
                    ss.swap_to_placeholder(error_video)
                self.camera_states[camera_name] = CameraState.ERROR
            return

        if current_state == CameraState.ERROR:
            # Auto-recovery: check if a clip has become available since ERROR was set.
            try:
                clip = await self.cam_manager.save_latest_clip(camera_name)
                if clip is not None:
                    log.info(f"{camera_name}: recovered from ERROR — going LIVE")
                    ss.add_video(clip)
                    self.camera_states[camera_name] = CameraState.LIVE
                    self.camera_starting_polls[camera_name] = 0
            except Exception as e:
                log.warning(f"{camera_name}: error during ERROR recovery check: {e}")
            return
    
    async def _restart_failed_streams(self) -> None:
        """Restart any failed stream servers.
        
        Checks each stream server's health and attempts restart if needed.
        Disables cameras that exceed maximum failure count.
        Respects restart delay between attempts.
        """
        for camera_name in list(self.stream_servers.keys()):
            if not self.running:
                break
            
            try:
                ss = self.stream_servers[camera_name]
                if ss.is_running():
                    ss.failure_detected = False
                    continue

                # Log once when the failure is first detected.
                if not ss.failure_detected:
                    ss.failure_detected = True
                    log.warning(f"{camera_name}: stream stopped (failure count: {ss.failure_count + 1})")

                if ss.failure_count >= CONFIG['cameras']['max_failures'] - 1:
                    log.warning(f"{camera_name}: max failures ({CONFIG['cameras']['max_failures']}) reached, disabling")
                    try:
                        self.stream_servers.pop(camera_name)
                    except KeyError:
                        log.debug(f"{camera_name}: already removed from stream servers")
                    continue

                if datetime.now() < ss.datetime_started + DELAY_RESTART:
                    log.debug(f"{camera_name}: waiting for restart delay to elapse")
                    continue

                log.warning(f"{camera_name}: attempting restart (failure {ss.failure_count + 1})")
                ss.failure_detected = False  # reset so the next failure logs again

                ss_new = await self.start_stream(camera_name)
                if ss_new is None:
                    log.debug(f"{camera_name}: restart failed, will retry later")
                    ss.datetime_started = datetime.now()
                    continue
                
                ss_new.failure_count = ss.failure_count + 1
                ss_new.datetime_started = datetime.now()
                log.info(f"{camera_name}: stream restarted successfully")
            except Exception as e:
                log.error(f"{camera_name}: error during stream restart: {e}", exc_info=True)

    async def close(self) -> None:
        """Close the application and stop all streams.
        
        Stops all  stream servers, waits for graceful FFmpeg shutdown,
        and closes the camera manager connection.
        """
        log.info("Closing application and stopping all streams...")
        log.info("Note: FFmpeg 'Broken pipe' errors during shutdown are normal")
        self.running = False

        for camera_name, ss in list(self.stream_servers.items()):
            try:
                log.debug(f"{camera_name}: stopping stream")
                ss.close()
            except Exception as e:
                log.warning(f"{camera_name}: error stopping stream: {e}")
        
        await asyncio.sleep(SHUTDOWN_GRACE_PERIOD)

        if self.web_server:
            try:
                await self.web_server.stop()
            except Exception as e:
                log.warning(f"Error stopping web server: {e}")

        # Remove the Frigate camera snippet so a stale file is never served
        # after the bridge goes offline.
        try:
            export_cfg = CONFIG.get('frigate_export', {})
            export_path = Path(str(export_cfg.get('output_path', PATH_CONFIG / 'frigate_cameras.yml')))
            if export_path.exists():
                export_path.unlink()
                log.debug(f"Removed Frigate export file: {export_path}")
        except Exception as e:
            log.warning(f"Failed to remove Frigate export file on shutdown: {e}")
        
        if self.cam_manager:
            try:
                await self.cam_manager.close()
            except Exception as e:
                log.warning(f"Error closing camera manager: {e}")
        
        log.info("Application closed")

    async def restart(self) -> None:
        """Restart camera streams and Blink connection without stopping the web server.

        Tears down all running streams and the camera manager, then re-initialises
        them from scratch. The web server keeps running throughout so the /restart
        endpoint remains reachable.
        """
        log.info("Restarting BlinkBridge (streams + Blink connection)...")

        # Cancel the running monitor loop
        self.running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        # Stop all streams
        for camera_name, ss in list(self.stream_servers.items()):
            try:
                ss.close()
            except Exception as e:
                log.warning(f"{camera_name}: error stopping stream during restart: {e}")
        self.stream_servers.clear()

        # Remove stale Frigate export
        try:
            export_cfg = CONFIG.get('frigate_export', {})
            export_path = Path(str(export_cfg.get('output_path', PATH_CONFIG / 'frigate_cameras.yml')))
            if export_path.exists():
                export_path.unlink()
        except Exception as e:
            log.warning(f"Failed to remove Frigate export file during restart: {e}")

        # Close the old camera manager
        if self.cam_manager:
            try:
                await self.cam_manager.close()
            except Exception as e:
                log.warning(f"Error closing camera manager during restart: {e}")
            self.cam_manager = None

        # Re-initialise
        self.running = True
        self.camera_states.clear()
        self.camera_starting_polls.clear()
        try:
            self.cam_manager = CameraManager()
            if self.web_server is not None:
                self.cam_manager.twofa_provider = self.web_server.request_2fa_code
                self.cam_manager.credentials_provider = self.web_server.request_credentials
            await self.cam_manager.start()
        except Exception as e:
            log.error(f"Restart: failed to initialise camera manager: {e}")
            return

        try:
            enabled_cameras = self._get_enabled_cameras()
            await self._initialize_camera_streams(enabled_cameras)
        except Exception as e:
            log.error(f"Restart: error initialising camera streams: {e}")

        try:
            self._export_frigate_camera_block()
        except Exception as e:
            log.warning(f"Restart: failed to export Frigate camera block: {e}")

        log.info("Restart complete — resuming camera monitoring")
        # Start a fresh monitoring task.
        self._monitor_task = asyncio.create_task(self._monitor_cameras())

async def main() -> None:
    """Main entry point for the application.
    
    Sets up signal handlers, starts the application, and handles graceful shutdown.
    """
    app = Application()
    shutdown_event = asyncio.Event()

    def handle_exit() -> None:
        """Signal handler for SIGINT and SIGTERM."""
        log.info("Shutdown signal received...")
        app.running = False
        shutdown_event.set()

    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, handle_exit)
    except Exception as e:
        log.error(f"Failed to set up signal handlers: {e}")
        raise

    try:
        start_task = asyncio.create_task(app.start())
        await shutdown_event.wait()
        
        start_task.cancel()
        try:
            await start_task
        except asyncio.CancelledError:
            log.debug("Start task cancelled successfully")
        except Exception as e:
            log.error(f"Error in start task: {e}", exc_info=True)

    except KeyboardInterrupt:
        log.info("Keyboard interrupt received")
    except Exception as e:
        log.error(f"Unexpected error in main: {e}", exc_info=True)
    finally:
        try:
            await app.close()
        except Exception as e:
            log.error(f"Error during application cleanup: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s", datefmt="[%X]", handlers=[RichHandler(highlighter=NullHighlighter())]
    )
    logging.getLogger('blinkbridge').setLevel(CONFIG['log_level'])
    logging.getLogger(__name__).setLevel(CONFIG['log_level'])
    
    asyncio.run(main())

