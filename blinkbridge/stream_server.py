"""RTSP streaming server management using FFmpeg.

Provides the StreamServer class for managing FFmpeg-based RTSP streams.
Handles video concatenation, still video creation, and stream lifecycle.
"""
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Union

from blinkbridge.config import *
from blinkbridge.ffmpeg import (
    FrameToVideo,
    StillVideoCreator,
    StreamParameters,
    probe_stream_shape,
    sdp_fields,
)
from blinkbridge.utils import get_socket_states, wait_until_file_open


log = logging.getLogger(__name__)

class StreamServer:
    """Manages RTSP streaming of Blink camera videos using FFmpeg.
    
    Creates and manages an FFmpeg process that streams video via RTSP protocol.
    Uses FFmpeg's concat demuxer to seamlessly loop and update videos. Maintains
    a "still video" created from the last frame of clips for smooth looping.
    
    Attributes:
        stream_name: Human-readable camera name
        stream_name_sanitized: URL-safe version of stream name
        current_still_video: Path to the current still video file
        process: FFmpeg subprocess handle
    """
    
    def __init__(self, stream_name: str):
        """Initialize a new stream server.
        
        Args:
            stream_name: Name of the camera/stream
        """
        self.stream_name: str = stream_name
        self.stream_name_sanitized: str = stream_name.replace(' ', '_').lower()
        self.current_still_video: Optional[Path] = None
        self.process: Optional[subprocess.Popen] = None
        self.failure_detected: bool = False  # True after first failure spotted, until restart attempted
        # Stream shape the running FFmpeg publisher wrote its RTSP SDP from.
        self.published_shape: Optional[Dict] = None

    def _run_server(self) -> str:
        """Start the FFmpeg RTSP streaming process.
        
        Returns:
            RTSP URL where the stream is available
            
        Raises:
            FileNotFoundError: If FFmpeg is not found
            Exception: If subprocess creation fails
            
        Note:
            FFmpeg reads from a concat file that loops infinitely (-stream_loop -1).
            The concat file itself references another concat file that can be
            dynamically updated to add new clips.
        """
        output_url = f"{RTSP_URL}/{self.stream_name_sanitized}"
        input_concat_file = PATH_CONCAT / f"{self.stream_name_sanitized}.concat"

        if not input_concat_file.exists():
            raise FileNotFoundError(f"Concat file not found: {input_concat_file}")

        ffmpeg_args = [
            'ffmpeg', *COMMON_FFMPEG_ARGS,
            '-fflags', '+igndts+genpts',
            '-re',
            '-stream_loop', '-1',
            '-f', 'concat', '-safe', '0',
            '-i', str(input_concat_file.resolve()),
            '-flush_packets', '0',
            '-c:v', 'copy', '-c:a', 'copy',
            # FFmpeg's RTSP muxer defaults to 1472-byte packets, above
            # MediaMTX's 1440 limit, so it logs "RTP packets are too big
            # (1460 > 1440), remuxing them into smaller ones" on every path and
            # then repacks every oversized packet for the life of the stream.
            # That cost is paid by the same component that discards frames when
            # a reader falls behind, and it is avoidable by not producing
            # oversized packets in the first place.
            '-pkt_size', '1200',
            '-f', 'rtsp',
            '-fps_mode', 'drop',
            output_url
        ]
        
        try:
            self.process = subprocess.Popen(ffmpeg_args, stdout=sys.stdout, stderr=sys.stderr)
            log.debug(f"{self.stream_name}: FFmpeg process started (PID: {self.process.pid})")
        except FileNotFoundError:
            log.error(f"{self.stream_name}: FFmpeg not found. Please ensure FFmpeg is installed and in PATH")
            raise
        except Exception as e:
            log.error(f"{self.stream_name}: failed to start FFmpeg process: {e}")
            raise
            
        return output_url

    def _make_concat_files(self) -> Path:
        """Create the main concat file that loops the next concat file.
        
        Returns:
            Path to the created main concat file
            
        Raises:
            IOError: If file creation fails
            
        Note:
            Creates a two-level concat structure:
            - Main concat file: loops and references next.concat
            - Next concat file: contains the actual video to play (updated dynamically)
            
            The 'safe 0' option is propagated to allow absolute paths.
        """
        next_concat = PATH_CONCAT / f"{self.stream_name_sanitized}_next.concat"
        concat_file = PATH_CONCAT / f"{self.stream_name_sanitized}.concat"

        try:
            # Ensure directory exists
            PATH_CONCAT.mkdir(parents=True, exist_ok=True)
            
            with open(concat_file, 'w') as f:
                f.write("ffconcat version 1.0\n")
                # Reference next concat file twice for seamless looping
                for _ in range(2):
                    f.write(f"file '{next_concat.resolve()}'\n")
                    f.write("option safe 0\n")  # Allow absolute paths
        except IOError as e:
            log.error(f"{self.stream_name}: failed to create concat file: {e}")
            raise
        except Exception as e:
            log.error(f"{self.stream_name}: unexpected error creating concat file: {e}")
            raise

        return concat_file

    def _enqueue_clip(self, video_file_name: Union[str, Path]) -> Path:
        """Add a video clip to the next concat file.
        
        Args:
            video_file_name: Path to the video file to add to the stream
            
        Returns:
            Path to the updated next concat file
            
        Raises:
            FileNotFoundError: If video file doesn't exist
            IOError: If concat file cannot be written
            
        Note:
            Overwrites the next concat file with the new video. FFmpeg's concat
            demuxer will automatically switch to the new file when it loops.
        """
        video_file_name = Path(video_file_name)
        
        # Verify video file exists
        try:
            if not video_file_name.exists():
                raise FileNotFoundError(f"Video file not found: {video_file_name}")
        except OSError as e:
            log.error(f"{self.stream_name}: error checking video file: {e}")
            raise
        
        next_concat = PATH_CONCAT / f"{self.stream_name_sanitized}_next.concat"

        try:
            with open(next_concat, 'w') as f:
                f.write("ffconcat version 1.0\n")
                f.write(f"file '{video_file_name.resolve()}'\n")
        except IOError as e:
            log.error(f"{self.stream_name}: failed to write next concat file: {e}")
            raise
        except Exception as e:
            log.error(f"{self.stream_name}: unexpected error enqueueing clip: {e}")
            raise

        self._reconcile_published_shape(video_file_name)

        return next_concat

    def _reconcile_published_shape(self, video_file_name: Path) -> None:
        """Warn when a newly queued clip no longer matches the published stream.

        FFmpeg publishes with -c copy, so it derives the RTSP SDP from the first
        file it opens and never revises it. When a later file in the concat
        stream has a different resolution, frame rate, H264 profile/level or
        audio layout, the SDP keeps describing the old one and readers decode
        against the wrong parameters -- which is what makes Frigate drop the
        session and reconnect in a loop.

        Restarting FFmpeg here would re-publish with a correct SDP, but an
        in-place restart proved unreliable: the replacement publisher goes
        quiet after a few seconds and sits in CLOSE_WAIT, alive enough that
        is_running() keeps reporting it healthy while the path is gone. So this
        only reports the mismatch. The mismatch is instead avoided upstream:
        placeholders are built per camera from that camera's own clip (see
        CameraManager.get_placeholder), and start_stream() fetches a clip
        before opening the stream so the publisher starts at the right shape.
        What remains is a camera that has never produced a clip and later does,
        which this logs so the operator can restart it deliberately.
        """
        shape = probe_stream_shape(video_file_name)
        if shape is None:
            return

        if self.published_shape is None:
            self.published_shape = shape
            return

        if sdp_fields(shape) == sdp_fields(self.published_shape):
            self.published_shape = shape
            return

        log.warning(
            f"{self.stream_name}: stream shape changed "
            f"({self._describe_shape(self.published_shape)} -> {self._describe_shape(shape)}); "
            f"the published RTSP description still announces the old shape, "
            f"restart this stream to republish it"
        )
        self.published_shape = shape

    @staticmethod
    def _describe_shape(shape: Dict) -> str:
        """Render a stream shape for log output."""
        return (
            f"{shape['width']}x{shape['height']}@{shape['fps']} "
            f"{shape['profile']}/{shape['level']} "
            f"{shape['audio_rate']}Hz/{shape['audio_channels']}ch"
        )

    def add_video(self, file_name_input_video: Union[str, Path], still_only: bool=False,
                  defer_still: bool=False) -> None:
        """Add a video to the stream and create a still video from its last frame.
        
        Args:
            file_name_input_video: Path to the input video
            still_only: If True, only create still video without enqueueing the
                full clip first. Used for initial stream setup (default: False)
            defer_still: If True, build the still but leave the clip queued
                instead of replacing it. The caller is then responsible for
                calling swap_in_still() once the clip has played as often as
                it wants. Used to give a downstream detector more than one
                pass over the footage (default: False)
                
        Raises:
            Exception: If still video creation fails
            FileNotFoundError: If still video file wasn't created
            
        Note:
            Process flow:
            1. Enqueue full clip (unless still_only)
            2. Start creating still video in background
            3. Wait for FFmpeg to open the full clip (if enqueued)
            4. Wait for still video creation to complete
            5. Enqueue still video
            6. Delete previous still video
        """
        try:
            file_name_input_video = Path(file_name_input_video)
            
            # Verify input video exists
            if not file_name_input_video.exists():
                raise FileNotFoundError(f"Input video not found: {file_name_input_video}")
                
            if not still_only:
                self._enqueue_clip(file_name_input_video)
        except FileNotFoundError as e:
            log.error(f"{self.stream_name}: {e}")
            raise
        except Exception as e:
            log.error(f"{self.stream_name}: error enqueueing video: {e}")
            raise

        # Create timestamped filename for still video
        dt = datetime.now()
        next_still_video = PATH_VIDEOS / f"{self.stream_name_sanitized}_still_{dt.strftime('%Y-%m-%d_%H-%M-%S-%f')}.mp4"
        try:
            # Ensure videos directory exists
            PATH_VIDEOS.mkdir(parents=True, exist_ok=True)
            
            svc = StillVideoCreator(
                file_name_input_video,
                output_duration=CONFIG['still_video_duration'],
                file_name_still_video=next_still_video
            )
            
            if not still_only:
                log.debug(f"{self.stream_name}: waiting for new video to start")
                try:
                    if self.process is None:
                        log.warning(f"{self.stream_name}: process not started, cannot wait for video to open")
                    else:
                        wait_until_file_open(file_name_input_video, self.process.pid)
                except TimeoutError as e:
                    log.warning(f"{self.stream_name}: timeout waiting for video to open: {e}")
                    # Continue anyway - video might still work
                except Exception as e:
                    log.warning(f"{self.stream_name}: error waiting for video to open: {e}")
            
            log.debug(f'{self.stream_name}: waiting for still video creation to finish')
            svc.wait()
            
            if not next_still_video.exists():
                raise FileNotFoundError(f"Still video was not created: {next_still_video}")
            
            if next_still_video.stat().st_size == 0:
                raise ValueError(f"Still video is empty: {next_still_video}")
                
            if not defer_still:
                self._enqueue_clip(next_still_video)

            if self.current_still_video:
                try:
                    self.current_still_video.unlink()
                except OSError as e:
                    log.warning(f"{self.stream_name}: failed to delete old still video: {e}")
                except Exception as e:
                    log.warning(f"{self.stream_name}: unexpected error deleting old still video: {e}")
            
            self.current_still_video = next_still_video
        except Exception as e:
            log.error(
                f"{self.stream_name}: Failed to create still video from {file_name_input_video}: {e}",
                exc_info=True,
            )
            try:
                if next_still_video.exists():
                    next_still_video.unlink()  # Clean up failed still video
            except Exception as cleanup_err:
                log.warning(f"{self.stream_name}: failed to cleanup still video: {cleanup_err}")
            raise
    
    def refresh_still_from_image(
        self, image_file: Union[str, Path], shape_source: Union[str, Path]
    ) -> bool:
        """Replace the looping still with one built from an arbitrary image.

        The still normally comes from the last frame of the last motion clip,
        so between events it shows whatever happened to be in frame when that
        clip ended -- for hours, until the next event. This puts a freshly
        taken snapshot up instead.

        Stream parameters are taken from shape_source (the camera's own most
        recent clip), never from the image, so the new still matches what the
        publisher already announced in the RTSP SDP -- the same invariant
        get_placeholder() maintains. The image is scaled to the stream's
        resolution: Blink thumbnails are half the video's dimensions, so the
        result is softer than a still cut from the video itself.

        Args:
            image_file: JPEG to build the still from.
            shape_source: Video whose parameters the still must match.

        Returns:
            True if the still was replaced, False if it was left as it was.
        """
        image_file = Path(image_file)
        shape_source = Path(shape_source)

        if not image_file.exists():
            log.warning(f"{self.stream_name}: snapshot image not found: {image_file}")
            return False
        if not shape_source.exists():
            log.debug(f"{self.stream_name}: no clip to take still parameters from")
            return False

        dt = datetime.now()
        next_still_video = PATH_VIDEOS / f"{self.stream_name_sanitized}_still_{dt.strftime('%Y-%m-%d_%H-%M-%S-%f')}.mp4"

        try:
            params_audio, params_video = StreamParameters(shape_source).wait()
            if not params_video:
                raise ValueError(f"no H264 stream in {shape_source}")

            FrameToVideo(
                image_file, params_video, params_audio,
                output_duration=CONFIG['still_video_duration'],
                file_name_output_video=next_still_video,
            ).wait()

            if not next_still_video.exists() or next_still_video.stat().st_size == 0:
                raise FileNotFoundError(f"still video not created: {next_still_video}")
        except Exception as e:
            log.warning(f"{self.stream_name}: could not build still from snapshot: {e}")
            try:
                next_still_video.unlink(missing_ok=True)
            except Exception:
                pass
            return False

        self._enqueue_clip(next_still_video)

        previous = self.current_still_video
        self.current_still_video = next_still_video
        if previous and previous != next_still_video:
            try:
                previous.unlink()
            except OSError as e:
                log.debug(f"{self.stream_name}: could not delete previous still: {e}")

        return True

    def swap_in_still(self) -> None:
        """Put the current still back on the stream after a deferred clip.

        Pairs with add_video(defer_still=True): the clip stays queued and the
        concat loop repeats it until this is called. Safe to call when there is
        no still -- that just leaves the clip playing.
        """
        if not self.current_still_video:
            log.debug(f"{self.stream_name}: no still to swap in")
            return
        if not self.current_still_video.exists():
            log.warning(
                f"{self.stream_name}: still {self.current_still_video.name} is gone, "
                f"leaving the clip queued"
            )
            return
        self._enqueue_clip(self.current_still_video)

    def _sweep_orphaned_files(self) -> None:
        """Delete any still video and temp frame files left by a previous run."""
        for pattern in (
            f"{self.stream_name_sanitized}_still_*.mp4",
            f"{self.stream_name_sanitized}_still_*.frame.jpg",
        ):
            for path in PATH_VIDEOS.glob(pattern):
                try:
                    path.unlink()
                    log.debug(f"{self.stream_name}: removed orphaned file {path.name}")
                except OSError as e:
                    log.warning(f"{self.stream_name}: could not remove orphaned file {path.name}: {e}")

    # /proc/net/tcp state codes. Only the terminal ones justify a restart:
    # a socket still negotiating is not a failure, it is a publisher that has
    # not finished connecting.
    TCP_ESTABLISHED = '01'
    TCP_CONNECTING = frozenset({'02', '03'})            # SYN_SENT, SYN_RECV
    TCP_DEAD = frozenset({'04', '05', '06', '07', '08', '09', '0B'})

    def is_publisher_connected(self) -> Optional[bool]:
        """Whether the publisher still holds a live connection to the RTSP server.

        is_running() only asks whether the process exists, and FFmpeg outlives
        its connection: when the RTSP server drops the publisher, the socket
        sits in CLOSE_WAIT and the process keeps running with nothing to write
        to. The path disappears while every check the watchdog performs says
        the stream is healthy.

        That is not hypothetical. On this deployment a stream published to a
        file that had been deleted underneath it, FFmpeg failed demuxing,
        MediaMTX dropped the publisher, and the camera stayed dark for ten
        minutes without a single warning being logged -- because poll() kept
        returning None the whole time.

        Returns:
            True if at least one socket is ESTABLISHED, False if the process
            holds sockets and none of them is, and None when there is nothing
            to conclude from -- no sockets yet, or /proc unreadable. None must
            not be treated as a failure: a publisher that has only just started
            has no socket either.
        """
        if self.process is None:
            return None

        try:
            states = get_socket_states(self.process.pid)
        except Exception as e:
            log.debug(f"{self.stream_name}: could not read socket states: {e}")
            return None

        if not states:
            return None

        if self.TCP_ESTABLISHED in states:
            return True

        if any(s in self.TCP_CONNECTING for s in states):
            # Mid-handshake. Saying "dead" here would restart a publisher that
            # is merely still coming up.
            return None

        if all(s in self.TCP_DEAD for s in states):
            return False

        # Some other state (LISTEN, or something unrecognised). Not a
        # publisher socket we can reason about, so do not act on it.
        return None

    def is_running(self) -> bool:
        """Check if the streaming process is still running.
        
        Returns:
            True if FFmpeg process is active, False otherwise
        """
        return self.process is not None and self.process.poll() is None
    
    def close(self) -> None:
        """Stop the streaming process gracefully.
        
        Attempts graceful termination (SIGTERM) first, then forces kill (SIGKILL)
        if the process doesn't stop within 1 second.
        """
        if not self.is_running():
            log.debug(f"{self.stream_name}: process not running, nothing to close")
            return
            
        log.debug(f"{self.stream_name}: stopping stream server")
        try:
            try:
                self.process.terminate()
            except ProcessLookupError:
                log.debug(f"{self.stream_name}: process already terminated")
                return
            except Exception as e:
                log.warning(f"{self.stream_name}: error terminating process: {e}")
                return
                
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                    self.process.wait()
                except Exception as e:
                    log.warning(f"{self.stream_name}: error killing process: {e}")
        except Exception as e:
            log.warning(f"{self.stream_name}: unexpected error during shutdown: {e}")

    def swap_to_placeholder(self, placeholder_video: Union[str, Path]) -> None:
        """Immediately switch the stream to a static placeholder video.

        Enqueues the placeholder directly into the next concat file so the
        stream transitions without restarting FFmpeg.  The currently queued
        still video is left in place so there is no gap.

        Args:
            placeholder_video: Path to the placeholder video file to enqueue.
        """
        placeholder_video = Path(placeholder_video)
        if not placeholder_video.exists():
            log.warning(f"{self.stream_name}: placeholder video not found: {placeholder_video}")
            return
        log.debug(f"{self.stream_name}: swapping to placeholder {placeholder_video.name}")
        self._enqueue_clip(placeholder_video)

    def start_server(self, file_name_initial_video: Union[str, Path]) -> None:
        """Initialize and start the RTSP stream server.
        
        Args:
            file_name_initial_video: Path to the first video to stream
            
        Raises:
            FileNotFoundError: If initial video or FFmpeg not found
            Exception: If server initialization fails
            
        Note:
            Creates concat files, generates initial still video, and starts
            the FFmpeg RTSP streaming process.
        """
        try:
            file_name_initial_video = Path(file_name_initial_video)
            if not file_name_initial_video.exists():
                raise FileNotFoundError(f"Initial video not found: {file_name_initial_video}")
        except OSError as e:
            log.error(f"{self.stream_name}: error accessing initial video: {e}")
            raise
        
        try:
            self._make_concat_files()
        except Exception as e:
            log.error(f"{self.stream_name}: failed to create concat files: {e}")
            raise

        # Remove any orphaned still/frame files left by a previous crashed run.
        self._sweep_orphaned_files()

        try:
            self.add_video(file_name_initial_video, still_only=True)
        except Exception as e:
            log.error(f"{self.stream_name}: failed to add initial video: {e}")
            raise
            
        try:
            url = self._run_server()
            log.info(f"{self.stream_name}: stream ready at {url}")
        except Exception as e:
            log.error(f"{self.stream_name}: failed to start server: {e}")
            raise

    