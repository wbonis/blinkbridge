"""Blink camera integration and video clip management.

Provides the CameraManager class for authenticating with Blink cameras,
downloading video clips, and monitoring for motion detection events.
"""
import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Tuple, Union

from aiohttp import ClientSession
from blinkpy import api as blink_api
from blinkpy.auth import Auth, BlinkTwoFARequiredError, TokenRefreshFailed, LoginError
from blinkpy.blinkpy import Blink
from blinkpy.helpers.util import json_load

from blinkbridge.config import *
from blinkbridge.ffmpeg import (
    generate_placeholder_video,
    is_usable_clip,
    normalize_clip_container,
    probe_stream_shape,
    sdp_fields,
)


# How long to wait for a camera to upload a freshly taken snapshot before
# giving up: attempts x delay. Blink cameras typically need a few seconds.
SNAPSHOT_POLL_ATTEMPTS = 6
SNAPSHOT_POLL_DELAY_SECONDS = 2.0


log = logging.getLogger(__name__)


def _apply_blinkpy_oauth_compat_patch() -> None:
    """Patch BlinkPy OAuth signin to handle TSV challenge responses.

    Blink's OAuth endpoint may return HTTP 202 with TSV metadata instead of
    HTTP 412 for secondary verification. BlinkPy 0.25.5 treats only 412 as
    2FA-required, which causes login to fail even when verification can
    continue. This compatibility patch maps 202 to the same 2FA flow.
    """
    if getattr(blink_api, "_blinkbridge_oauth_signin_patched", False):
        return

    async def oauth_signin_compat(auth, email, password, csrf_token):
        headers = {
            "User-Agent": blink_api.OAUTH_USER_AGENT,
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "https://api.oauth.blink.com",
            "Referer": blink_api.OAUTH_SIGNIN_URL,
        }
        data = {
            "username": email,
            "password": password,
            "csrf-token": csrf_token,
        }

        response = await auth.session.post(
            blink_api.OAUTH_SIGNIN_URL, headers=headers, data=data, allow_redirects=False
        )

        if response.status in (412, 202):
            return "2FA_REQUIRED"
        if response.status in (301, 302, 303, 307, 308):
            return "SUCCESS"

        return None

    blink_api.oauth_signin = oauth_signin_compat
    blink_api._blinkbridge_oauth_signin_patched = True


def find_most_recent_clip_url(recent_clips: dict, date: str) -> str:
    """Find the most recent non-snapshot clip URL that is newer than the given date.
    
    Args:
        recent_clips: Dictionary of recent clips from Blink camera
        date: ISO format date string to compare against
        
    Returns:
        URL of the most recent clip, or empty string if none found
        
    Note:
        Filters out snapshots (which contain '/snapshot/' in the URL) and only
        returns actual video clips that are newer than the specified date.
    """
    sorted_data = sorted(recent_clips, key=lambda x: x['time'], reverse=True)

    # Find first entry that is not a snapshot
    clip_entry = next((entry for entry in sorted_data if '/snapshot/' not in entry['clip']), None)
    if not clip_entry:
        return ''
    
    # Check if entry is newer than the given date
    date = datetime.fromisoformat(date.replace('Z', '+00:00'))
    entry_time = datetime.fromisoformat(clip_entry['time'].replace('Z', '+00:00'))
    
    return clip_entry['clip'] if entry_time > date else '' 

class CameraManager:
    """Manages Blink camera connections and video clip downloads.
    
    Handles authentication, metadata management, clip downloads, and motion detection
    for Blink camera systems. Maintains state about which cameras have clips available
    and provides black video placeholders for cameras without recorded content.
    
    Attributes:
        session: aiohttp ClientSession for HTTP requests
        blink: BlinkPy Blink instance
        camera_last_record: Dict tracking last recorded event per camera
        clip_cache: Dict mapping camera name to its most-recent clip metadata
        black_video_path: Path to black placeholder video
    """
    
    def __init__(self) -> None:
        self.session: ClientSession = ClientSession()
        self.camera_last_record: Dict[str, Optional[str]] = defaultdict(lambda: None)
        self.clip_cache: Dict[str, dict] = {}
        self.last_metadata_refresh: Optional[datetime] = None
        self.black_video_path: Optional[Path] = None
        self.starting_placeholder_path: Optional[Path] = None
        self.offline_placeholder_path: Optional[Path] = None
        self.error_placeholder_path: Optional[Path] = None
        # Per-camera placeholder sets, plus the stream shape each was built for
        # so they can be rebuilt if a camera's clips ever change shape.
        self.camera_placeholders: Dict[str, Dict[str, Path]] = {}
        self.camera_placeholder_shapes: Dict[str, Dict] = {}
        self.twofa_provider: Optional[Callable[[], Awaitable[str]]] = None
        self.credentials_provider: Optional[Callable[[], Awaitable[dict]]] = None

    async def _login(self) -> None:
        """Login to Blink using OAuth v2 authentication.

        Tries saved credentials first. If they are stale or invalid the cache
        file is removed and a fresh login is attempted. The fresh login may
        require 2FA; if so, the code is collected via ``twofa_provider`` (web
        UI) when available, otherwise the terminal ``input()`` is used.

        Raises:
            LoginError: If authentication fails
            TokenRefreshFailed: If token refresh fails

        Note:
            Credentials are saved to .cred.json in the config directory for reuse.
        """
        path_cred = PATH_CONFIG / ".cred.json"

        # Apply compatibility patch for Blink OAuth signin status handling.
        _apply_blinkpy_oauth_compat_patch()

        # If no credentials are present in config, ask for them via the provider
        # (web UI) before attempting any login — but only when there is no saved
        # credential cache, since the cache is self-contained and doesn't need a
        # username/password to refresh.
        login_cfg = CONFIG['blink']['login']
        if not login_cfg.get('username') and not path_cred.exists():
            if self.credentials_provider is not None:
                log.info("No Blink credentials in config — requesting via web UI")
                creds = await self.credentials_provider()
                login_cfg = creds
            else:
                raise LoginError(
                    "No Blink credentials configured and no credentials provider available. "
                    "Set blink.login in config.json or enable the web server."
                )

        # Attempt login with saved credentials first; on failure delete the
        # cache file and retry once with fresh credentials. If credentials come
        # from the web UI, re-prompt on failure rather than crashing.
        use_saved = path_cred.exists()
        for attempt in range(10):  # generous upper bound; breaks on success or hard error
            self.blink = Blink(session=self.session)
            if use_saved:
                log.debug("Loading saved Blink credentials")
                try:
                    saved_data = await json_load(path_cred)
                    self.blink.auth = Auth(saved_data, no_prompt=True, session=self.session)
                except (json.JSONDecodeError, IOError) as e:
                    log.debug(f"Failed to load saved credentials: {e}, falling back to config")
                    self.blink.auth = Auth(login_cfg, no_prompt=True, session=self.session)
            else:
                log.debug("Using Blink credentials from config")
                self.blink.auth = Auth(login_cfg, no_prompt=True, session=self.session)

            try:
                started = await self.blink.start()
                if not started or not getattr(self.blink, 'available', False):
                    raise LoginError(
                        "Blink platform setup failed after authentication. "
                        "Check credentials/2FA and rerun initialization."
                    )
                log.info("Successfully authenticated with Blink")
                break  # success — exit retry loop
            except BlinkTwoFARequiredError:
                log.info("Two-factor authentication required")
                if self.twofa_provider is not None:
                    twofa_code = await self.twofa_provider()
                else:
                    twofa_code = input("Enter your 2FA code: ")
                success = await self.blink.send_2fa_code(twofa_code)
                if not success:
                    raise LoginError("2FA verification failed")
                log.info("Successfully authenticated with Blink (2FA completed)")
                break  # success after 2FA — exit retry loop
            except (TokenRefreshFailed, LoginError) as e:
                if use_saved:
                    # Stale / invalid saved credentials — discard and retry fresh.
                    log.warning(f"Saved credentials rejected ({e}), retrying with fresh credentials")
                    try:
                        path_cred.unlink()
                    except OSError as unlink_err:
                        log.warning(f"Failed to remove stale credentials file: {unlink_err}")
                    use_saved = False
                    # If config has no username we need to ask for credentials now.
                    if not login_cfg.get('username'):
                        if self.credentials_provider is not None:
                            log.info("Stale cache removed, no credentials in config — requesting via web UI")
                            login_cfg = await self.credentials_provider()
                        else:
                            raise LoginError(
                                "Saved credentials are invalid and no fallback credentials are available. "
                                "Set blink.login in config.json or enable the web server."
                            )
                    continue  # retry the loop
                # Fresh credentials (from config or web UI) were also rejected.
                if not login_cfg.get('username') or self.credentials_provider is not None:
                    # Credentials came from the web UI — re-prompt rather than crash.
                    log.warning(f"Credentials rejected ({e}), re-prompting via web UI")
                    if self.credentials_provider is not None:
                        login_cfg = await self.credentials_provider()
                        continue
                log.error(f"Authentication failed: {e}")
                raise
            except Exception as e:
                log.error(f"Unexpected error during authentication: {e}")
                raise

        try:
            log.debug("Saving Blink credentials")
            await self.blink.save(path_cred)
        except (IOError, OSError) as e:
            log.warning(f"Failed to save credentials (will need to re-authenticate next time): {e}")

    def _generate_black_video(self, width: int = 1920, height: int = 1080) -> Optional[Path]:
        """Generate a black video file to use as placeholder for cameras without clips.
        
        Args:
            width: Video width in pixels (default: 1920)
            height: Video height in pixels (default: 1080)
            
        Returns:
            Path to the generated black video file, or None if generation failed
            
        Note:
            Uses FFmpeg to create a video with black frames and silent audio.
            The duration matches CONFIG['still_video_duration'].
        """
        import subprocess
        
        black_video_path = PATH_VIDEOS / "_black_placeholder.mp4"
        
        try:
            if black_video_path.exists():
                log.debug(f"Black video already exists at {black_video_path}")
                return black_video_path
        except OSError as e:
            log.error(f"Error checking if black video exists: {e}")
            return None
        
        duration = CONFIG['still_video_duration']
        ffmpeg_cmd = [
            'ffmpeg', *COMMON_FFMPEG_ARGS,
            '-f', 'lavfi', '-i', f'color=black:s={width}x{height}:d={duration}',
            '-f', 'lavfi', '-i', f'anullsrc=channel_layout=stereo:sample_rate=44100',
            '-c:v', 'libx264', '-profile:v', 'high', '-level:v', '4.1',
            '-c:a', 'aac', '-ar', '44100', '-ac', '2', '-b:a', '128k',
            '-t', str(duration), '-pix_fmt', 'yuv420p', '-movflags', 'faststart',
            '-video_track_timescale', str(CONCAT_VIDEO_TIMESCALE),
            str(black_video_path)
        ]
        
        log.debug(f"Generating black placeholder video ({width}x{height}, {duration}s)")
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, timeout=30)
        except subprocess.TimeoutExpired:
            log.error("FFmpeg timed out while generating black video")
            return None
        except FileNotFoundError:
            log.error("FFmpeg not found. Please ensure FFmpeg is installed and in PATH")
            return None
        except Exception as e:
            log.error(f"Unexpected error running FFmpeg: {e}")
            return None
        
        if result.returncode != 0:
            stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else 'No error output'
            log.error(f"Failed to generate black video (exit code {result.returncode}): {stderr}")
            return None
        
        try:
            if not black_video_path.exists():
                log.error(f"Black video was not created at {black_video_path}")
                return None
        except OSError as e:
            log.error(f"Error verifying black video creation: {e}")
            return None
        
        log.debug(f"Black placeholder video created at {black_video_path}")
        return black_video_path
    
    def is_camera_offline(self, camera_name: str) -> bool:
        """Return True if the camera or its sync module is offline.

        Checks both the camera's own online status and its parent sync module's
        status. A camera cannot be considered online if its sync module is offline,
        even if the camera object itself reports online (blinkpy may not propagate
        sync-module outages to individual camera objects in all cases).

        A camera missing entirely from self.blink.cameras (e.g. its whole sync
        module dropped off Blink's cloud) is treated as offline -- that's a
        stronger signal than a camera being present but merely flagged offline,
        not a reason to be optimistic. Only self.blink itself not being ready
        (AttributeError, e.g. mid-(re)login) stays optimistic, so a startup race
        doesn't falsely flag every camera as offline.
        """
        try:
            camera = self.blink.cameras[camera_name]
        except KeyError:
            return True
        except AttributeError:
            return False

        try:
            camera_online = camera.online
        except Exception:
            camera_online = True  # optimistic

        if not camera_online:
            return True

        try:
            sync = getattr(camera, 'sync', None)
            if sync is not None:
                sync_online = sync.online
                if not sync_online:
                    return True
        except Exception:
            pass  # optimistic: don't mark offline if we can't read sync state

        return False

    # (filename stem, on-screen text, background colour, text colour)
    PLACEHOLDER_SPECS = {
        'starting': ('starting_placeholder', 'Starting...', 'gray',  'white'),
        'offline':  ('offline_placeholder',  'OFFLINE',     'black', 'red'),
        'error':    ('error_placeholder',    'ERROR',       'black', 'red'),
    }

    async def snap_and_fetch_thumbnail(self, camera_name: str) -> Optional[Path]:
        """Take a fresh photo on the camera and save it as a JPEG.

        Blink only refreshes a camera's thumbnail on its own events, so the
        cached one can be hours old -- measured at 123 minutes on both cameras
        of the instance this was written for, which is why simply re-fetching
        the existing thumbnail does not help. Only snap_picture() produces a
        current image, and it wakes the camera to do so. That costs battery on
        the battery-powered models, so callers must rate-limit; see
        Application._refresh_still_if_due().

        Returns:
            Path to the saved JPEG, or None if the camera is unknown, the
            snapshot did not arrive in time, or the file could not be written.
        """
        camera = self.blink.cameras.get(camera_name) if self.blink.cameras else None
        if camera is None:
            log.debug(f"{camera_name}: not in Blink\'s camera list, cannot take a snapshot")
            return None

        thumbnail_before = camera.thumbnail

        try:
            await camera.snap_picture()
        except Exception as e:
            log.warning(f"{camera_name}: snap_picture failed: {e}")
            return None

        # The camera uploads asynchronously, so the thumbnail URL only changes a
        # few seconds later. Poll for a different URL rather than saving the old
        # image again and believing it is new.
        for _ in range(SNAPSHOT_POLL_ATTEMPTS):
            await asyncio.sleep(SNAPSHOT_POLL_DELAY_SECONDS)
            try:
                await self.blink.refresh(force=True)
            except Exception as e:
                log.debug(f"{camera_name}: refresh while waiting for snapshot failed: {e}")
                continue
            if camera.thumbnail and camera.thumbnail != thumbnail_before:
                break
        else:
            log.debug(
                f"{camera_name}: thumbnail unchanged {SNAPSHOT_POLL_ATTEMPTS * SNAPSHOT_POLL_DELAY_SECONDS:.0f}s "
                f"after snap_picture, leaving the current still in place"
            )
            return None

        camera_name_sanitized = camera_name.replace(' ', '_').lower()
        snapshot_path = PATH_VIDEOS / f"{camera_name_sanitized}_snapshot.jpg"
        try:
            await camera.image_to_file(str(snapshot_path))
        except Exception as e:
            log.warning(f"{camera_name}: could not save snapshot image: {e}")
            return None

        try:
            if not snapshot_path.exists() or snapshot_path.stat().st_size == 0:
                log.warning(f"{camera_name}: snapshot image is missing or empty")
                return None
        except OSError as e:
            log.warning(f"{camera_name}: could not check snapshot image: {e}")
            return None

        return snapshot_path

    def _generate_placeholders(self) -> None:
        """Generate the shared fallback placeholder videos.

        Creates:
        - starting_placeholder.mp4 : grey screen with white "Starting..." text
        - offline_placeholder.mp4  : black screen with red "OFFLINE" text
        - error_placeholder.mp4    : black screen with red "ERROR" text

        These are 1920x1080 H264 at 15 fps and are only used for a camera that
        has not produced a clip yet, since there is nothing to match against at
        that point. Once a clip exists, get_placeholder() serves a per-camera
        set built to that camera's stream shape instead.
        """
        duration = CONFIG['still_video_duration']
        for state, (stem, text, bg, fg) in self.PLACEHOLDER_SPECS.items():
            path = PATH_VIDEOS / f"{stem}.mp4"
            attr = f"{state}_placeholder_path"
            if path.exists():
                log.debug(f"Placeholder already exists: {path}")
                setattr(self, attr, path)
                continue
            ok = generate_placeholder_video(
                output_path=path,
                text=text,
                bg_color=bg,
                text_color=fg,
                width=1920,
                height=1080,
                fps=15,
                duration=duration,
            )
            if ok:
                setattr(self, attr, path)
            else:
                log.error(f"Failed to generate placeholder video: {path.name}")

    def get_placeholder(self, state: str, camera_name: str) -> Optional[Path]:
        """Return the placeholder video for a camera state, matched to that camera.

        Placeholders share a concat stream with the camera's real clips, and
        FFmpeg publishes that stream with -c copy, so the RTSP SDP is written
        once from whatever plays first and never updated. A placeholder whose
        resolution, frame rate, H264 profile/level or audio layout differs from
        the camera's clips therefore leaves the published stream describing
        something other than what it carries, which breaks readers such as
        Frigate. So each camera gets its own placeholder set, built from the
        shape of its most recent clip.

        Before a camera has downloaded a clip there is nothing to match, so the
        shared 1920x1080 fallback is returned; StreamServer restarts the
        publisher when the first real clip changes the stream shape, which
        re-writes the SDP.

        Args:
            state: One of 'starting', 'offline', 'error'.
            camera_name: Camera the placeholder is for.

        Returns:
            Path to a placeholder video, or None if none could be produced.
        """
        fallback = getattr(self, f"{state}_placeholder_path", None)
        if state not in self.PLACEHOLDER_SPECS:
            log.warning(f"{camera_name}: unknown placeholder state '{state}'")
            return fallback

        camera_name_sanitized = camera_name.replace(' ', '_').lower()
        reference = PATH_VIDEOS / f"{camera_name_sanitized}_latest.mp4"
        if not reference.exists():
            return fallback

        shape = probe_stream_shape(reference)
        if shape is None:
            return fallback

        # Compared on SDP fields, not the whole shape. Blink's clips vary in
        # frame rate between recordings, and rebuilding on that would re-encode
        # three placeholders at full camera resolution every time a clip with a
        # slightly different rate arrives -- which is exactly when a motion clip
        # is being spliced and the publisher needs the CPU. Frame rate is not in
        # the SDP and the concat stream already carries mixed rates from the
        # clips themselves, so matching it buys nothing.
        known = self.camera_placeholder_shapes.get(camera_name)
        if known is None or sdp_fields(known) != sdp_fields(shape):
            self._generate_camera_placeholders(camera_name, camera_name_sanitized, shape)

        return self.camera_placeholders.get(camera_name, {}).get(state, fallback)

    def _generate_camera_placeholders(
        self, camera_name: str, camera_name_sanitized: str, shape: Dict
    ) -> None:
        """Build this camera's placeholder set for the given stream shape.

        Called only when the shape differs from the set already on disk, so a
        camera that keeps producing clips of the same shape encodes its
        placeholders once.
        """
        duration = CONFIG['still_video_duration']
        generated: Dict[str, Path] = {}

        for state, (stem, text, bg, fg) in self.PLACEHOLDER_SPECS.items():
            path = PATH_VIDEOS / f"{camera_name_sanitized}_{stem}.mp4"
            ok = generate_placeholder_video(
                output_path=path,
                text=text,
                bg_color=bg,
                text_color=fg,
                duration=duration,
                **shape,
            )
            if ok:
                generated[state] = path
            else:
                log.error(f"{camera_name}: failed to generate {state} placeholder")

        if len(generated) == len(self.PLACEHOLDER_SPECS):
            self.camera_placeholders[camera_name] = generated
            self.camera_placeholder_shapes[camera_name] = shape
            log.info(
                f"{camera_name}: placeholders rebuilt for {shape['width']}x{shape['height']} "
                f"@ {shape['fps']}, audio {shape['audio_rate']}Hz/{shape['audio_channels']}ch"
            )
        else:
            # Partial sets would reintroduce the mismatch on whichever state
            # failed, so discard and keep using the shared fallback.
            self.camera_placeholders.pop(camera_name, None)
            self.camera_placeholder_shapes.pop(camera_name, None)

    def _detect_resolution_from_clips(self) -> Tuple[int, int]:
        """Detect resolution from clips. Returns default Blink resolution (1920x1080).
        
        Returns:
            Tuple of (width, height) in pixels
            
        Note:
            Currently returns hardcoded 1920x1080 as all Blink cameras use this resolution.
            Could be extended to detect actual resolution from clip metadata.
        """
        return (1920, 1080)
    
    def _clip_cache_path(self) -> Path:
        return PATH_CONFIG / "clip_cache.json"

    def _load_clip_cache(self) -> None:
        """Load the persistent per-camera clip cache from disk, if present."""
        path = self._clip_cache_path()
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    self.clip_cache = json.load(f)
                log.debug(f"Loaded clip cache from {path} ({len(self.clip_cache)} cameras)")
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load clip cache from {path}: {e} -- starting fresh")
            self.clip_cache = {}

    def _save_clip_cache(self) -> None:
        """Persist the per-camera clip cache to disk so it survives restarts."""
        path = self._clip_cache_path()
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.clip_cache, f)
        except IOError as e:
            log.warning(f"Failed to save clip cache to {path}: {e}")

    def _merge_clips(self, items: list) -> int:
        """Merge fetched clip metadata into the per-camera cache.

        Keeps only the single most recent non-deleted, non-snapshot clip per
        camera -- this is what makes the cache immune to being crowded out
        by high-motion cameras: a quiet camera's one cached entry is never
        evicted by another camera's clips, no matter how many there are.

        Returns:
            Number of cameras whose cached entry was updated.
        """
        updated = 0
        for m in items or []:
            if m.get('deleted') or m.get('source') == 'snapshot':
                continue
            name = m.get('device_name')
            if not name:
                continue
            existing = self.clip_cache.get(name)
            if existing is None or m.get('created_at', '') > existing.get('created_at', ''):
                self.clip_cache[name] = m
                updated += 1
        return updated

    async def refresh_metadata(self) -> None:
        """Refresh the per-camera clip cache from the Blink API.

        Maintains a persistent cache (clip_cache.json in the config directory)
        mapping each camera to its single most-recent known clip. This avoids
        the crowding-out problem of a shared, re-derived-every-poll pooled
        clip list: on an account with many cameras and highly uneven motion
        activity, a few high-motion cameras can otherwise fill the entire
        fetched window, permanently hiding quieter cameras' most recent clip
        even though it's well within CONFIG['blink']['history_days'].

        - First run (empty cache): a one-time deep seed fetch, paginating
          from now until every currently known camera has a cached entry or
          a safety page cap is hit (whichever comes first).
        - Subsequent runs: a small incremental fetch of only what's new
          since the last successful refresh (CONFIG['blink']['metadata_pages']
          pages, plenty for a single poll interval's worth of new clips),
          merged into the existing cache. Cameras that stayed quiet keep
          their previously cached entry indefinitely.

        Raises:
            Exception: If the API call fails
        """
        SEED_PAGE_CAP = 200  # ~5000 clips; safety ceiling so one silent camera can't stall startup forever
        INCREMENTAL_OVERLAP = timedelta(minutes=5)  # cheap insurance against clock drift / race conditions

        if not self.clip_cache:
            self._load_clip_cache()

        known_cameras = set(self.blink.cameras.keys()) if self.blink.cameras else set()
        seeding = not self.clip_cache

        try:
            if seeding:
                log.info("No cached clips found -- performing one-time deep seed fetch")
                dt_past = datetime.now(timezone.utc) - timedelta(days=CONFIG['blink']['history_days'])
                page = 1
                fetched_total = 0
                while page < SEED_PAGE_CAP:
                    response = await blink_api.request_videos(self.blink, time=dt_past.timestamp(), page=page)
                    try:
                        result = response["media"]
                        if not result:
                            break
                    except (KeyError, TypeError):
                        break
                    fetched_total += len(result)
                    self._merge_clips(result)
                    page += 1
                    if known_cameras and known_cameras <= self.clip_cache.keys():
                        log.info(
                            f"Seed fetch covered all {len(known_cameras)} known cameras "
                            f"after {page - 1} pages ({fetched_total} clips)"
                        )
                        break
                else:
                    missing = known_cameras - self.clip_cache.keys()
                    log.warning(
                        f"Seed fetch hit the {SEED_PAGE_CAP}-page cap with cameras still "
                        f"uncovered: {sorted(missing)} -- they'll get a cached clip on their "
                        f"next real motion event instead"
                    )
            else:
                floor_dt = datetime.now(timezone.utc) - timedelta(days=CONFIG['blink']['history_days'])
                since_dt = max((self.last_metadata_refresh or floor_dt) - INCREMENTAL_OVERLAP, floor_dt)
                stop = CONFIG['blink']['metadata_pages'] + 1  # BlinkPy uses range(1, stop)
                new_items = await self.blink.get_videos_metadata(since=str(since_dt), stop=stop)
                updated = self._merge_clips(new_items)
                log.debug(f"Incremental metadata refresh: {updated} camera(s) updated")

            self._save_clip_cache()
            self.last_metadata_refresh = datetime.now(timezone.utc)
            log.debug(f"Clip cache covers {len(self.clip_cache)} camera(s): {sorted(self.clip_cache.keys())}")
        except Exception as e:
            log.error(f"Failed to refresh video metadata: {e}")
            raise

    async def save_latest_clip(self, camera_name: str, force: bool=False) -> Optional[Path]:
        """Download and save latest clip for camera.
        
        Args:
            camera_name: Name of the camera
            force: Force re-download even if clip exists (default: False)
        
        Returns:
            Path to the video file. Falls back to black placeholder if no clip is
            available. Returns None only if both the clip download and the
            placeholder are unavailable.
        """
        try:
            camera_name_sanitized = camera_name.lower().replace(' ', '_')
            file_name = PATH_VIDEOS / f"{camera_name_sanitized}_latest.mp4"
        except Exception as e:
            log.error(f"{camera_name}: error creating file path: {e}")
            return None
    
        try:
            if file_name.exists() and not force:
                # Validate before reusing. The download path below checks what
                # it writes, but this path hands back whatever is on disk, and
                # PATH_VIDEOS can outlive the container -- a file truncated by a
                # kill mid-download would otherwise be reused indefinitely. This
                # clip also seeds the stream, and the publisher derives its RTSP
                # SDP from the first thing it plays, so a bad file here is not a
                # local failure: it mis-describes the stream for as long as the
                # publisher runs.
                if is_usable_clip(file_name):
                    # A clip left by an older version may still be in Blink's
                    # native container; bring it in line before it seeds the
                    # stream (no-op when it already is).
                    if not normalize_clip_container(file_name):
                        log.warning(
                            f"{camera_name}: could not normalize cached clip {file_name.name}; "
                            f"streaming it as-is may stall the stream"
                        )
                    log.debug(f"{camera_name}: skipping download, {file_name} exists")
                    return file_name
                log.warning(
                    f"{camera_name}: cached clip {file_name.name} is unreadable, "
                    f"discarding it and downloading again"
                )
                try:
                    file_name.unlink()
                except OSError as e:
                    log.warning(f"{camera_name}: could not delete unreadable cached clip: {e}")
        except OSError as e:
            log.warning(f"{camera_name}: error checking if file exists: {e}")

        try:
            media = self.clip_cache.get(camera_name)
        except Exception as e:
            log.error(f"{camera_name}: error searching clip cache: {e}")
            media = None

        if media is None:
            log.warning(f"{camera_name}: no clips found for camera")
            return None

        try:
            log.debug(f'{camera_name}: downloading video: {media}')
            response = await self.blink.do_http_get(media['media'])
            
            if not response:
                log.error(f"{camera_name}: received empty response from Blink API")
                raise ValueError("Empty response from API")

            log.debug(f'{camera_name}: saving video to {file_name}')
            video_data = await response.read()
            
            if not video_data:
                log.error(f"{camera_name}: received empty video data")
                raise ValueError("Empty video data")
                
            # Written beside the target and renamed into place, like
            # _save_clip(): the publisher may hold the existing file open.
            tmp_name = file_name.with_suffix(file_name.suffix + '.part')
            try:
                with open(tmp_name, 'wb') as f:
                    f.write(video_data)

                if not tmp_name.exists() or tmp_name.stat().st_size == 0:
                    raise IOError("Failed to write video file")

                self._normalize_downloaded_clip(camera_name, tmp_name)
                os.replace(tmp_name, file_name)
            except BaseException:
                try:
                    tmp_name.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
            
            log.debug(f"{camera_name}: successfully downloaded real clip ({file_name.stat().st_size} bytes)")
            return file_name
        except IOError as e:
            log.error(f"{camera_name}: file I/O error downloading clip: {e}")
        except Exception as e:
            log.error(f"{camera_name}: failed to download clip: {e}")
        
        # Download failed — try cached file, then black placeholder
        try:
            if file_name.exists():
                log.warning(f"{camera_name}: using cached clip after download failure")
                return file_name
        except OSError:
            pass

        return None
    
    @staticmethod
    def _normalize_downloaded_clip(camera_name: str, clip: Path) -> None:
        """Rewrite a freshly downloaded clip's container for the concat stream.

        Blink serves clips with 1/1000 time bases; the concat demuxer needs
        every file in a stream to share one (see CONCAT_VIDEO_TIMESCALE), so
        this runs on the temp file before it is renamed into place. Failure is
        logged, not raised: a clip in the wrong container plays, just with
        stalls at its edges, which beats losing the motion footage.
        """
        if not normalize_clip_container(clip):
            log.warning(
                f"{camera_name}: could not normalize downloaded clip container; "
                f"streaming it as-is may stall the stream"
            )

    async def _save_clip(self, camera_name: str, url: str, file_name: Path) -> None:
        """Save a video clip from URL to file.
        
        Args:
            camera_name: Name of the camera
            url: URL of the video clip to download
            file_name: Path where the clip should be saved
            
        Raises:
            Exception: If download or save fails
        """
        try:
            camera = self.blink.cameras[camera_name]
            response = await camera.get_video_clip(url)
            
            if not response:
                raise ValueError("Empty response from get_video_clip")

            log.debug(f'{camera_name}: saving video to {file_name}')
            video_data = await response.read()
            
            if not video_data:
                raise ValueError("Empty video data received")
                
            # Write beside the target and rename into place rather than
            # writing over it. The publisher may be reading this very file: a
            # clip stays queued for several plays now, so a second motion event
            # arriving during that window used to truncate the file mid-read.
            # FFmpeg reported "Invalid NAL unit size", then "moov atom not
            # found", then "Impossible to open" and the publisher died.
            #
            # rename() is atomic and does not disturb an already-open
            # descriptor: the running FFmpeg keeps reading the old inode to the
            # end of its pass, and only the next open sees the new clip. Both
            # are complete files, so either is safe to play.
            tmp_name = file_name.with_suffix(file_name.suffix + '.part')
            try:
                with open(tmp_name, 'wb') as f:
                    f.write(video_data)

                if not tmp_name.exists() or tmp_name.stat().st_size == 0:
                    raise IOError("Failed to write video file or file is empty")

                self._normalize_downloaded_clip(camera_name, tmp_name)
                os.replace(tmp_name, file_name)
            except BaseException:
                try:
                    tmp_name.unlink(missing_ok=True)
                except OSError:
                    pass
                raise
                
            log.debug(f'{camera_name}: video saved ({file_name.stat().st_size} bytes)')
        except IOError as e:
            log.error(f"{camera_name}: file I/O error saving clip: {e}")
            raise
        except Exception as e:
            log.error(f"{camera_name}: error in _save_clip: {e}")
            raise
    
    async def check_for_motion(self, camera_name: str) -> Optional[Path]:
        """Check if camera detected motion and download new clip if available.
        
        Args:
            camera_name: Name of the camera to check
            
        Returns:
            Path to the downloaded clip file, or None if no new motion detected
            
        Note:
            Handles both regular video clips and snapshot events. For snapshots,
            searches for the most recent actual clip in the recent_clips list.
        """
        try:
            await self.blink.refresh()
        except Exception as e:
            log.error(f"{camera_name}: failed to refresh Blink data: {e}")
            raise
            
        try:
            camera = self.blink.cameras[camera_name]
        except KeyError:
            log.error(f"{camera_name}: camera not found in Blink cameras")
            return None
        except Exception as e:
            log.error(f"{camera_name}: error accessing camera: {e}")
            return None

        try:
            motion_detected = camera.attributes.get('motion_detected', False)
            last_record = camera.attributes.get('last_record', 'N/A')
            cached_last_record = self.camera_last_record[camera_name]
            
            log.debug(
                f"{camera_name}: motion_detected={motion_detected}, "
                f"last_record={last_record}, cached={cached_last_record}"
            )
        except Exception as e:
            log.error(f"{camera_name}: error reading camera attributes: {e}")
            return None

        if not motion_detected or cached_last_record == last_record:
            return None

        log.info(f"{camera_name}: motion detected (last_record: {last_record})")

        try:
            camera_name_sanitized = camera_name.lower().replace(' ', '_')
            file_name = PATH_VIDEOS / f"{camera_name_sanitized}_latest.mp4"
        except Exception as e:
            log.error(f"{camera_name}: error creating file path: {e}")
            return None

        # Handle snapshot events by finding recent clip
        try:
            if '/snapshot/' in camera.attributes.get('video', ''):
                recent_clips = camera.attributes.get('recent_clips', [])
                if url := find_most_recent_clip_url(recent_clips, camera.attributes['last_record']):
                    log.debug(f"{camera_name}: found recent clip in snapshot, saving to {file_name}")
                    try:
                        await self._save_clip(camera_name, url, file_name)
                        self.camera_last_record[camera_name] = last_record
                        log.debug(f"{camera_name}: clip saved to {file_name}")
                        return file_name
                    except Exception as e:
                        log.error(f"{camera_name}: failed to save clip from snapshot: {e}")
                        self.camera_last_record[camera_name] = last_record
                        return None

                log.debug(f"{camera_name}: no recent clip in snapshot, skipping")
                self.camera_last_record[camera_name] = last_record
                return None
        except Exception as e:
            log.error(f"{camera_name}: error processing snapshot: {e}")
            return None
        
        # Download regular video clip. Downloaded beside the target and renamed
        # into place for the same reason as _save_clip(): with clip_repeats the
        # publisher can still be reading this very file when the next motion
        # event arrives, and writing over it truncates it mid-read.
        try:
            log.debug(f"{camera_name}: downloading clip to {file_name}")
            tmp_name = file_name.with_suffix(file_name.suffix + '.part')
            try:
                await camera.video_to_file(tmp_name)

                if not tmp_name.exists() or tmp_name.stat().st_size == 0:
                    raise IOError("video file not created or is empty")

                self._normalize_downloaded_clip(camera_name, tmp_name)
                os.replace(tmp_name, file_name)
            except BaseException:
                try:
                    tmp_name.unlink(missing_ok=True)
                except OSError:
                    pass
                raise

            self.camera_last_record[camera_name] = last_record
            log.debug(f"{camera_name}: clip saved to {file_name} ({file_name.stat().st_size} bytes)")
            return file_name
        except IOError as e:
            log.error(f"{camera_name}: file I/O error saving clip: {e}")
            return None
        except Exception as e:
            log.error(f"{camera_name}: error downloading clip: {e}")
            return None
        
    def get_cameras(self) -> iter:
        """Get iterator of all available camera names.
        
        Returns:
            Iterator of camera name strings
        """
        return self.blink.cameras.keys()

    def recently_known_cameras(self, within_days: int = 14) -> set:
        """Return cached camera names with a clip newer than within_days.

        self.blink.cameras only reflects the live Blink snapshot -- if an
        entire sync module drops off Blink's cloud, its cameras vanish from
        that snapshot immediately, not just when they go offline. Without
        this, such a camera would never get a stream server at all (since
        it's absent from get_cameras()), so it could never reach
        is_camera_offline()'s KeyError->True check and would stay invisible
        indefinitely instead of showing OFFLINE. Falling back to the
        persistent clip cache for recently-active cameras closes that gap.
        A camera with no clip in that long is assumed retired rather than
        temporarily down, so it isn't resurrected forever.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=within_days)
        result = set()
        for name, clip in self.clip_cache.items():
            created_at = clip.get('created_at')
            if not created_at:
                continue
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except ValueError:
                continue
            if dt >= cutoff:
                result.add(name)
        return result
    
    async def start(self) -> None:
        """Initialize the camera manager.
        
        Performs authentication, refreshes metadata, and generates the black
        video placeholder for cameras without clips.
        
        Raises:
            LoginError: If authentication fails
            TokenRefreshFailed: If token refresh fails
        """
        try:
            await self._login()
        except Exception as e:
            log.error(f"Login failed: {e}")
            raise
            
        try:
            await self.refresh_metadata()
        except Exception as e:
            log.warning(f"Failed to refresh metadata during startup: {e}")
            # Continue with whatever's in the cache (possibly empty) -- individual
            # cameras will still progress through STARTING and pick up a clip once
            # a later refresh or a real motion event succeeds.
        
        # Generate placeholder videos for Starting / Offline / Error states
        try:
            PATH_VIDEOS.mkdir(parents=True, exist_ok=True)
            self._generate_placeholders()
        except Exception as e:
            log.error(f"Error generating placeholder videos: {e}")
    
    async def close(self) -> None:
        """Properly close all connections and clean up resources.
        
        Closes the aiohttp session and waits briefly for SSL cleanup.
        """
        try:
            if hasattr(self, 'session') and self.session is not None and not self.session.closed:
                await self.session.close()
                # Give the event loop time to clean up SSL transports
                await asyncio.sleep(0.25)
        except Exception as e:
            log.warning(f"Error closing session: {e}")
