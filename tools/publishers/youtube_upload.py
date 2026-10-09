"""YouTube publisher — uploads a finished render straight to YouTube.

The free networked counterpart to ``export_bundle``: same ``publish``
capability and the same schema-valid ``publish_log`` output, but instead of
writing a hand-off bundle it pushes the video to a YouTube channel and
returns the real ``video_id``/``url``.

Free path: YouTube Data API v3 with **user OAuth** (no paid API key, no
credit card). Setup is one-time:

1. Google Cloud project with the YouTube Data API v3 enabled.
2. OAuth client credentials (Desktop app) saved as ``.secrets/youtube_client.json``.
3. First upload opens a local browser for consent; the token is cached at
   ``.secrets/youtube_token.json`` and auto-refreshed afterwards.

Secrets never leave ``.secrets/`` (gitignored) and are never logged.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tools.base_tool import (
    BaseTool,
    Determinism,
    ExecutionMode,
    ResourceProfile,
    RetryPolicy,
    ToolResult,
    ToolRuntime,
    ToolStability,
    ToolTier,
)

# The registry imports every module under tools/ at discovery time, so a
# missing optional dependency must never break discovery — it is caught here
# and surfaced through check_dependencies() as UNAVAILABLE instead.
try:  # pragma: no cover - exercised by whichever branch the machine has
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload

    _GOOGLE_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # ImportError, or a broken transitive dependency
    Request = Credentials = InstalledAppFlow = build = MediaFileUpload = None  # type: ignore[assignment]
    _GOOGLE_IMPORT_ERROR = str(exc)


SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# YouTube API hard limits — the API rejects anything longer.
_MAX_TITLE_CHARS = 100
_MAX_DESCRIPTION_CHARS = 5000
_MAX_TAGS_CHARS = 500  # combined length of all snippet.tags
_MAX_THUMBNAIL_BYTES = 2 * 1024 * 1024
_DEFAULT_CATEGORY_ID = "27"  # 27 = Education on YouTube
_INSERT_QUOTA_UNITS = 1600  # videos.insert quota cost (free default quota: 10000/day)


class YouTubeUpload(BaseTool):
    name = "youtube_upload"
    version = "0.1.0"
    tier = ToolTier.PUBLISH
    capability = "publish"
    provider = "youtube"
    stability = ToolStability.EXPERIMENTAL
    execution_mode = ExecutionMode.SYNC
    determinism = Determinism.DETERMINISTIC
    runtime = ToolRuntime.API

    dependencies = [
        "python:googleapiclient",
        "python:google_auth_oauthlib",
        "python:google.auth",
    ]
    install_instructions = (
        "pip install google-api-python-client google-auth-oauthlib google-auth-httplib2, "
        "then place a Google Cloud OAuth client JSON (Desktop app type, YouTube Data API v3 "
        "enabled) at .secrets/youtube_client.json. The first upload opens a browser for "
        "consent and caches the token at .secrets/youtube_token.json."
    )

    agent_skills = []

    capabilities = ["upload_video", "set_thumbnail", "write_publish_log"]
    supports = {
        "free": True,  # OAuth user quota, no paid API key
        "uploads": True,
        "oauth_required": True,
        "network_required": True,
        "quota_units_per_upload": _INSERT_QUOTA_UNITS,
    }
    best_for = [
        "uploading a finished render directly to YouTube (free, OAuth)",
        "producing a schema-valid publish_log that carries the real video_id and url",
    ]
    not_good_for = [
        "publishing without a one-time Google OAuth setup (client JSON required)",
        "bulk scheduling or analytics — one upload per call, bounded by daily API quota",
        "platforms other than YouTube (use export_bundle for a hand-off bundle)",
    ]

    resource_profile = ResourceProfile(
        cpu_cores=1, ram_mb=256, vram_mb=0, disk_mb=0, network_required=True
    )
    retry_policy = RetryPolicy(
        max_retries=3,
        backoff_seconds=2.0,
        retryable_errors=["HttpError 500", "HttpError 503", "Connection reset"],
    )
    side_effects = [
        "uploads a video to the authenticated YouTube channel",
        "may open a local browser window for first-time OAuth consent",
        "writes/refreshes .secrets/youtube_token.json",
    ]
    user_visible_verification = [
        "Open the returned url and confirm the video plays with the intended title, description, and visibility",
    ]

    input_schema = {
        "type": "object",
        "required": ["video_path", "title"],
        "properties": {
            "video_path": {
                "type": "string",
                "description": "Path to the finished render (render_report.outputs[].path).",
            },
            "title": {"type": "string", "description": "Video title (YouTube max 100 chars)."},
            "description": {
                "type": "string",
                "description": "Video description (YouTube max 5000 chars; trimmed to fit).",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "YouTube tags (combined 500-char limit; trimmed to fit).",
            },
            "hashtags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Hashtags appended to the description.",
            },
            "chapters": {
                "type": "array",
                "items": {
                    "type": "object",
                    "description": "Either {start_seconds, title} or {time, label}; rendered as description chapter lines.",
                },
            },
            "visibility": {
                "type": "string",
                "enum": ["public", "private", "unlisted"],
                "default": "unlisted",
                "description": "YouTube privacy status. Defaults to unlisted for safe review.",
            },
            "made_for_kids": {
                "type": "boolean",
                "default": False,
                "description": "selfDeclaredMadeForKids — required by YouTube; false unless the content is made for children.",
            },
            "category_id": {
                "type": "string",
                "default": _DEFAULT_CATEGORY_ID,
                "description": f"YouTube category id (default {_DEFAULT_CATEGORY_ID} = Education).",
            },
            "thumbnail_path": {
                "type": "string",
                "description": "Optional custom thumbnail image (<= 2MB).",
            },
            "timestamp": {
                "type": "string",
                "description": "Override the ISO-8601 timestamp (mainly for deterministic tests).",
            },
        },
    }
    output_schema = {
        "type": "object",
        "properties": {
            "publish_log": {"type": "object"},
            "video_id": {"type": "string"},
            "url": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }

    # ---- Secrets helpers ----

    @staticmethod
    def _repo_root() -> Path:
        return Path(__file__).resolve().parents[2]

    def _secrets_dir(self) -> Path:
        override = os.environ.get("OPENMONTAGE_SECRETS_DIR")
        return Path(override).expanduser() if override else self._repo_root() / ".secrets"

    def _client_secret_path(self) -> Optional[Path]:
        override = os.environ.get("YOUTUBE_CLIENT_SECRET_FILE")
        if override:
            p = Path(override).expanduser()
            return p if p.is_file() else None
        for name in ("youtube_client.json", "client_secret.json"):
            p = self._secrets_dir() / name
            if p.is_file():
                return p
        return None

    def _token_path(self) -> Path:
        return self._secrets_dir() / "youtube_token.json"

    # ---- OAuth ----

    def _credentials(self):
        """Load cached OAuth tokens, refreshing them if needed, else consent."""
        if Credentials is None:  # deps missing — check_dependencies catches first
            raise RuntimeError(f"Google API libraries not importable: {_GOOGLE_IMPORT_ERROR}")
        token_path = self._token_path()
        creds = None
        if token_path.is_file():
            creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_path = self._client_secret_path()
            if client_path is None:
                raise FileNotFoundError(
                    f"No OAuth client found at {self._secrets_dir() / 'youtube_client.json'} "
                    f"(or YOUTUBE_CLIENT_SECRET_FILE). {self.install_instructions}"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(client_path), SCOPES)
            creds = flow.run_local_server(port=0, open_browser=True)
        # Persist for the next run, readable only by this user.
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(creds.to_json(), encoding="utf-8")
        try:
            token_path.chmod(0o600)
        except OSError:  # pragma: no cover - non-POSIX filesystems
            pass
        return creds

    def _service(self):
        return build("youtube", "v3", credentials=self._credentials(), cache_discovery=False)

    # ---- Input validation ----

    def _validate_inputs(self, inputs: dict[str, Any]) -> Optional[str]:
        video_path = Path(str(inputs.get("video_path", ""))).expanduser()
        if not inputs.get("video_path"):
            return "video_path is required"
        if not video_path.is_file():
            return f"video_path not found: {video_path}"
        if video_path.suffix.lower() not in (".mp4", ".mov", ".mkv", ".webm", ".avi"):
            return f"video_path must be a video file, got: {video_path.suffix or 'no extension'}"
        title = str(inputs.get("title", "")).strip()
        if not title:
            return "title is required"
        if not title[: _MAX_TITLE_CHARS].strip():
            return f"title must have a non-empty first {_MAX_TITLE_CHARS} characters"
        visibility = inputs.get("visibility", "unlisted")
        if visibility not in ("public", "private", "unlisted"):
            return f"visibility must be one of public|private|unlisted, got: {visibility}"
        thumb = inputs.get("thumbnail_path")
        if thumb and not Path(str(thumb)).expanduser().is_file():
            return f"thumbnail_path provided but not found: {thumb}"
        return None

    # ---- Metadata assembly ----

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}:{m:02d}:{s:02d}"
        return f"{m}:{s:02d}"

    def _chapter_lines(self, chapters: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for ch in chapters:
            label = ch.get("title") or ch.get("label") or ""
            if "start_seconds" in ch or "time_seconds" in ch:
                ts = self._format_time(ch.get("start_seconds", ch.get("time_seconds", 0)))
            elif "time" in ch:
                ts = str(ch["time"])
            else:
                ts = "0:00"
            lines.append(f"{ts} - {label}".rstrip(" -"))
        return lines

    def _compose_description(self, inputs: dict[str, Any]) -> str:
        """Description + chapter lines + hashtags, trimmed to YouTube's limit."""
        parts: list[str] = []
        if inputs.get("description"):
            parts.append(str(inputs["description"]).strip())
        chapters = inputs.get("chapters") or []
        if chapters:
            parts.append("\n".join(self._chapter_lines(chapters)))
        hashtags = inputs.get("hashtags") or []
        if hashtags:
            tags = " ".join(
                str(h) if str(h).startswith("#") else f"#{str(h).lstrip('#')}" for h in hashtags
            )
            parts.append(tags)
        description = "\n\n".join(p for p in parts if p)
        if len(description) > _MAX_DESCRIPTION_CHARS:
            description = description[: _MAX_DESCRIPTION_CHARS - 1].rstrip() + "…"
        return description

    @staticmethod
    def _compose_tags(tags: list[str]) -> list[str]:
        """Keep tags within YouTube's combined 500-char budget."""
        kept: list[str] = []
        used = 0
        for raw in tags:
            tag = str(raw).strip()
            if not tag:
                continue
            cost = len(tag) + (1 if kept else 0)
            if used + cost > _MAX_TAGS_CHARS:
                break
            kept.append(tag)
            used += cost
        return kept

    # ---- Publish log ----

    def _entry(
        self,
        *,
        status: str,
        inputs: dict[str, Any],
        timestamp: str,
        description: str,
        video_id: Optional[str] = None,
        error: Optional[str] = None,
    ) -> dict[str, Any]:
        visibility = inputs.get("visibility", "unlisted")
        entry: dict[str, Any] = {
            "platform": "youtube",
            "status": status,
            "visibility": visibility,
            "export_path": str(Path(str(inputs["video_path"])).expanduser()),
            "timestamp": timestamp,
            "metadata_used": {
                "title": str(inputs["title"]).strip(),
                "description": description,
                "hashtags": list(inputs.get("hashtags") or []),
                "chapters": list(inputs.get("chapters") or []),
            },
        }
        if video_id:
            entry["video_id"] = video_id
            entry["url"] = f"https://www.youtube.com/watch?v={video_id}"
        if error:
            entry["error"] = error
        return entry

    def _validated_log(self, entry: dict[str, Any]) -> Optional[str]:
        """Validate against the canonical schema. Returns an error string or None."""
        publish_log = {"version": "1.0", "entries": [entry]}
        try:
            from schemas.artifacts import validate_artifact

            validate_artifact("publish_log", publish_log)
        except Exception as exc:
            return f"publish_log failed schema validation: {exc}"
        return None

    # ---- Upload ----

    def _insert_video(self, service: Any, body: dict, video_path: Path) -> str:
        media = MediaFileUpload(
            str(video_path), chunksize=-1, resumable=True, mimetype="video/mp4"
        )
        request = service.videos().insert(part="snippet,status", body=body, media_body=media)
        response: dict[str, Any] = {}
        while response is None or not response:
            _, response = request.next_chunk(num_retries=3)
        video_id = response.get("id")
        if not video_id:
            raise RuntimeError(f"YouTube returned no video id: {json.dumps(response)[:300]}")
        return str(video_id)

    # ---- Preflight ----

    def dry_run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Preflight: validates inputs and reports OAuth readiness — no network calls."""
        report = super().dry_run(inputs)
        input_error = self._validate_inputs(inputs) if inputs else "no inputs provided"
        video_path = Path(str(inputs.get("video_path", ""))).expanduser() if inputs else None
        client = self._client_secret_path()
        token_exists = self._token_path().is_file()
        libs_ok = _GOOGLE_IMPORT_ERROR is None
        report.update(
            {
                "input_error": input_error,
                "video_exists": bool(video_path and video_path.is_file()),
                "video_size_mb": round(video_path.stat().st_size / 1e6, 1)
                if video_path and video_path.is_file()
                else None,
                "libraries_installed": libs_ok,
                "library_error": _GOOGLE_IMPORT_ERROR,
                "oauth": {
                    "client_secret_found": client is not None,
                    "token_cached": token_exists,
                    "consent_required": client is None or not token_exists,
                },
                "quota_units_per_upload": _INSERT_QUOTA_UNITS,
                "quota_note": (
                    "videos.insert costs "
                    f"{_INSERT_QUOTA_UNITS} units against YouTube's free daily quota "
                    "(default 10000 units ≈ 6 uploads/day)"
                ),
                "would_execute": (
                    input_error is None and libs_ok and client is not None
                ),
                "next_step": (
                    "ready to upload"
                    if input_error is None and libs_ok and client is not None and token_exists
                    else (
                        "OAuth client JSON missing — see install_instructions"
                        if client is None
                        else (
                            "libraries not installed — pip install (see install_instructions)"
                            if not libs_ok
                            else input_error
                            or "first upload will open a browser for Google consent"
                        )
                    )
                ),
            }
        )
        return report

    # ---- Execution ----

    def execute(self, inputs: dict[str, Any]) -> ToolResult:
        error = self._validate_inputs(inputs)
        if error:
            return ToolResult(success=False, error=error)

        libs_ok = _GOOGLE_IMPORT_ERROR is None
        if not libs_ok:
            return ToolResult(
                success=False,
                error=f"Google API libraries not installed ({_GOOGLE_IMPORT_ERROR}). {self.install_instructions}",
            )

        video_path = Path(str(inputs["video_path"])).expanduser()
        title = str(inputs["title"]).strip()[:_MAX_TITLE_CHARS]
        description = self._compose_description(inputs)
        tags = self._compose_tags(list(inputs.get("tags") or []))
        visibility = str(inputs.get("visibility", "unlisted"))
        timestamp = str(
            inputs.get("timestamp") or datetime.now(timezone.utc).isoformat()
        )
        warnings: list[str] = []

        # --- Service (loads/refreshes OAuth; may open the browser once) ---
        try:
            service = self._service()
        except Exception as exc:
            entry = self._entry(
                status="failed", inputs=inputs, timestamp=timestamp,
                description=description, error=str(exc),
            )
            return ToolResult(
                success=False,
                error=f"YouTube OAuth/service setup failed: {exc}",
                data={"publish_log": {"version": "1.0", "entries": [entry]}},
            )

        body = {
            "snippet": {
                "title": title,
                "description": description,
                "tags": tags,
                "categoryId": str(inputs.get("category_id") or _DEFAULT_CATEGORY_ID),
            },
            "status": {
                "privacyStatus": visibility,
                "selfDeclaredMadeForKids": bool(inputs.get("made_for_kids", False)),
            },
        }

        # --- Upload ---
        try:
            video_id = self._insert_video(service, body, video_path)
        except Exception as exc:
            entry = self._entry(
                status="failed", inputs=inputs, timestamp=timestamp,
                description=description, error=str(exc),
            )
            if (schema_err := self._validated_log(entry)) is not None:
                return ToolResult(success=False, error=f"{schema_err} | upload failed: {exc}")
            return ToolResult(
                success=False,
                error=f"YouTube upload failed: {exc}",
                data={"publish_log": {"version": "1.0", "entries": [entry]}},
            )

        # --- Custom thumbnail (optional, non-fatal) ---
        thumb = inputs.get("thumbnail_path")
        if thumb:
            thumb_path = Path(str(thumb)).expanduser()
            try:
                if thumb_path.stat().st_size > _MAX_THUMBNAIL_BYTES:
                    warnings.append(
                        f"thumbnail skipped: {thumb_path.name} exceeds YouTube's 2MB limit"
                    )
                else:
                    service.thumbnails().set(
                        videoId=video_id,
                        media_body=MediaFileUpload(str(thumb_path), mimetype="image/jpeg"),
                    ).execute(num_retries=2)
            except Exception as exc:
                warnings.append(f"thumbnail upload failed (video is live): {exc}")

        entry = self._entry(
            status="published", inputs=inputs, timestamp=timestamp,
            description=description, video_id=video_id,
        )
        if (schema_err := self._validated_log(entry)) is not None:
            return ToolResult(success=False, error=schema_err)

        return ToolResult(
            success=True,
            data={
                "publish_log": {"version": "1.0", "entries": [entry]},
                "video_id": video_id,
                "url": entry["url"],
                "visibility": visibility,
                "warnings": warnings,
            },
            artifacts=[str(video_path)],
        )
