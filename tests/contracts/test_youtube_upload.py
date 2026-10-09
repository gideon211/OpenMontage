"""Contract tests for the YouTube publisher (youtube_upload).

Network is always stubbed — these tests never call Google and never require
OAuth credentials. They verify the BaseTool contract, input validation,
metadata trimming, and that both success and failure paths emit a
schema-valid publish_log.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tools.publishers.youtube_upload import YouTubeUpload
from tools.publishers.youtube_upload import _MAX_DESCRIPTION_CHARS, _MAX_TAGS_CHARS, _MAX_TITLE_CHARS
from schemas.artifacts import validate_artifact


def _clip(tmp_path: Path) -> Path:
    """A stand-in render: validation checks existence/extension, not codecs."""
    clip = tmp_path / "final.mp4"
    clip.write_bytes(b"\x00\x00\x00\x18ftypmp42 fake")
    return clip


# ---- Discovery / contract ----

def test_registry_discovers_youtube_upload(isolated_tool_registry):
    isolated_tool_registry.discover("tools")
    tool = isolated_tool_registry.get("youtube_upload")
    assert tool is not None
    assert tool.capability == "publish"
    assert tool.provider == "youtube"
    assert tool.tier.value == "publish"
    # Deps may be absent on a bare machine; discovery must never crash either way.
    assert tool.get_status().value in ("available", "unavailable")


def test_contract_metadata():
    tool = YouTubeUpload()
    assert tool.name == "youtube_upload"
    assert tool.runtime.value == "api"
    assert tool.supports["uploads"] is True
    assert tool.supports["free"] is True
    assert all(d.startswith("python:") for d in tool.dependencies)
    assert any("youtube_client.json" in s for s in tool.side_effects) is False
    assert any("consent" in s for s in tool.side_effects)  # browser side effect is declared
    assert tool.resource_profile.network_required is True


def test_input_schema_declares_required_and_visibility_enum():
    props = YouTubeUpload().input_schema["properties"]
    assert set(YouTubeUpload().input_schema["required"]) == {"video_path", "title"}
    assert props["visibility"]["enum"] == ["public", "private", "unlisted"]
    assert props["visibility"]["default"] == "unlisted"


# ---- Validation ----

def test_validation_rejects_missing_file_bad_visibility_and_title(tmp_path):
    tool = YouTubeUpload()
    good = str(_clip(tmp_path))

    assert "video_path not found" in tool.execute({"video_path": str(tmp_path / "nope.mp4"), "title": "t"}).error
    assert "title is required" in tool.execute({"video_path": good, "title": "  "}).error
    assert "visibility must be one of" in tool.execute(
        {"video_path": good, "title": "t", "visibility": "secret"}
    ).error
    assert "thumbnail_path provided but not found" in tool.execute(
        {"video_path": good, "title": "t", "thumbnail_path": str(tmp_path / "missing.jpg")}
    ).error


def test_dry_run_reports_oauth_readiness_without_network(tmp_path):
    tool = YouTubeUpload()
    report = tool.dry_run({"video_path": str(_clip(tmp_path)), "title": "Episode 1"})
    assert report["input_error"] is None
    assert report["video_exists"] is True
    assert report["libraries_installed"] is True
    assert report["oauth"]["consent_required"] is True  # no client JSON in tests
    assert report["would_execute"] is False  # blocked until OAuth client is placed
    assert report["quota_units_per_upload"] == 1600
    assert "uploads/day" in report["quota_note"]


# ---- Metadata assembly ----

def test_metadata_is_trimmed_to_youtube_limits(tmp_path):
    tool = YouTubeUpload()
    desc = tool._compose_description(
        {"description": "x" * 6000, "hashtags": ["history", "shipping"], "chapters": []}
    )
    assert len(desc) <= _MAX_DESCRIPTION_CHARS

    tags = tool._compose_tags(["tag" * 400, "overflow"])
    assert sum(len(t) for t in tags) <= _MAX_TAGS_CHARS

    inputs = {"title": "y" * 300}
    assert len(str(inputs["title"]).strip()[:_MAX_TITLE_CHARS]) == _MAX_TITLE_CHARS


def test_chapters_and_hashtags_rendered_into_description(tmp_path):
    tool = YouTubeUpload()
    desc = tool._compose_description(
        {
            "description": "A story about containers.",
            "chapters": [
                {"start_seconds": 0, "title": "Before the Box"},
                {"time": "01:00", "label": "The World Transformed"},
            ],
            "hashtags": ["history", "#documentary"],
        }
    )
    assert "0:00 - Before the Box" in desc
    assert "01:00 - The World Transformed" in desc
    assert "#history" in desc and "#documentary" in desc


# ---- End-to-end (network stubbed) ----

def test_successful_upload_returns_valid_publish_log(tmp_path, monkeypatch):
    tool = YouTubeUpload()
    monkeypatch.setattr(tool, "_service", lambda: object())
    monkeypatch.setattr(tool, "_insert_video", lambda *a, **k: "vid123ABC")

    result = tool.execute(
        {
            "video_path": str(_clip(tmp_path)),
            "title": "The Box: How One Object Changed Shipping",
            "description": "Containers changed everything.",
            "tags": ["history", "documentary"],
            "hashtags": ["history"],
            "chapters": [{"start_seconds": 0, "title": "Start"}],
            "visibility": "unlisted",
            "timestamp": "2026-10-09T10:00:00+00:00",
        }
    )

    assert result.success is True, result.error
    assert result.data["video_id"] == "vid123ABC"
    assert result.data["url"] == "https://www.youtube.com/watch?v=vid123ABC"
    entry = result.data["publish_log"]["entries"][0]
    assert entry["platform"] == "youtube"
    assert entry["status"] == "published"
    assert entry["visibility"] == "unlisted"
    validate_artifact("publish_log", result.data["publish_log"])  # raises if invalid


def test_failed_upload_still_returns_valid_publish_log(tmp_path, monkeypatch):
    tool = YouTubeUpload()
    monkeypatch.setattr(tool, "_service", lambda: object())

    def boom(*a, **k):
        raise RuntimeError("HttpError 403: quotaExceeded")

    monkeypatch.setattr(tool, "_insert_video", boom)

    result = tool.execute(
        {
            "video_path": str(_clip(tmp_path)),
            "title": "Episode",
            "visibility": "private",
            "timestamp": "2026-10-09T10:05:00+00:00",
        }
    )

    assert result.success is False
    assert "YouTube upload failed" in result.error
    entry = result.data["publish_log"]["entries"][0]
    assert entry["status"] == "failed"
    assert "quotaExceeded" in entry["error"]
    validate_artifact("publish_log", result.data["publish_log"])


def test_oauth_setup_failure_returns_valid_publish_log(tmp_path, monkeypatch):
    tool = YouTubeUpload()
    monkeypatch.setattr(
        tool, "_service", lambda: (_ for _ in ()).throw(FileNotFoundError("no client JSON"))
    )
    result = tool.execute({"video_path": str(_clip(tmp_path)), "title": "Episode"})
    assert result.success is False
    assert "OAuth/service setup failed" in result.error
    validate_artifact("publish_log", result.data["publish_log"])


# ---- Secrets hygiene ----

def test_secrets_resolve_under_dot_secrets_and_are_gitignored():
    tool = YouTubeUpload()
    secrets_dir = tool._secrets_dir()
    assert secrets_dir.name == ".secrets"
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".secrets/" in gitignore
    # The token path must live inside the ignored folder — never in the repo tree.
    assert tool._token_path().parent == secrets_dir
