import json
import os
import sys
import shutil
from pathlib import Path
from typing import Any


def _read_stdin_json() -> dict[str, Any] | None:
    """
    If stdin has content, try parse as JSON object and return it.
    Otherwise return None.
    """
    try:
        data = sys.stdin.buffer.read()
    except Exception:
        return None

    if not data:
        return None

    s = data.decode("utf-8", errors="replace").strip()
    if not s:
        return None

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        # Not JSON => treat as no payload
        return None


def _collect_targets(out_dir: Path, keep_gitkeep: bool = True) -> list[Path]:
    """
    Collect all files/dirs under out_dir, excluding .gitkeep if keep_gitkeep=True.
    Return absolute paths.
    """
    if not out_dir.exists():
        return []

    targets: list[Path] = []
    for p in out_dir.iterdir():
        if keep_gitkeep and p.name == ".gitkeep":
            continue
        targets.append(p.resolve())
    return sorted(targets, key=lambda x: x.as_posix())


def _delete_path(p: Path) -> tuple[bool, str | None]:
    """
    Delete file/dir path. Return (ok, error_message).
    """
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink(missing_ok=True)
        return True, None
    except Exception as e:
        return False, str(e)


def main() -> None:
    payload = _read_stdin_json()

    # Defaults
    out_dir_str = "out"
    dry_run = False
    keep_gitkeep = True
    force = None  # None => auto (non-interactive stdin => True; interactive => ask)

    if payload is not None:
        out_dir_str = str(payload.get("out_dir", out_dir_str))
        dry_run = bool(payload.get("dry_run", dry_run))
        keep_gitkeep = bool(payload.get("keep_gitkeep", keep_gitkeep))
        # If user explicitly sets force, respect it
        if "force" in payload:
            force = bool(payload["force"])
    else:
        # Optional: allow CLI usage without stdin JSON
        # e.g. python tools/clean_out.py out --dry-run
        if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
            out_dir_str = sys.argv[1]
        if "--dry-run" in sys.argv:
            dry_run = True
        if "--no-keep-gitkeep" in sys.argv:
            keep_gitkeep = False
        if "--force" in sys.argv:
            force = True

    out_dir = Path(out_dir_str)
    # Interpret relative paths from current working directory
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    targets = _collect_targets(out_dir, keep_gitkeep=keep_gitkeep)

    result: dict[str, Any] = {
        "ok": True,
        "tool": "clean_out",
        "out_dir": str(out_dir),
        "dry_run": dry_run,
        "keep_gitkeep": keep_gitkeep,
        "to_delete": [str(p) for p in targets],
        "deleted": [],
        "failed": [],
    }

    # If nothing to do, return success
    if not targets:
        print(json.dumps(result, ensure_ascii=False))
        return

    if dry_run:
        # No deletions
        print(json.dumps(result, ensure_ascii=False))
        return

    # Decide force behavior
    if force is None:
        # If payload is provided (stdin JSON), default to non-interactive force=True.
        # Otherwise (manual CLI), ask for confirmation.
        force = (payload is not None)

    if not force:
        # Interactive confirmation
        try:
            sys.stdout.write(f"[clean_out] About to delete {len(targets)} item(s) under: {out_dir}\n")
            sys.stdout.write("Proceed? (y/N): ")
            sys.stdout.flush()
            ans = sys.stdin.readline().strip().lower()
        except Exception:
            ans = ""
        if ans not in {"y", "yes"}:
            result["ok"] = False
            result["error"] = "User declined deletion."
            print(json.dumps(result, ensure_ascii=False))
            return

    # Perform deletions
    for p in targets:
        ok, err = _delete_path(p)
        if ok:
            result["deleted"].append(str(p))
        else:
            result["failed"].append({"path": str(p), "error": err})

    # Mark ok false if failures exist
    if result["failed"]:
        result["ok"] = False

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
