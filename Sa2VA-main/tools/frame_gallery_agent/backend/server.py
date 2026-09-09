#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import shutil
import tempfile
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse
import zipfile


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FLAT_MASK_SENTINEL = "__video__"


@dataclass(frozen=True)
class DatasetRoot:
    dataset: str
    split: str
    frames_root: Path
    masks_root: Path | None = None


def build_known_roots(data_root: Path) -> list[DatasetRoot]:
    roots = [
        DatasetRoot(
            "davis17",
            "train",
            data_root / "davis17" / "train" / "JPEGImages",
            data_root / "davis17" / "gt_masks_train_expr",
        ),
        DatasetRoot("davis17", "valid", data_root / "davis17" / "valid" / "JPEGImages", None),
        DatasetRoot(
            "mevis",
            "train",
            data_root / "mevis" / "train" / "JPEGImages",
            data_root.parent.parent / "gt_masks",
        ),
        DatasetRoot(
            "mevis",
            "valid",
            data_root / "mevis" / "valid" / "JPEGImages",
            data_root.parent.parent / "gt_masks",
        ),
        DatasetRoot(
            "mevis",
            "valid_u",
            data_root / "mevis" / "valid_u" / "JPEGImages",
            data_root.parent.parent / "gt_masks",
        ),
        DatasetRoot(
            "refytros21",
            "train",
            data_root / "ref-youtube-ros-21" / "21" / "train" / "JPEGImages",
            data_root / "ref-youtube-ros-21" / "21" / "train" / "Annotations",
        ),
        DatasetRoot(
            "refytros21",
            "valid",
            data_root / "ref-youtube-ros-21" / "21" / "valid" / "JPEGImages",
            data_root / "ref-youtube-ros-21" / "21" / "valid" / "Annotations",
        ),
        DatasetRoot(
            "refytros",
            "valid",
            data_root / "ref-youtube-ros" / "valid" / "JPEGImages",
            data_root / "ref-youtube-ros" / "valid" / "Annotations",
        ),
    ]
    return [root for root in roots if root.frames_root.is_dir()]


DATASET_ALIASES = {
    "all": {"davis17", "mevis", "refytros", "refytros21"},
    "ref-youtube-ros": {"refytros", "refytros21"},
    "refytros": {"refytros", "refytros21"},
    "refytros21": {"refytros21"},
    "davis": {"davis17"},
    "davis17": {"davis17"},
    "mevis": {"mevis"},
}


def dataset_matches(value: str | None, dataset_name: str) -> bool:
    if not value or value == "all":
        return True
    allowed = DATASET_ALIASES.get(value.lower(), {value.lower()})
    return dataset_name.lower() in allowed


def split_matches(value: str | None, split_name: str) -> bool:
    if not value or value == "all":
        return True
    return value == split_name


def list_frame_files(video_dir: Path) -> list[Path]:
    frames = [path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    return sorted(frames, key=lambda path: path.name)


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def json_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


class FrameGalleryHandler(BaseHTTPRequestHandler):
    server_version = "FrameGalleryAgent/0.1"

    @property
    def known_roots(self) -> list[DatasetRoot]:
        return self.server.known_roots  # type: ignore[attr-defined]

    @property
    def frontend_dist(self) -> Path | None:
        return self.server.frontend_dist  # type: ignore[attr-defined]

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        super().end_headers()

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            self._send_json({"ok": True})
            return
        if parsed.path == "/api/roots":
            self._handle_roots()
            return
        if parsed.path == "/api/search":
            self._handle_search(parse_qs(parsed.query))
            return
        if parsed.path == "/api/frames":
            self._handle_frames(parse_qs(parsed.query))
            return
        if parsed.path == "/api/masks":
            self._handle_masks(parse_qs(parsed.query))
            return
        if parsed.path == "/api/mask-frames":
            self._handle_mask_frames(parse_qs(parsed.query))
            return
        if parsed.path == "/frame":
            self._handle_frame(parse_qs(parsed.query))
            return
        if parsed.path == "/mask":
            self._handle_mask(parse_qs(parsed.query))
            return
        if parsed.path == "/download":
            self._handle_download(parse_qs(parsed.query))
            return
        if parsed.path == "/" or parsed.path.startswith("/assets/"):
            if self._try_serve_frontend(parsed.path):
                return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def _handle_roots(self) -> None:
        roots = [
            {
                "dataset": root.dataset,
                "split": root.split,
                "frames_root": str(root.frames_root),
                "masks_root": str(root.masks_root) if root.masks_root else None,
            }
            for root in self.known_roots
        ]
        self._send_json({"roots": roots})

    def _handle_search(self, query: dict[str, list[str]]) -> None:
        raw_video_id = (query.get("video_id") or [""])[0].strip()
        if not raw_video_id:
            self._send_json({"matches": []})
            return

        dataset_filter = (query.get("dataset") or ["all"])[0].strip().lower()
        split_filter = (query.get("split") or ["all"])[0].strip()
        mode = (query.get("mode") or ["contains"])[0].strip().lower()
        max_results = int((query.get("max_results") or ["50"])[0])

        matches: list[dict[str, object]] = []
        video_id_lower = raw_video_id.lower()

        for root in self.known_roots:
            if not dataset_matches(dataset_filter, root.dataset):
                continue
            if not split_matches(split_filter, root.split):
                continue

            try:
                video_dirs = [
                    path
                    for path in root.frames_root.iterdir()
                    if path.is_dir()
                    and (
                        (mode == "exact" and path.name == raw_video_id)
                        or (mode != "exact" and video_id_lower in path.name.lower())
                    )
                ]
            except FileNotFoundError:
                continue

            for video_dir in sorted(video_dirs, key=lambda path: path.name):
                frame_files = list_frame_files(video_dir)
                if not frame_files:
                    continue
                first_frame = frame_files[0].name
                matches.append(
                    {
                        "dataset": root.dataset,
                        "split": root.split,
                        "video_id": video_dir.name,
                        "frame_count": len(frame_files),
                        "frames_root": str(root.frames_root),
                        "video_dir": str(video_dir),
                        "preview_url": self._frame_url(root.dataset, root.split, video_dir.name, first_frame),
                        "first_frame": first_frame,
                    }
                )
                if len(matches) >= max_results:
                    self._send_json({"matches": matches})
                    return

        self._send_json({"matches": matches})

    def _handle_frames(self, query: dict[str, list[str]]) -> None:
        dataset = (query.get("dataset") or [""])[0].strip()
        split = (query.get("split") or [""])[0].strip()
        video_id = (query.get("video_id") or [""])[0].strip()

        root = self._resolve_root(dataset, split)
        if root is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unknown dataset/split")
            return

        video_dir = root.frames_root / video_id
        if not video_dir.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
            return

        frame_files = list_frame_files(video_dir)
        frames = [
            {
                "name": frame.name,
                "url": self._frame_url(root.dataset, root.split, video_id, frame.name),
            }
            for frame in frame_files
        ]
        self._send_json(
            {
                "dataset": root.dataset,
                "split": root.split,
                "video_id": video_id,
                "video_dir": str(video_dir),
                "frame_count": len(frames),
                "frames": frames,
            }
        )

    def _handle_masks(self, query: dict[str, list[str]]) -> None:
        dataset = (query.get("dataset") or [""])[0].strip()
        split = (query.get("split") or [""])[0].strip()
        video_id = (query.get("video_id") or [""])[0].strip()

        root = self._resolve_root(dataset, split)
        if root is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unknown dataset/split")
            return
        if root.masks_root is None or not root.masks_root.is_dir():
            self._send_json(
                {
                    "dataset": root.dataset,
                    "split": root.split,
                    "video_id": video_id,
                    "mask_root": None,
                    "exp_ids": [],
                }
            )
            return

        video_mask_dir = root.masks_root / video_id
        if not video_mask_dir.is_dir():
            self._send_json(
                {
                    "dataset": root.dataset,
                    "split": root.split,
                    "video_id": video_id,
                    "mask_root": str(root.masks_root),
                    "exp_ids": [],
                }
            )
            return

        flat_mask_files = list_frame_files(video_mask_dir)
        if flat_mask_files:
            self._send_json(
                {
                    "dataset": root.dataset,
                    "split": root.split,
                    "video_id": video_id,
                    "mask_root": str(video_mask_dir),
                    "layout": "flat",
                    "exp_ids": [
                        {
                            "exp_id": FLAT_MASK_SENTINEL,
                            "frame_count": len(flat_mask_files),
                            "preview_url": self._mask_url(
                                root.dataset,
                                root.split,
                                video_id,
                                FLAT_MASK_SENTINEL,
                                flat_mask_files[0].name,
                            ),
                        }
                    ],
                }
            )
            return

        exp_entries = []
        for exp_dir in sorted([path for path in video_mask_dir.iterdir() if path.is_dir()], key=lambda path: path.name):
            frame_files = list_frame_files(exp_dir)
            if not frame_files:
                continue
            exp_entries.append(
                {
                    "exp_id": exp_dir.name,
                    "frame_count": len(frame_files),
                    "preview_url": self._mask_url(root.dataset, root.split, video_id, exp_dir.name, frame_files[0].name),
                }
            )

        self._send_json(
            {
                "dataset": root.dataset,
                "split": root.split,
                "video_id": video_id,
                "mask_root": str(video_mask_dir),
                "layout": "exp",
                "exp_ids": exp_entries,
            }
        )

    def _handle_mask_frames(self, query: dict[str, list[str]]) -> None:
        dataset = (query.get("dataset") or [""])[0].strip()
        split = (query.get("split") or [""])[0].strip()
        video_id = (query.get("video_id") or [""])[0].strip()
        exp_id = (query.get("exp_id") or [""])[0].strip()

        root = self._resolve_root(dataset, split)
        if root is None or root.masks_root is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Mask root is unavailable")
            return

        if exp_id == FLAT_MASK_SENTINEL:
            exp_dir = root.masks_root / video_id
        else:
            exp_dir = root.masks_root / video_id / exp_id
        if not exp_dir.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND, "Mask sequence not found")
            return

        frame_files = list_frame_files(exp_dir)
        frames = [
            {
                "name": frame.name,
                "url": self._mask_url(root.dataset, root.split, video_id, exp_id, frame.name),
            }
            for frame in frame_files
        ]
        self._send_json(
            {
                "dataset": root.dataset,
                "split": root.split,
                "video_id": video_id,
                "exp_id": exp_id,
                "mask_dir": str(exp_dir),
                "frame_count": len(frames),
                "frames": frames,
            }
        )

    def _handle_frame(self, query: dict[str, list[str]]) -> None:
        dataset = (query.get("dataset") or [""])[0].strip()
        split = (query.get("split") or [""])[0].strip()
        video_id = (query.get("video_id") or [""])[0].strip()
        frame_name = (query.get("name") or [""])[0].strip()

        root = self._resolve_root(dataset, split)
        if root is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unknown dataset/split")
            return

        if not frame_name:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing frame name")
            return

        video_dir = root.frames_root / video_id
        frame_path = (video_dir / frame_name).resolve()
        if not video_dir.is_dir() or not frame_path.is_file() or not is_relative_to(frame_path, video_dir):
            self.send_error(HTTPStatus.NOT_FOUND, "Frame not found")
            return

        mime_type, _ = mimetypes.guess_type(frame_path.name)
        with frame_path.open("rb") as handle:
            payload = handle.read()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _handle_download(self, query: dict[str, list[str]]) -> None:
        dataset = (query.get("dataset") or [""])[0].strip()
        split = (query.get("split") or [""])[0].strip()
        video_id = (query.get("video_id") or [""])[0].strip()

        root = self._resolve_root(dataset, split)
        if root is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Unknown dataset/split")
            return

        video_dir = root.frames_root / video_id
        if not video_dir.is_dir():
            self.send_error(HTTPStatus.NOT_FOUND, "Video not found")
            return

        frame_files = list_frame_files(video_dir)
        if not frame_files:
            self.send_error(HTTPStatus.NOT_FOUND, "No frame files found")
            return

        archive_name = f"{dataset}_{split}_{video_id}.zip"
        with tempfile.TemporaryFile() as tmp:
            with zipfile.ZipFile(tmp, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for frame_path in frame_files:
                    zf.write(frame_path, arcname=f"{video_id}/{frame_path.name}")

            tmp.seek(0, 2)
            total_size = tmp.tell()
            tmp.seek(0)

            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{archive_name}"')
            self.send_header("Content-Length", str(total_size))
            self.end_headers()
            shutil.copyfileobj(tmp, self.wfile)

    def _handle_mask(self, query: dict[str, list[str]]) -> None:
        dataset = (query.get("dataset") or [""])[0].strip()
        split = (query.get("split") or [""])[0].strip()
        video_id = (query.get("video_id") or [""])[0].strip()
        exp_id = (query.get("exp_id") or [""])[0].strip()
        frame_name = (query.get("name") or [""])[0].strip()

        root = self._resolve_root(dataset, split)
        if root is None or root.masks_root is None:
            self.send_error(HTTPStatus.BAD_REQUEST, "Mask root is unavailable")
            return

        if not exp_id or not frame_name:
            self.send_error(HTTPStatus.BAD_REQUEST, "Missing exp_id or frame name")
            return

        if exp_id == FLAT_MASK_SENTINEL:
            exp_dir = root.masks_root / video_id
        else:
            exp_dir = root.masks_root / video_id / exp_id
        frame_path = (exp_dir / frame_name).resolve()
        if not exp_dir.is_dir() or not frame_path.is_file() or not is_relative_to(frame_path, exp_dir):
            self.send_error(HTTPStatus.NOT_FOUND, "Mask frame not found")
            return

        mime_type, _ = mimetypes.guess_type(frame_path.name)
        with frame_path.open("rb") as handle:
            payload = handle.read()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _resolve_root(self, dataset: str, split: str) -> DatasetRoot | None:
        for root in self.known_roots:
            if root.dataset == dataset and root.split == split:
                return root
        return None

    def _frame_url(self, dataset: str, split: str, video_id: str, frame_name: str) -> str:
        return (
            f"/frame?dataset={quote(dataset)}&split={quote(split)}"
            f"&video_id={quote(video_id)}&name={quote(frame_name)}"
        )

    def _mask_url(self, dataset: str, split: str, video_id: str, exp_id: str, frame_name: str) -> str:
        return (
            f"/mask?dataset={quote(dataset)}&split={quote(split)}"
            f"&video_id={quote(video_id)}&exp_id={quote(exp_id)}&name={quote(frame_name)}"
        )

    def _try_serve_frontend(self, request_path: str) -> bool:
        if self.frontend_dist is None or not self.frontend_dist.is_dir():
            body = f"""
            <!doctype html>
            <html lang="en">
              <head>
                <meta charset="utf-8" />
                <title>Frame Gallery Agent</title>
                <style>
                  body {{ font-family: sans-serif; padding: 32px; line-height: 1.6; }}
                  code {{ background: #f3f3f3; padding: 2px 6px; border-radius: 4px; }}
                </style>
              </head>
              <body>
                <h1>Frame Gallery Agent Backend</h1>
                <p>The backend is running, but the frontend build was not found.</p>
                <p>Use the React dev server or build the frontend first.</p>
                <p>Expected frontend dist: <code>{self.frontend_dist}</code></p>
                <p>API health: <a href="/api/health">/api/health</a></p>
              </body>
            </html>
            """.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return True

        if request_path == "/":
            file_path = self.frontend_dist / "index.html"
        else:
            rel_path = request_path.lstrip("/")
            file_path = (self.frontend_dist / rel_path).resolve()
            if not is_relative_to(file_path, self.frontend_dist):
                self.send_error(HTTPStatus.FORBIDDEN, "Forbidden")
                return True
            if not file_path.exists():
                file_path = self.frontend_dist / "index.html"

        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Frontend asset not found")
            return True

        mime_type, _ = mimetypes.guess_type(file_path.name)
        payload = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
        return True

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[3]
    default_data_root = repo_root.parent / "Sa2VA" / "data" / "video_datas"
    default_frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"

    parser = argparse.ArgumentParser(description="Local frame gallery backend")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind")
    parser.add_argument(
        "--data-root",
        default=str(default_data_root),
        help="Root directory containing dataset folders",
    )
    parser.add_argument(
        "--frontend-dist",
        default=str(default_frontend_dist),
        help="Built frontend directory to serve at / when available",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    known_roots = build_known_roots(data_root)
    if not known_roots:
        raise SystemExit(f"No dataset frame roots found under: {data_root}")

    server = ThreadingHTTPServer((args.host, args.port), FrameGalleryHandler)
    server.known_roots = known_roots  # type: ignore[attr-defined]
    server.frontend_dist = Path(args.frontend_dist).resolve()  # type: ignore[attr-defined]

    print(f"[frame-gallery] serving on http://{args.host}:{args.port}")
    print("[frame-gallery] known roots:")
    for root in known_roots:
        print(f"  - {root.dataset}/{root.split}: {root.frames_root}")
    server.serve_forever()


if __name__ == "__main__":
    main()
