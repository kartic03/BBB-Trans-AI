"""
BBB-TransAI web application.

Serves the interface in web/ and a small JSON API over Python's standard
library http.server, so the deployment needs nothing beyond the packages the
prediction pipeline already requires.

    python app.py                  # http://localhost:8501
    BBB_PORT=9000 python app.py

Environment:
    BBB_HOST   interface to bind, default 0.0.0.0
    BBB_PORT   port to bind, default 8501
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import secrets
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import bbb_core as core

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

HOST = os.environ.get("BBB_HOST", "0.0.0.0")
PORT = int(os.environ.get("BBB_PORT", "8501"))

MAX_BODY = 4 * 1024 * 1024
MAX_ACTIVE_JOBS = 4
JOB_TTL_SECONDS = 1800

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".webmanifest": "application/manifest+json",
}

JOB_ID = re.compile(r"^[0-9a-f]{16}$")


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, job_id: str) -> None:
        self.id = job_id
        self.state = "running"
        self.stage = "Preparing"
        self.step = 0
        self.total = len(core.DESCRIPTORS)
        self.result: dict | None = None
        self.error: str | None = None
        self.created = time.time()

    def snapshot(self) -> dict:
        payload = {
            "job": self.id,
            "state": self.state,
            "stage": self.stage,
            "step": self.step,
            "total": self.total,
        }
        if self.state == "done":
            payload["result"] = self.result
        elif self.state == "error":
            payload["error"] = self.error
        return payload


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        with self._lock:
            self._sweep()
            active = sum(1 for j in self._jobs.values() if j.state == "running")
            if active >= MAX_ACTIVE_JOBS:
                raise core.PipelineError(
                    "The server is already running the maximum number of predictions. "
                    "Please try again in a moment."
                )
            job = Job(secrets.token_hex(8))
            self._jobs[job.id] = job
            return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _sweep(self) -> None:
        cutoff = time.time() - JOB_TTL_SECONDS
        for key in [k for k, j in self._jobs.items() if j.created < cutoff and j.state != "running"]:
            del self._jobs[key]


JOBS = JobStore()


def run_job(job: Job, pairs: list[tuple[str, str]], structures: dict[str, float]) -> None:
    def progress(step: int, total: int, descriptor: str) -> None:
        job.step, job.total = step, total
        job.stage = f"Computing {descriptor} descriptors"

    try:
        job.stage = "Loading model"
        payload = core.predict(pairs, structures=structures, progress=progress)
        job.result = payload
        job.stage = "Complete"
        job.step = job.total
        job.state = "done"
    except core.PipelineError as exc:
        job.error, job.state = str(exc), "error"
    except Exception:
        traceback.print_exc()
        job.error = "The prediction failed unexpectedly. The server log has the details."
        job.state = "error"


# ---------------------------------------------------------------------------
# request handling
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "BBBTransAI"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # -- helpers ----------------------------------------------------------
    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf8")
        self._send(status, body, "application/json; charset=utf-8", {"Cache-Control": "no-store"})

    def _fail(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise core.PipelineError("Malformed request length.")
        if length <= 0:
            raise core.PipelineError("Empty request.")
        if length > MAX_BODY:
            raise core.PipelineError("The request is larger than this server accepts.")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise core.PipelineError("The request body is not valid JSON.")
        if not isinstance(payload, dict):
            raise core.PipelineError("The request body must be a JSON object.")
        return payload

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    # -- routes -----------------------------------------------------------
    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            return self._static("index.html", cache=False)
        if path == "/api/status":
            try:
                return self._json(HTTPStatus.OK, core.status())
            except core.PipelineError as exc:
                return self._fail(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
        if path.startswith("/api/jobs/"):
            return self._job(path.rsplit("/", 1)[-1])
        if path.startswith("/api/export/"):
            return self._export(path.rsplit("/", 1)[-1])
        if path.startswith("/assets/"):
            return self._static(path[len("/assets/"):])

        self._fail(HTTPStatus.NOT_FOUND, "Not found.")

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/jobs":
            return self._create_job()
        if path == "/api/structure":
            return self._structure()
        self._fail(HTTPStatus.NOT_FOUND, "Not found.")

    # -- implementations --------------------------------------------------
    def _static(self, relative: str, cache: bool = True) -> None:
        candidate = (WEB / relative).resolve()
        try:
            candidate.relative_to(WEB.resolve())
        except ValueError:
            return self._fail(HTTPStatus.FORBIDDEN, "Forbidden.")
        if not candidate.is_file():
            return self._fail(HTTPStatus.NOT_FOUND, "Not found.")

        body = candidate.read_bytes()
        # Revalidate rather than cache for a fixed time, so a redeployed
        # interface is never served from a stale copy; unchanged files still
        # answer 304 and cost nothing.
        etag = '"%s"' % hashlib.sha256(body).hexdigest()[:24]
        if self.headers.get("If-None-Match") == etag:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        content_type = CONTENT_TYPES.get(candidate.suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, body, content_type, {
            "Cache-Control": "no-cache",
            "ETag": etag,
        })

    def _create_job(self) -> None:
        try:
            payload = self._read_body()
            text = payload.get("peptides")
            if not isinstance(text, str) or not text.strip():
                raise core.PipelineError("No peptide sequences were submitted.")
            if len(text) > MAX_BODY // 2:
                raise core.PipelineError("The peptide list is too large.")

            pairs = core.parse_batch(text)
            if not pairs:
                raise core.PipelineError("No peptide sequences could be read from the input.")

            structures = {}
            raw_structures = payload.get("structures") or {}
            if isinstance(raw_structures, dict):
                for pid, value in list(raw_structures.items())[:core.MAX_PEPTIDES]:
                    try:
                        rg = float(value)
                    except (TypeError, ValueError):
                        continue
                    if 0.5 < rg < 500:
                        structures[str(pid)] = rg

            job = JOBS.create()
        except core.PipelineError as exc:
            return self._fail(HTTPStatus.BAD_REQUEST, str(exc))

        thread = threading.Thread(target=run_job, args=(job, pairs, structures), daemon=True)
        thread.start()
        self._json(HTTPStatus.ACCEPTED, job.snapshot())

    def _job(self, job_id: str) -> None:
        if not JOB_ID.match(job_id):
            return self._fail(HTTPStatus.BAD_REQUEST, "Malformed job id.")
        job = JOBS.get(job_id)
        if job is None:
            return self._fail(HTTPStatus.NOT_FOUND, "That prediction has expired.")
        self._json(HTTPStatus.OK, job.snapshot())

    def _structure(self) -> None:
        """Compute a measured radius of gyration from an uploaded PDB/mmCIF model."""
        try:
            payload = self._read_body()
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                raise core.PipelineError("The structure file was empty.")
            rg, residues = core.radius_of_gyration_from_structure(content)
        except core.PipelineError as exc:
            return self._fail(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception:
            traceback.print_exc()
            return self._fail(HTTPStatus.BAD_REQUEST, "That file could not be read as a structure.")

        self._json(HTTPStatus.OK, {
            "radius_gyration": round(rg, 4),
            "residues": residues,
            "name": str(payload.get("name") or "")[:120],
        })

    def _export(self, filename: str) -> None:
        job_id = filename.removesuffix(".csv")
        if not JOB_ID.match(job_id):
            return self._fail(HTTPStatus.BAD_REQUEST, "Malformed job id.")
        job = JOBS.get(job_id)
        if job is None or job.state != "done" or not job.result:
            return self._fail(HTTPStatus.NOT_FOUND, "No finished prediction with that id.")

        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow([
            "ID", "Sequence", "BBB_probability", "BBB_prediction", "structure_mode",
            "length", "net_charge", "frac_positive", "frac_hydrophobic", "frac_aromatic",
            "radius_gyration", "compactness",
        ])
        for row in job.result["results"]:
            comp = row["composition"]
            writer.writerow([
                row["id"], row["sequence"], f"{row['probability']:.4f}",
                int(row["permeable"]), row["structure_mode"],
                comp["length"], comp["net_charge"], comp["frac_positive"],
                comp["frac_hydrophobic"], comp["frac_aromatic"],
                comp["radius_gyration"], comp["compactness"],
            ])

        body = buffer.getvalue().encode("utf8")
        self._send(HTTPStatus.OK, body, "text/csv; charset=utf-8", {
            "Content-Disposition": 'attachment; filename="bbb-transai-predictions.csv"',
            "Cache-Control": "no-store",
        })


def main() -> None:
    if not WEB.is_dir():
        raise SystemExit(f"Interface files not found. Expected a web/ directory at {WEB}")

    info = core.status()
    print("BBB-TransAI")
    if info.get("model"):
        card = info["model"]
        print(f"  model     {card['name']}  ({card['kind']}, {card['n_trees']} trees, "
              f"{card['n_features']} descriptors)")
    else:
        print(f"  model     NOT LOADED - {info.get('error', 'unknown error')}")
    if not info["ifeature"]:
        print(f"  iFeature  MISSING - expected at {info['ifeature_path']}")
        print("            Predictions will fail until iFeature is installed there.")
    else:
        print("  iFeature  ready")
    print(f"  listening on http://{'localhost' if HOST == '0.0.0.0' else HOST}:{PORT}")
    print("  press Ctrl+C to stop")

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
