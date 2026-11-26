"""Asynchronous image and PowerPoint import/rendering services.

Selected files are copied into this workspace before processing. PowerPoint
rendering deliberately uses LibreOffice -> PDF -> PyMuPDF; python-pptx is not
used as a renderer. All expensive work runs on daemon worker threads and the
camera loop receives results through a thread-safe queue.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

import cv2
import numpy as np
import pymupdf


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
PRESENTATION_EXTENSIONS = {".ppt", ".pptx"}


class ImportFailure(RuntimeError):
    """A safe, user-displayable import error."""


def find_libreoffice() -> str | None:
    """Return a LibreOffice executable on Linux, Windows, or macOS."""
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if executable:
        return executable
    candidates = [
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        Path("C:/Program Files/LibreOffice/program/soffice.exe"),
        Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
    ]
    return next((str(path) for path in candidates if path.is_file()), None)


def load_image_file(source: Path, destination: Path) -> np.ndarray:
    """Decode a supported image without relying on OS-specific path handling."""
    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ImportFailure("Choose a PNG, JPG, or JPEG image.")
    data = np.fromfile(str(destination), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None or image.size == 0 or min(image.shape[:2]) < 1:
        raise ImportFailure("The selected image is invalid, empty, or damaged.")
    return image


def render_pdf_slides(pdf_path: Path, max_slides: int = 200) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render every PDF page to BGR pixels and create aspect-correct thumbnails."""
    slides: list[np.ndarray] = []
    thumbnails: list[np.ndarray] = []
    try:
        document = pymupdf.open(str(pdf_path))
    except Exception as exc:
        raise ImportFailure("LibreOffice did not produce a readable PDF.") from exc
    try:
        if len(document) == 0:
            raise ImportFailure("The imported presentation contains no slides.")
        if len(document) > max_slides:
            raise ImportFailure(f"This demo accepts up to {max_slides} slides per presentation.")
        for page in document:
            # Approximately 1440 px wide for a crisp projected 16:9 slide.
            scale = float(np.clip(1440.0 / max(page.rect.width, 1), 1.0, 2.0))
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), alpha=False)
            array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, pixmap.n)
            if pixmap.n < 3:
                array = np.repeat(array[..., :1], 3, axis=2)
            slide = array[..., :3][..., ::-1].copy()  # RGB/RGBA -> owned BGR
            slides.append(slide)

            thumb_w, thumb_h = 184, 82
            ratio = min(thumb_w / slide.shape[1], thumb_h / slide.shape[0])
            resized = cv2.resize(slide, (max(1, round(slide.shape[1] * ratio)),
                                         max(1, round(slide.shape[0] * ratio))),
                                interpolation=cv2.INTER_AREA)
            thumbnail = np.full((thumb_h, thumb_w, 3), (35, 40, 50), dtype=np.uint8)
            ox, oy = (thumb_w - resized.shape[1]) // 2, (thumb_h - resized.shape[0]) // 2
            thumbnail[oy:oy + resized.shape[0], ox:ox + resized.shape[1]] = resized
            thumbnails.append(thumbnail)
    except ImportFailure:
        raise
    except Exception as exc:
        raise ImportFailure("A slide could not be rendered. The presentation may be damaged.") from exc
    finally:
        document.close()
    return slides, thumbnails


class ImportManager:
    """Small queue-based worker service; no imports run on the camera thread."""

    def __init__(self, workspace: Path):
        # Track active daemon jobs separately from completed queued results.
        self.workspace = Path(workspace)
        self.results: queue.Queue[dict] = queue.Queue()
        self._lock = threading.Lock()
        self._active = 0
        self._closed = False

    @property
    def busy(self) -> bool:
        """Return whether at least one import worker is still active."""
        with self._lock:
            return self._active > 0

    def submit_image(self, source: str | Path, generation: int = 0) -> None:
        """Queue image decoding and copy the source into the workspace."""
        self._submit("image", Path(source), generation)

    def submit_presentation(self, source: str | Path, generation: int = 0) -> None:
        """Queue PowerPoint conversion and slide rendering."""
        self._submit("presentation", Path(source), generation)

    def _submit(self, kind: str, source: Path, generation: int) -> None:
        """Create one daemon worker and increment the active-job counter."""
        with self._lock:
            if self._closed:
                raise ImportFailure("The import service is closed.")
            self._active += 1
        thread = threading.Thread(target=self._run, args=(kind, source, generation),
                                  name=f"StudyBoard-{kind}-import", daemon=True)
        thread.start()

    def _run(self, kind: str, source: Path, generation: int) -> None:
        """Copy and convert one image or presentation without touching the UI thread."""
        job = self.workspace / "assets" / "imports" / uuid.uuid4().hex
        try:
            job.mkdir(parents=True, exist_ok=False)
            if not source.is_file():
                raise ImportFailure("The selected file no longer exists.")
            local_source = job / ("source" + source.suffix.lower())
            shutil.copy2(source, local_source)
            if kind == "image":
                result = {"kind": "image", "name": source.name, "path": local_source,
                        "image": load_image_file(source, local_source)}
            else:
                if source.suffix.lower() not in PRESENTATION_EXTENSIONS:
                    raise ImportFailure("Choose a PowerPoint .ppt or .pptx file.")
                executable = find_libreoffice()
                if not executable:
                    raise ImportFailure("LibreOffice is required to import PowerPoint. Install it, then try again.")
                profile = job / "libreoffice-profile"
                profile.mkdir()
                completed = subprocess.run(
                    [executable, "--headless", f"-env:UserInstallation={profile.as_uri()}",
                    "--convert-to", "pdf", "--outdir", str(job), str(local_source)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120, check=False,
                )
                pdf_path = job / "source.pdf"
                if completed.returncode != 0 or not pdf_path.is_file():
                    detail = (completed.stderr or completed.stdout).strip().splitlines()
                    suffix = f" ({detail[-1][:120]})" if detail else ""
                    raise ImportFailure("PowerPoint conversion failed. Try saving the deck again." + suffix)
                slides, thumbnails = render_pdf_slides(pdf_path)
                result = {"kind": "presentation", "name": source.name, "path": local_source,
                        "pdf_path": pdf_path, "slides": slides, "thumbnails": thumbnails}
            result["generation"] = generation
            self.results.put(result)
        except ImportFailure as exc:
            self.results.put({"kind": "error", "message": str(exc), "generation": generation})
        except subprocess.TimeoutExpired:
            self.results.put({"kind": "error", "generation": generation,
                            "message": "PowerPoint conversion timed out. Try a smaller presentation."})
        except Exception:
            self.results.put({"kind": "error", "generation": generation,
                            "message": "Import failed. Check that the file is valid and try again."})
        finally:
            with self._lock:
                self._active -= 1

    def poll(self) -> list[dict]:
        """Return all currently completed import results and leave the queue empty."""
        results = []
        while True:
            try:
                results.append(self.results.get_nowait())
            except queue.Empty:
                return results

    def close(self) -> None:
        """Prevent new jobs during application shutdown; existing jobs remain bounded."""
        # Workers are daemons and conversion has a hard timeout; never freeze exit.
        with self._lock:
            self._closed = True
