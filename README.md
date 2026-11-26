# Smart Interactive Whiteboard

Computer Vision project with real 21-landmark hand tracking, an index-finger cursor, pinch-to-click, distant-hand search, colored-marker fallback, four-corner calibration, presentation-remote input, drawing, editable objects, image viewing, and PowerPoint presentation support. The interface uses a projected pixel-art study theme.

## Start

```bash
python -m venv .venv
# macOS / Linux
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python main.py
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

The default source is **Synthetic**, so the interface can be tested immediately without hardware. Move the mouse to take temporary control in synthetic mode.

## Image Viewer

1. Open **Images → + Import Image** and select a PNG, JPG, or JPEG.
2. The file is copied into the project and decoded on a worker thread.
3. Use **Zoom +**, **Zoom -**, the mouse wheel, **2x close-up**, **Pan**, and **Fit / Reset View**.
4. Images retain their original aspect ratio at every zoom level. Right-click resets the view.


## Lesson presentation

PowerPoint import requires [LibreOffice](https://www.libreoffice.org/download/) installed with its `soffice` command available. On macOS the app also checks `/Applications/LibreOffice.app/Contents/MacOS/soffice`.

1. Open **Lesson → + Import PPTX** and select a `.ppt` or `.pptx` file.
2. LibreOffice converts it to PDF in an isolated background process; PyMuPDF renders the actual pages. `python-pptx` is not used for rendering.
3. Select a thumbnail or use **Previous slide / Next slide**. Long decks have paged thumbnails.
4. Choose **PRESENT FULLSCREEN** for a distraction-free slide view. Press **Escape** or **EXIT** to return without losing the deck.
5. Pinch or left-click on a slide to annotate. Ink uses normalized coordinates, survives fullscreen changes, and is stored independently for every slide. **CLEAR INK** affects only the current slide.

Missing LibreOffice, invalid images, corrupt decks, conversion failures, and empty PDFs produce a GUI status message without discarding existing content or blocking camera tracking.

## Tracking and remote

- **Settings → Pointer**: Hand + Fingers is the default. The index finger controls the cursor; thumb/index pinch clicks. Marker color mode remains available with sampling, auto-tune, and HSV controls.
- **Home → Calibrate**: click the four projected board corners. The app auto-orders them, calculates the homography, displays RMSE, and saves the configuration.
- **Settings → Camera**: cycle Synthetic → camera 0 → 1 → 2 and configure flips. Recalibrate afterward.
- **Settings → Remote**: inspect and rebind both remote buttons. Previous/right-click moves back during presentations.

## Other controls

| Input | Behavior |
|---|---|
| Remote Next / left mouse | Click at current pointer; start/stop ink in drawing areas |
| Remote Previous / right mouse | Undo, previous slide, reset image view, or transform an object |
| Draw left-click | Start/stop a stroke in Toggle mode; press/release in Hold mode |
| Presentation `N`, Space, Enter | Next slide |
| Presentation `P`, Backspace | Previous slide |
| Image `+` / `-` / wheel | Zoom in / out |
| `H` / `F` / `D` / `C` | Home / Fullscreen / Debug / Calibration |
| Escape during presentation | Exit presentation, preserving slides and annotations |
| `Q` / Escape elsewhere | Quit |

The current ink swatch and line-thickness preview are always visible in Draw; color buttons and the presentation **INK** control show their selected color directly.

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python main.py --synthetic --smoke-test --no-remote
.venv/bin/python tools/capture_demo.py
.venv/bin/python tools/benchmark_metrics.py
```

`benchmark_metrics.json` contains measured performance and calibration data. `REPORT_DATA_AND_GUIDE.md` provides the mathematical explanation and report-ready structure.

## Project map

- `core/`: camera, MediaPipe hand tracking, NumPy marker vision, calibration, asynchronous imports, remote input.
- `ui/`: aspect-correct NumPy primitives, pixel renderer, interaction state machine.
- `assets/`: educational images, mini-lesson, and workspace-local imported files.
- `tests/`: numerical, worker, image, PDF, presentation, annotation, and GUI workflow tests.
- `tools/`: asset generation, screenshots, and benchmarks.
