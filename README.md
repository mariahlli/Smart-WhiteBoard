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

## Drawing

Open **Draw** from the home screen to write or sketch on the board. Select a red, blue, green, or black pen from the left menu; the color swatches show the available ink. **Thickness** cycles through 3, 5, and 9 pixels. **Eraser** uses a wider stroke to remove ink.

- In **Toggle** mode, click or pinch once to start a stroke and again to stop.
- In **Hold** mode, hold the click/pinch while moving to draw; releasing ends the stroke.
- Use **Undo stroke** to remove the most recent stroke, or **Clear canvas** to erase all drawing from the current canvas.

The hand cursor follows the index finger, and pinching thumb and index finger clicks. Mouse clicks also work; in synthetic mode, mouse movement temporarily controls the pointer. The drawing tools are available alongside imported slides, where annotations stay separate for each slide.

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

Color swatches in the Draw menu identify each pen, and the presentation **INK** control shows its selected color.


## Project map

- `core/`: camera, MediaPipe hand tracking, NumPy marker vision, calibration, asynchronous imports, remote input.
- `ui/`: aspect-correct NumPy primitives, pixel renderer, interaction state machine.

