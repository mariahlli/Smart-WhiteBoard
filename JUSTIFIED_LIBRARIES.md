# Justified Libraries and External Calls

## Implementation policy

The essential vision, geometry, and rendering operations are implemented in Python and NumPy. OpenCV is not used as an image-processing or geometry shortcut in the running application. The only OpenCV operations are camera acquisition, window/event management, image-file I/O, and text rendering. OpenCV HSV conversion appears **only in the unit test** as an independent reference oracle.

| Library / call | File | Purpose and justification |
|---|---|---|
| `cv2.VideoCapture`, `isOpened`, `read`, `release` | `core/camera.py` | Cross-platform camera acquisition and resource cleanup; explicitly allowed by the brief. |
| `VideoCapture.set/get`, `CAP_PROP_*` | `core/camera.py` | Request and inspect camera dimensions. These are device operations, not vision algorithms. |
| `cv2.namedWindow`, `imshow`, `destroyAllWindows` | `main.py` | Create and display the application window. |
| `cv2.setMouseCallback` | `main.py` | GUI calibration, color sampling, and an offline mouse fallback. |
| `cv2.setWindowProperty` | `main.py` | Fullscreen/windowed display toggle. |
| `cv2.waitKey` | `main.py` | Pump the GUI event loop and provide focused-window fallback keys; remote input uses `pynput` instead. |
| `cv2.imread` | `ui/interaction.py` | Load educational PNG assets. |
| `cv2.imwrite` | `tools/generate_assets.py` | Save original programmatically generated PNG assets. |
| `cv2.putText`, `getTextSize`, font/line constants | `ui/renderer.py`, `core/camera.py`, `tools/generate_assets.py` | Readable projected labels and synthetic-camera annotation. Text layout is not a core CV algorithm. |
| `cv2.cvtColor(..., COLOR_BGR2HSV)` | `tests/test_core.py` only | Independent test reference for our NumPy BGR→HSV conversion; never called by the application. |
| `pynput.keyboard.Listener` | `core/remote_input.py` | OS-level remote key press/release capture, including Page Up/Down and arrow keys. A focused-window fallback remains available. |
| `mediapipe.solutions.hands.Hands` | `core/hand_tracking.py` | Learned 21-landmark hand inference, explicitly requested for true hand/finger recognition. The project owns crop search, landmark restoration, fingertip selection, finger-state geometry, pinch hysteresis, and skeleton rendering. |
| `numpy.linalg.svd/inv` | `core/calibration.py` | Numerical linear-algebra primitives used inside our explicitly constructed normalized DLT solver; no predefined homography function is used. |
| Shifted NumPy slices, `minimum`, `maximum` | `core/vision.py` | Separable vectorized erosion and dilation implemented directly by the project; no predefined morphology function. |
| `pytest` | `tests/` | Automated verification only; not needed for the vision pipeline. |

## NumPy-owned operations

- BGR→HSV and BGR→YCbCr conversion.
- Multi-channel thresholding and HSV/YCbCr skin fusion.
- Binary erosion, dilation, opening, and closing.
- Run-length connected components, largest-region selection, exact area, bounding box, and centroid.
- Adaptive EMA smoothing, deadzone, and outlier rejection.
- Hartley normalization, DLT matrix assembly, SVD solution, homogeneous projection, inverse projection, and RMSE validation.
- Thick antialiased lines, circles, rectangles, crosshair, ripples, alpha blending, bilinear resize, zoom crop, and pan.
- Synthetic-camera board rendering and marker generation.
