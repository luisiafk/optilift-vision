# Optilif-Vision

> Numerical mechanics engine and kinematic analysis tool for powerlifting, powered by computer vision andrigorous mathematical modeling.

---

## What is OptiLift-Vision?

**OptiLift-Vision** is not a conventional fitness app driven by black-box Ai. It is a precision engineering tool that decouples visual data extraction from **pure numerical mathematics**.

The system uses computer vision solely as a discrete spatio-temporal coordinate generator, applying **Smoothed Cubic Splines**, high-precision numerical differentiation, and analytical physics models to compute the real kinematics of the lift (velocities, instantaneos accelerations, and detection of critical sticking points).

---

## Architecture & Numerical Pipeline

1. **Extraction (Vision Tracker):** Frame-by-frame video processing via `MediaPipe` to isolate $(x,y)$ coordinates of main joints and the barbell, which is detected directly from the video frames via edge tracking (Canny) and line detection (Hough); if no barbell is found the tracker falls back to the wrists midpoint as proxy.
2. **Conditioning & Smoothing (Math Engine):** Application of Smoothed Cubic Splines with curvature penalization ($E(S)$) to eliminate pixel quantization noise without losing the true dynamic of the movement.
3. **Analytical Differentiation:** Exact calculation of the first and second-order derivatives over the splines to obtain instantaneous velocity $v(t)$ and acceleration $a(t)$.
4. **Performance Metrics:** Automatic detection of the *Sticking Point* by identifying the local minimun of vertical acceleration during the concentric phase.
5. **Rendering (Video Overlay):** Generation of output video with superimposed telemetry, motion vectors, and graphical markers.

---

## Requirements & Installation

The project is optimized for **`uv`**, ensuring ultra-fast and clean dependecy managment.

### Prerequisites

* Python 3.10+
* `uv` installed on your system.

### Quick installation

1. Clone the repository:

    ```bash
    git clone https://github.com/luisiafk/optilift-vision.git
    cd optilift-vision

2. Sync dependencies with uv:

    ```bash
    uv sync

3. Run the analysis directly from your terminal using ClI interface:

    ```bash
    uv run python -m optilift_vision.cli --video assets/my_lift.mp4 --output assets/analyzed_results.mp4


## Available Arguments

* --video: Path to the profile video of the lift (.mp4).
* --output: Path where the resulting video with telemetry overlay will be saved (default: output_analyzed.mp4).

## Tech Stack

* Language: Python 3.10+
* Environment Manager: `uv`
* Mathematics & Numerical Computing: `NumPy`, `SciPy` (UnivariateSpline)
* Computer Vision: `OpenCV`, `MediaPipe`
* Terminal Interface: `Rich`

## Author

Developed with analytical rigor and a passion for system engineering by **Luis Enrique Cohalla Cabrera**