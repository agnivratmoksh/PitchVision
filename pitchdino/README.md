# PitchVision — Tactical Football Vision & Analytics

PitchVision is a computer vision and tactical analysis system that converts single-frame match broadcast images into 2D bird's-eye pitch geometry, formation structures, and passing recommendations.

---

## Architecture Overview

```
Frontend (HTML / Vanilla CSS / JS)
          │  (Port 5050)
          ▼
Node.js Express 5 Gateway (`backend/server.js`)
          │  (Port 8000 via HTTP Multipart & Proxied Results)
          ▼
FastAPI Tactical Analytics Engine (`python_pipeline/football_analyzer.py`)
          ├── 1. DINOv2 + Faster R-CNN Object Detection (`models/best_dino.pt`)
          ├── 2. Pitch Calibration & Homography (Keypoints / RANSAC / Grass-shape fallback)
          ├── 3. Team Classification & Keeper Association (LAB Color Clustering)
          ├── 4. Geometric Pass Modeling (Ground, Lofted, Through Balls)
          ├── 5. Formation Detection & Line Analysis (K-Means Template Matching)
          └── 6. Recommendation Engine (Open Passing Lanes & Dribble Angles)
```

---

## Quickstart

### 1. Start the Python Tactical Service

Navigate to the `python_pipeline` directory and activate the virtual environment:

```bash
cd python_pipeline
source venv/bin/activate          # Python 3.11 — use venv/, not .venv/
export ROBOFLOW_API_KEY="your_key_here"
python football_analyzer.py --serve --checkpoint models/best_dino.pt --port 8000
```

> **Note:** If `--checkpoint` is omitted, the service automatically defaults to `models/best_dino.pt` if present.
> The `ROBOFLOW_API_KEY` environment variable is required for pitch calibration.

### 2. Start the Node.js Web Server

In a separate terminal, navigate to the `backend` directory:

```bash
cd backend
npm install   # if dependencies are not already installed
npm start     # or npm run dev
```

Open your browser at `http://localhost:5050`.

---

## API Endpoints

### Node Gateway (`http://localhost:5050`)
- `GET /api/health` — Checks status of the Node gateway and the Python DINO service.
- `POST /api/analyze` — Accepts an image upload (`file`) along with optional tactical parameters (`passer_team`, `passer_index`, `attack_red`, `attack_blue`).
- `GET /python-results/*` — Proxies generated tactical diagram PNGs and `analysis.json` from the Python engine.
- `GET /*` — Serves the frontend web application.

### Python Engine (`http://127.0.0.1:8000`)
- `GET /health` — Health check endpoint returning detector status and model availability.
- `POST /analyze` — Core image processing pipeline returning tactical detections, formations, pass evaluations, and generated diagrams.
- `GET /results/*` — Serves static output diagrams and analysis artifacts.
