# PitchVision Web App

End-to-end starter for:

Browser → Node/Express → Python/FastAPI → YOLO/OpenCV/Scikit-Learn → JSON + visual outputs → Browser

## Folder

- `frontend/` — minimal PitchVision website
- `backend/` — Node.js + Express upload/API gateway
- `python_pipeline/` — FastAPI inference service and CV pipeline
- `python_pipeline/models/best.pt` — put your trained YOLO weights here
- `storage/uploads/` — incoming files
- `storage/outputs/` — generated results

## Run

### 1. Python

```bash
cd python_pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

### 2. Node

Open a second terminal:

```bash
cd backend
npm install
npm run dev
```

Open `http://localhost:5050`.

## Model

Copy your trained YOLO weights to:

`python_pipeline/models/best.pt`

The current starter expects player/ball class names to include `player`, `person`, or `football player`, and `ball`. Change the class-name logic in `pipeline/pipeline.py` if your dataset uses different labels.

## Important

Image inference is wired end-to-end. Video currently uses the first frame as a working integration test; the next step is to replace `run_video()` with your frame-by-frame detector + tracker + analytics loop and write the annotated MP4.
