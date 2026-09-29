from pathlib import Path
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "python_pipeline" / "models" / "best.pt"

OUTPUT_ROOT = ROOT / "storage" / "outputs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

MODEL = None


# ==================================================
# YOLO MODEL
# ==================================================

def get_model():
    global MODEL

    if MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"""
YOLO model not found.

Expected:
{MODEL_PATH}

Put your trained YOLO weights at:
python_pipeline/models/best.pt
"""
            )

        MODEL = YOLO(str(MODEL_PATH))

    return MODEL


# ==================================================
# PROGRESS
# ==================================================

def update(callback, progress, message, stage):
    if callback:
        callback(progress, message, stage)


# ==================================================
# CLASS NAME
# ==================================================

def get_class_name(names, index):
    if isinstance(names, dict):
        return str(names.get(index, index))

    if isinstance(names, list):
        return str(names[index])

    return str(index)


# ==================================================
# TEAM CLASSIFICATION
# ==================================================

def team_cluster(frame, boxes):
    if len(boxes) < 2:
        return [0 for _ in boxes]

    features = []

    for (x1, y1, x2, y2) in boxes:
        height = max(1, y2 - y1)
        xa = max(0, int(x1))
        xb = min(frame.shape[1], int(x2))
        ya = max(0, int(y1 + height * 0.20))
        yb = min(frame.shape[0], int(y1 + height * 0.75))

        crop = frame[ya:yb, xa:xb]

        if crop.size == 0:
            features.append([0, 0, 0])
            continue

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        features.append(np.mean(hsv.reshape(-1, 3), axis=0))

    labels = KMeans(
        n_clusters=2,
        n_init=10,
        random_state=42
    ).fit_predict(np.asarray(features))

    return labels.tolist()


# ==================================================
# PITCH VISUALIZATION
# ==================================================

def draw_pitch(points_a, points_b, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#19733f")

    # Pitch boundary
    ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color="white", linewidth=1)

    # Halfway line
    ax.plot([52.5, 52.5], [0, 68], color="white", linewidth=1)

    # Centre circle
    circle = plt.Circle((52.5, 34), 9.15, fill=False, color="white", linewidth=1)
    ax.add_patch(circle)

    for points, color in [(points_a, "#e63946"), (points_b, "#4da3ff")]:
        if points:
            ax.scatter(
                [p[0] for p in points],
                [p[1] for p in points],
                s=65,
                c=color,
                edgecolors="black",
                linewidths=0.5
            )

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ==================================================
# HEATMAP
# ==================================================

def draw_heatmap(points_a, points_b, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#19733f")

    all_points = np.asarray(points_a + points_b)

    if len(all_points):
        ax.hist2d(
            all_points[:, 0],
            all_points[:, 1],
            bins=(18, 12),
            range=[[0, 105], [0, 68]],
            cmap="turbo"
        )

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(output_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


# ==================================================
# DETECTION
# ==================================================

def detect_frame(frame):
    model = get_model()
    result = model(frame, verbose=False)[0]
    names = model.names

    boxes = []
    confidences = []
    classes = []

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = float(box.conf[0])
        class_index = int(box.cls[0])

        boxes.append((x1, y1, x2, y2))
        confidences.append(confidence)
        classes.append(get_class_name(names, class_index))

    return boxes, confidences, classes


# ==================================================
# FRAME ANALYSIS
# ==================================================

def analyze_frame(
    frame,
    output_dir,
    callback,
    write_visuals=True
):
    (
        boxes,
        confidences,
        classes
    ) = detect_frame(frame)

    player_indices = [
        i
        for i, name in enumerate(classes)
        if name.lower() in {"player", "person", "football player"}
    ]

    player_boxes = [boxes[i] for i in player_indices]
    labels = team_cluster(frame, player_boxes)

    annotated = frame.copy()
    team_points = [[], []]

    for j, index in enumerate(player_indices):
        x1, y1, x2, y2 = boxes[index]
        team = int(labels[j]) if j < len(labels) else 0
        color = (40, 70, 240) if team == 0 else (230, 170, 40)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"Team {'A' if team == 0 else 'B'} {confidences[index]:.2f}"
        cv2.putText(
            annotated,
            label,
            (x1, max(16, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2
        )

        # Baseline image -> pitch mapping (normalized)
        px = ((x1 + x2) / 2) / frame.shape[1] * 105
        py = y2 / frame.shape[0] * 68
        team_points[team].append((px, 68 - py))

    if write_visuals:
        annotated_path = output_dir / "annotated.jpg"
        cv2.imwrite(str(annotated_path), annotated)

    return annotated, classes, confidences, team_points


# ==================================================
# TEAM WIDTH
# ==================================================

def width(points):
    if not points:
        return 0.0

    xs = [p[0] for p in points]
    return max(xs) - min(xs)


# ==================================================
# BUILD RESULTS
# ==================================================

def build_result(
    job_id,
    output_dir,
    team_points,
    classes,
    confidences
):
    pitch_path = output_dir / "pitch.png"
    formation_path = output_dir / "formations.png"
    heatmap_path = output_dir / "heatmap.png"

    draw_pitch(team_points[0], team_points[1], pitch_path)
    draw_pitch(team_points[0], team_points[1], formation_path)
    draw_heatmap(team_points[0], team_points[1], heatmap_path)

    ball_confidences = [
        confidence
        for name, confidence in zip(classes, confidences)
        if "ball" in name.lower()
    ]

    metrics = {
        "players_detected": len(team_points[0]) + len(team_points[1]),
        "team_a_width": width(team_points[0]),
        "team_b_width": width(team_points[1]),
        "ball_confidence": max(ball_confidences, default=0.0)
    }

    return {
        "annotated_image": f"/files/outputs/{job_id}/annotated.jpg",
        "pitch_view": f"/files/outputs/{job_id}/pitch.png",
        "formations": f"/files/outputs/{job_id}/formations.png",
        "heatmap": f"/files/outputs/{job_id}/heatmap.png",
        "output_file": f"/files/outputs/{job_id}/annotated.jpg",
        "metrics": metrics
    }


# ==================================================
# IMAGE
# ==================================================

def run_image(job_id, image_path, callback):
    output_dir = OUTPUT_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise ValueError("OpenCV could not read the image.")

    update(callback, 15, "Image loaded.", "Preprocessing · OpenCV")

    annotated, classes, confidences, team_points = analyze_frame(
        frame, output_dir, callback
    )

    update(callback, 40, "Objects detected.", "Object Detection · YOLO")
    update(callback, 58, "Teams classified.", "Team Classification · K-Means")
    update(callback, 72, "Pitch coordinates generated.", "Spatial Mapping · OpenCV")
    update(callback, 90, "Metrics calculated.", "Formation + Tactical Analytics")

    result = build_result(
        job_id, output_dir, team_points, classes, confidences
    )
    result["input_type"] = "image"

    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2)
    )

    update(callback, 100, "Analysis complete.", "Result Aggregation · JSON + Visualizations")
    return result


# ==================================================
# VIDEO
# ==================================================

def run_video(job_id, video_path, callback):
    output_dir = OUTPUT_ROOT / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = output_dir / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width_px, height_px)
    )

    all_team_points = [[], []]
    first_frame_data = None
    frame_index = 0

    FRAME_STRIDE = 3

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % FRAME_STRIDE == 0:
            (
                annotated,
                classes,
                confidences,
                team_points
            ) = analyze_frame(
                frame,
                output_dir,
                callback,
                write_visuals=False
            )

            all_team_points[0].extend(team_points[0])
            all_team_points[1].extend(team_points[1])

            if first_frame_data is None:
                first_frame_data = (
                    annotated,
                    classes,
                    confidences,
                    team_points
                )

            writer.write(annotated)
        else:
            writer.write(frame)

        if total_frames > 0:
            progress = 15 + int(65 * frame_index / total_frames)
            update(
                callback,
                min(progress, 80),
                f"Processing video · frame {frame_index + 1}/{total_frames}",
                "Object Detection · Tracking baseline"
            )

        frame_index += 1

    cap.release()
    writer.release()

    if first_frame_data is None:
        raise ValueError("The video contained no readable frames.")

    first_annotated, classes, confidences, team_points = first_frame_data

    cv2.imwrite(str(output_dir / "annotated.jpg"), first_annotated)

    update(callback, 84, "Aggregating player positions.", "Spatial Mapping")

    result = build_result(
        job_id,
        output_dir,
        all_team_points,
        classes,
        confidences
    )

    result["annotated_video"] = f"/files/outputs/{job_id}/annotated.mp4"
    result["output_file"] = f"/files/outputs/{job_id}/annotated.mp4"
    result["input_type"] = "video"
    result["video_note"] = (
        "Video inference currently processes every 3rd frame. "
        "Add ByteTrack or BoT-SORT for persistent player IDs "
        "and temporal analytics."
    )

    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2)
    )

    update(callback, 100, "Analysis complete.", "Result Aggregation · Video + Analytics")
    return result


# ==================================================
# MAIN PIPELINE
# ==================================================

def run_pipeline(job_id, file_path, callback):
    extension = Path(file_path).suffix.lower()

    update(callback, 5, "Validating input.", "Input Validation")

    if extension in {".jpg", ".jpeg", ".png", ".webp"}:
        return run_image(job_id, file_path, callback)

    if extension in {".mp4", ".mov", ".avi", ".mkv"}:
        return run_video(job_id, file_path, callback)

    raise ValueError(f"Unsupported file type: {extension}")