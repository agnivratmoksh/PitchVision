#!/usr/bin/env python3
"""
football_analyzer.py  --  ONE IMAGE IN  ->  separate PNGs + analysis.json OUT.

Usage
-----
  python football_analyzer.py match.jpg  --checkpoint last_checkpoint.pt --out results
  python football_analyzer.py ./images/  --checkpoint last_checkpoint.pt --out results
  python football_analyzer.py --serve    --checkpoint last_checkpoint.pt        # FastAPI for the website

Environment
-----------
  ROBOFLOW_API_KEY   pitch-keypoint model key (or set CONFIG["keypoint_weights_path"] to local YOLOv8-pose weights)
  FOOTBALL_CKPT      alternative to --checkpoint
  ALLOWED_ORIGINS    comma separated CORS origins for --serve (default "*")

Per image folder  results/<image_name>/
  01_calibration_check.png     photo + projected pitch lines (cyan) + agreeing keypoints
  02_detections.png            photo + team-coloured boxes, goalkeepers, referees, ball
  03_birdseye.png              bird's-eye view, carrier marked
  04_ground_passes_<team>.png  every teammate lane, open/blocked, defender perpendicular distance
  05_lofted_passes_<team>.png  lofted option to every teammate + landing-zone pressure
  06_through_balls_<team>.png  runs beyond the line: lane, race margin, offside risk
  07_formation_<team>.png      detected lines (e.g. 4-3-3) + 0-10 rating
  08_recommendation_<team>.png best option, or dribble direction
  analysis.json                every number + warnings + skip reasons
All PNGs are the same pixel size.  <team> is red / blue (only the team in possession when the ball is found,
both teams when it is not).
"""
import os, sys, json, hashlib, argparse, warnings, shutil, uuid
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans

try:
    from inference_sdk import InferenceHTTPClient
except ImportError:
    InferenceHTTPClient = None

# =====================================================================================================
# CONFIG
# =====================================================================================================
_default_ckpt = os.environ.get("FOOTBALL_CKPT", "")
if not _default_ckpt:
    _candidate = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "best_dino.pt")
    if os.path.exists(_candidate):
        _default_ckpt = _candidate

CONFIG = {
    "checkpoint_path": _default_ckpt,
    "score_threshold": 0.5,
    "score_threshold_per_class": {"player": 0.6},     # a little lower than the notebook: keeps more players
    "nms_iou_threshold": 0.4,

    "min_person_count": 4,             # below this the image really has nothing to analyse
    "min_players_for_team_split": 4,

    # ---- calibration ----
    "roboflow_api_key": os.environ.get("ROBOFLOW_API_KEY", ""),
    "keypoint_model_id": "football-field-detection-f07vi/14",
    "keypoint_weights_path": None,
    "keypoint_attempts": [(0.6, 6.0), (0.45, 8.0), (0.3, 10.0)],   # (min confidence, RANSAC px) - tried in order
    "min_keypoints": 4,                # 4 non-collinear points define a homography; quality is labelled
    "min_pitch_hull_m2": 100.0,
    "min_on_pitch_frac": 0.6,
    "min_spread_m": 3.0,
    "max_outside_m": 15.0,
    "allow_approx_fallback": True,     # grass-shape fallback when keypoints fail (flagged "approximate")
    "min_grass_frac": 0.20,
    "cache_dir": "./keypoint_cache",

    # ---- pass model ----
    "interception_radius_m": 2.0,
    "lane_t_range": (0.1, 0.9),
    "contest_radius_base_m": 4.0,
    "contest_radius_per_10m": 0.6,
    "min_lofted_distance_m": 10.0,
    "through_max_runner_dist_m": 30.0,
    "through_target_beyond_line_m": 2.0,
    "min_recommend_score": 45.0,       # below this -> dribble
    "dribble_corridor_m": 3.0,
    "dribble_max_look_m": 15.0,

    # ---- formation ----
    "formation_min_outfield": 4,
    "formation_reliable_outfield": 8,

    # ---- rendering (all PNGs identical size) ----
    "fig_size": (13, 7.5),
    "dpi": 110,
}

L, W = 120.0, 70.0
PB_W, PB_L = 41.0, 20.15
GB_W, GB_L = 18.32, 5.5
CC_R, PEN_SPOT = 9.15, 11.0
PITCH_VERTICES = np.array([
    (0, 0), (0, (W-PB_W)/2), (0, (W-GB_W)/2), (0, (W+GB_W)/2), (0, (W+PB_W)/2), (0, W),
    (GB_L, (W-GB_W)/2), (GB_L, (W+GB_W)/2), (PEN_SPOT, W/2),
    (PB_L, (W-PB_W)/2), (PB_L, (W-GB_W)/2), (PB_L, (W+GB_W)/2), (PB_L, (W+PB_W)/2),
    (L/2, 0), (L/2, W/2-CC_R), (L/2, W/2+CC_R), (L/2, W),
    (L-PB_L, (W-PB_W)/2), (L-PB_L, (W-GB_W)/2), (L-PB_L, (W+GB_W)/2), (L-PB_L, (W+PB_W)/2),
    (L-PEN_SPOT, W/2),
    (L-GB_L, (W-GB_W)/2), (L-GB_L, (W+GB_W)/2),
    (L, 0), (L, (W-PB_W)/2), (L, (W-GB_W)/2), (L, (W+GB_W)/2), (L, (W+PB_W)/2), (L, W),
    (L/2-CC_R, W/2), (L/2+CC_R, W/2),
], dtype=np.float32)

TEAMS = ("red", "blue")
PERSON = ("player", "goalkeeper", "referee")
TEAM_COL = {"red": "#e63946", "blue": "#3a86ff", "unknown": "#9aa0a6"}
BG = "#101418"


def to_py(o):
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer, int)): return int(o)
    if isinstance(o, (np.bool_, bool)): return bool(o)
    if isinstance(o, np.generic): return o.item()
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, set): return list(o)
    raise TypeError(type(o))


def clamp01(x): return float(max(0.0, min(1.0, x)))
def P(d): return np.array([d["pitch_x"], d["pitch_y"]], float)
def r1(x): return None if x is None else round(float(x), 2)


# =====================================================================================================
# 1. DETECTOR  (your DINOv2 + Faster R-CNN checkpoint, loaded exactly like the notebook)
# =====================================================================================================
_DET = {}

def load_detector(ckpt):
    if "model" in _DET:
        return _DET
    import torch, torch.nn as nn
    from torchvision.models.detection import FasterRCNN
    from torchvision.models.detection.rpn import AnchorGenerator
    from torchvision.models.detection.transform import GeneralizedRCNNTransform
    from torchvision.ops import MultiScaleRoIAlign
    PATCH = 14
    DIMS = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}
    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class Backbone(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.model = torch.hub.load("facebookresearch/dinov2", name, pretrained=False)
            self.embed_dim = DIMS[name]; self.out_channels = self.embed_dim
            for p in self.model.parameters(): p.requires_grad = False
            self.model.eval()
        def forward(self, x):
            b, c, h, w = x.shape
            with torch.no_grad():
                f = self.model.forward_features(x)["x_norm_patchtokens"]
            return {"0": f.permute(0, 2, 1).reshape(b, self.embed_dim, h // PATCH, w // PATCH)}
    try:
        ck = torch.load(ckpt, map_location=dev, weights_only=False)
    except TypeError:
        ck = torch.load(ckpt, map_location=dev)
    tc, names = ck["config"], ck["class_names"]
    m = FasterRCNN(Backbone(tc["model_name"]), num_classes=len(names) + 1,
                   rpn_anchor_generator=AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),)),
                   box_roi_pool=MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2),
                   min_size=tc["img_size"], max_size=tc["img_size"], image_mean=MEAN, image_std=STD)
    m.transform = GeneralizedRCNNTransform(min_size=tc["img_size"], max_size=tc["img_size"],
                                           image_mean=MEAN, image_std=STD, size_divisible=PATCH)
    m.load_state_dict(ck["model_state_dict"]); m.to(dev).eval()
    _DET.update(model=m, names=names, dev=dev)
    return _DET


def run_detection(img_pil, ckpt):
    import torch
    import torchvision.transforms.functional as TF
    from torchvision.ops import nms
    d = load_detector(ckpt); m, names, dev = d["model"], d["names"], d["dev"]
    with torch.no_grad():
        pred = m([TF.to_tensor(img_pil).to(dev)])[0]
    boxes, labels, scores = pred["boxes"].cpu(), pred["labels"].cpu(), pred["scores"].cpu()
    per = CONFIG["score_threshold_per_class"] or {}
    keep = torch.tensor([bool(scores[i] >= per.get(names[labels[i].item() - 1], CONFIG["score_threshold"]))
                         for i in range(len(scores))], dtype=torch.bool)
    boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
    idx = []
    for c in labels.unique():
        cm = (labels == c).nonzero(as_tuple=True)[0]
        idx += cm[nms(boxes[cm], scores[cm], CONFIG["nms_iou_threshold"])].tolist()
    out = []
    for i in idx:
        x1, y1, x2, y2 = boxes[i].tolist()
        out.append({"class": names[labels[i].item() - 1], "confidence": float(scores[i]),
                    "bbox_px": [x1, y1, x2, y2], "ground_point_px": [(x1 + x2) / 2, y2]})
    return out


# =====================================================================================================
# 2. CALIBRATION  (keypoints -> homography; uses whichever keypoints agree with each other)
# =====================================================================================================
_client = None; _local_kp_model = None

def get_keypoints(path):
    """[[x, y, conf] * 32] indexed by canonical pitch-landmark id. Cached by image content."""
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    key = hashlib.md5(open(path, "rb").read()).hexdigest()
    cp = os.path.join(CONFIG["cache_dir"], key + ".json")
    if os.path.exists(cp):
        return json.load(open(cp))
    kps = []
    global _client, _local_kp_model
    if CONFIG["keypoint_weights_path"]:
        if _local_kp_model is None:
            from ultralytics import YOLO
            _local_kp_model = YOLO(CONFIG["keypoint_weights_path"])
        r = _local_kp_model(path, verbose=False)[0]
        if r.keypoints is not None and len(r.boxes) > 0:
            b = int(r.boxes.conf.argmax())
            xy = r.keypoints.xy[b].cpu().numpy()
            cf = r.keypoints.conf[b].cpu().numpy() if r.keypoints.conf is not None else np.ones(len(xy))
            kps = [[float(x), float(y), float(c)] for (x, y), c in zip(xy, cf)]
    else:
        if _client is None:
            if InferenceHTTPClient is None:
                raise RuntimeError("pip install inference-sdk (or set keypoint_weights_path)")
            if not CONFIG["roboflow_api_key"]:
                raise RuntimeError("set ROBOFLOW_API_KEY (or keypoint_weights_path)")
            _client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key=CONFIG["roboflow_api_key"])
        res = _client.infer(path, model_id=CONFIG["keypoint_model_id"])
        preds = res.get("predictions", [])
        if preds:
            best = max(preds, key=lambda p: p.get("confidence", 0))
            kps = [[0.0, 0.0, 0.0] for _ in range(len(PITCH_VERTICES))]
            for k in best.get("keypoints", []):            # index by class_id, never by list position
                idx = k.get("class_id")
                if idx is None:
                    try: idx = int(k.get("class_name", k.get("class", "")))
                    except (TypeError, ValueError): continue
                if 0 <= idx < len(kps):
                    kps[idx] = [float(k["x"]), float(k["y"]), float(k.get("confidence", 0.0))]
    json.dump(kps, open(cp, "w"))
    return kps


def _hull_area(pts):
    pts = np.asarray(pts, np.float32)
    return float(cv2.contourArea(cv2.convexHull(pts))) if len(pts) >= 3 else 0.0


def compute_homography(kps, img_size, conf, ransac_px):
    """-> (H image->pitch | None, inlier ids, rmse, reason)."""
    iw, ih = img_size
    ids = [i for i, (x, y, c) in enumerate(kps) if c >= conf and i < len(PITCH_VERTICES)]
    if len(ids) < CONFIG["min_keypoints"]:
        return None, ids, None, f"only {len(ids)} keypoints >= conf {conf}"
    ip = np.array([kps[i][:2] for i in ids], np.float32); pp = PITCH_VERTICES[ids]
    if _hull_area(pp) < CONFIG["min_pitch_hull_m2"] or _hull_area(ip) < 0.01 * iw * ih:
        return None, ids, None, "keypoints (almost) collinear"
    Hp, mask = cv2.findHomography(pp, ip, cv2.RANSAC, ransac_px)
    if Hp is None or mask is None:
        return None, ids, None, "RANSAC failed"
    inl = mask.ravel().astype(bool)
    if inl.sum() < CONFIG["min_keypoints"]:
        return None, ids, None, f"only {int(inl.sum())} keypoints agree"
    if _hull_area(pp[inl]) < CONFIG["min_pitch_hull_m2"]:
        return None, ids, None, "agreeing keypoints (almost) collinear"
    Hp, _ = cv2.findHomography(pp[inl], ip[inl], 0)
    if Hp is None:
        return None, ids, None, "refit failed"
    proj = cv2.perspectiveTransform(pp[inl].reshape(-1, 1, 2), Hp).reshape(-1, 2)
    rmse = float(np.sqrt(((proj - ip[inl]) ** 2).sum(1).mean()))
    if rmse > ransac_px:
        return None, ids, rmse, f"reprojection error {rmse:.1f}px"
    try: H = np.linalg.inv(Hp)
    except np.linalg.LinAlgError: return None, ids, rmse, "singular"
    return H, [i for i, k in zip(ids, inl) if k], rmse, ""


def plausible_projection(xy):
    xy = np.asarray(xy, float)
    xy = xy[np.isfinite(xy).all(1)] if len(xy) else xy
    if len(xy) < 3: return False, "projection produced invalid positions"
    m = 8.0
    inside = float(((xy[:, 0] >= -m) & (xy[:, 0] <= L + m) & (xy[:, 1] >= -m) & (xy[:, 1] <= W + m)).mean())
    if inside < CONFIG["min_on_pitch_frac"]: return False, "most people project off the pitch"
    sd = np.linalg.svd(xy - xy.mean(0), compute_uv=False) / np.sqrt(len(xy))
    if sd[1] < CONFIG["min_spread_m"]: return False, "projected people collapse onto a line"
    return True, ""


def grass_quad_homography(img_rgb):
    h, w = img_rgb.shape[:2]; sc = 640.0 / max(h, w)
    small = cv2.resize(img_rgb, (max(1, int(w * sc)), max(1, int(h * sc))))
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 40) & (hsv[..., 2] >= 40)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None, None, "no grass found"
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) / mask.size < CONFIG["min_grass_frac"]: return None, None, "too little grass"
    hull = cv2.convexHull(cnt); peri = cv2.arcLength(hull, True); quad = None
    for eps in np.linspace(0.01, 0.15, 30):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4: quad = ap.reshape(4, 2).astype(np.float32); break
        if len(ap) < 4: break
    if quad is None: quad = cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float32)
    q = quad[np.argsort(quad[:, 1])]
    top, bot = q[:2][np.argsort(q[:2, 0])], q[2:][np.argsort(q[2:, 0])]
    quad = np.array([top[0], top[1], bot[1], bot[0]], np.float32)
    if cv2.contourArea(quad) / mask.size < 0.15: return None, None, "grass shape degenerate"
    qf = quad / sc
    H = cv2.getPerspectiveTransform(qf.astype(np.float32), np.array([[0, 0], [L, 0], [L, W], [0, W]], np.float32))
    return H, qf, ""


def project_points(H, pts):
    return cv2.perspectiveTransform(np.array(pts, np.float32).reshape(-1, 1, 2), H).reshape(-1, 2)


def calibrate(image_path, img_rgb, dets):
    """Returns (H | None, info, overlay).  Tries progressively looser keypoint settings, then the grass fallback."""
    ih, iw = img_rgb.shape[:2]
    info = {"method": None, "quality": None, "n_keypoints_used": 0, "rmse_px": None, "attempts": [], "warnings": []}
    overlay = {}
    people = [d["ground_point_px"] for d in dets if d["class"] in PERSON]
    have_src = bool(CONFIG["keypoint_weights_path"] or CONFIG["roboflow_api_key"])
    kps = None
    if have_src and image_path:
        try: kps = get_keypoints(image_path)
        except Exception as e: info["warnings"].append(f"keypoint detection failed: {e}")
    else:
        info["warnings"].append("no keypoint source configured (ROBOFLOW_API_KEY / keypoint_weights_path)")
    if kps:
        for conf, rpx in CONFIG["keypoint_attempts"]:
            H, ids, rmse, why = compute_homography(kps, (iw, ih), conf, rpx)
            att = {"conf": conf, "ransac_px": rpx, "ok": H is not None, "why": why}
            if H is not None:
                ok, why2 = plausible_projection(project_points(H, people)) if people else (True, "")
                att["plausible"] = ok
                if not ok: att["why"] = why2
            info["attempts"].append(att)
            if H is None or not att.get("plausible", True): continue
            n = len(ids)
            q = "good" if (n >= 8 and rmse <= 4) else ("fair" if n >= 5 else "low")
            info.update(method="keypoints", quality=q, n_keypoints_used=n, rmse_px=r1(rmse))
            if q == "low":
                info["warnings"].append(f"only {n} agreeing keypoints - positions are less accurate")
            overlay = {"H": H, "ids": ids, "kps": kps}
            return H, info, overlay
    if CONFIG["allow_approx_fallback"]:
        H, quad, why = grass_quad_homography(img_rgb)
        if H is not None:
            ok, why2 = plausible_projection(project_points(H, people)) if people else (False, "no people")
            if ok:
                info.update(method="approx_grass", quality="approximate")
                info["warnings"].append("keypoint calibration failed; grass-shape fallback used - "
                                        "treat all distances as rough (only valid if the whole pitch is visible)")
                return H, info, {"H": H, "quad": quad}
            info["attempts"].append({"method": "approx_grass", "ok": False, "why": why2})
        else:
            info["attempts"].append({"method": "approx_grass", "ok": False, "why": why})
    return None, info, overlay


def set_pitch_positions(dets, H):
    if not dets: return
    xy = project_points(H, [d["ground_point_px"] for d in dets]); m = CONFIG["max_outside_m"]
    for d, (x, y) in zip(dets, xy):
        if np.isfinite(x) and np.isfinite(y) and -m <= x <= L + m and -m <= y <= W + m:
            d["pitch_x"], d["pitch_y"] = float(x), float(y)


# =====================================================================================================
# 3. TEAMS, KEEPERS, ATTACK DIRECTION
# =====================================================================================================
def torso_color_lab(img, bbox):
    Hh, Ww = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]; bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 8: return None
    x1, x2 = max(0, x1 + int(.2 * bw)), min(Ww, x2 - int(.2 * bw))
    ty1, ty2 = max(0, y1 + int(.15 * bh)), min(Hh, y1 + int(.55 * bh))
    crop = img[ty1:ty2, x1:x2]
    if crop.size == 0: return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    green = (hsv[..., 0] >= 30) & (hsv[..., 0] <= 90) & (hsv[..., 1] >= 50) & (hsv[..., 2] >= 40)
    px = crop[~green]
    if len(px) < 20: px = crop.reshape(-1, 3)
    lab = cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    return np.median(lab.astype(np.float32), 0)


def assign_outfield_teams(img, dets, warnings_):
    for d in dets: d["team"] = None
    players = [d for d in dets if d["class"] == "player"]
    feats = [torso_color_lab(img, d["bbox_px"]) for d in players]
    ok = [i for i, f in enumerate(feats) if f is not None]
    if len(ok) >= CONFIG["min_players_for_team_split"]:
        X = np.array([feats[i] for i in ok])
        km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
        RED, BLUE = np.array([80., 67.]), np.array([79., -108.])
        def redness(c):
            ab = X[km.labels_ == c][:, 1:3].mean(0) - 128.0
            return np.linalg.norm(ab - BLUE) - np.linalg.norm(ab - RED)
        rc = max(range(2), key=redness)
        for i, lab in zip(ok, km.labels_): players[i]["team"] = "red" if lab == rc else "blue"
        if np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]) < 15:
            warnings_.append("the two team colours are very similar - team split may be unreliable")
    for d in players:
        if d["team"] is None: d["team"] = "unknown"


def assign_keepers(dets, warnings_):
    """Keeper -> team whose OUTFIELD players sit closest to that keeper's goal line on average.
    (Replaces 'nearest player', which fails when an attacker stands next to the keeper.)
    With two keepers at opposite ends the pair is assigned jointly."""
    gks = [d for d in dets if d["class"] == "goalkeeper" and "pitch_x" in d]
    for d in dets:
        if d["class"] == "goalkeeper": d["team"] = "unknown"; d["gk_confidence"] = None
    if not gks: return
    xs = {t: np.array([d["pitch_x"] for d in dets if d["class"] == "player" and d.get("team") == t and "pitch_x" in d])
          for t in TEAMS}
    def cost(t, gk):
        end = 0.0 if gk["pitch_x"] < L / 2 else L
        return float(np.mean(np.abs(xs[t] - end))) if len(xs[t]) >= 2 else np.inf
    lo, hi = min(gks, key=lambda g: g["pitch_x"]), max(gks, key=lambda g: g["pitch_x"])
    if len(gks) >= 2 and lo["pitch_x"] < L / 2 <= hi["pitch_x"]:
        a = cost("red", lo) + cost("blue", hi); b = cost("blue", lo) + cost("red", hi)
        if np.isfinite(min(a, b)):
            lt, ht = ("red", "blue") if a <= b else ("blue", "red")
            conf = "high" if abs(a - b) > 8 else "low"
            lo["team"], hi["team"], lo["gk_confidence"], hi["gk_confidence"] = lt, ht, conf, conf
            if conf == "low": warnings_.append("goalkeeper team assignment is ambiguous")
            gks = [g for g in gks if g is not lo and g is not hi]
    for g in gks:
        cr, cb = cost("red", g), cost("blue", g)
        if not np.isfinite(min(cr, cb)): continue
        g["team"] = "red" if cr < cb else "blue"
        g["gk_confidence"] = "high" if abs(cr - cb) > 8 else "low"
        if g["gk_confidence"] == "low": warnings_.append("goalkeeper team assignment is ambiguous")


def infer_attack_directions(dets, override, warnings_):
    """Each team attacks away from its own keeper. Returns {team: 'increasing_x'|'decreasing_x'|None}, source."""
    dirs = {t: None for t in TEAMS}; src = {t: "unknown" for t in TEAMS}
    for t, v in (override or {}).items():
        if v in ("increasing_x", "decreasing_x"): dirs[t], src[t] = v, "user_override"
    for g in dets:
        if g["class"] != "goalkeeper" or g.get("team") not in TEAMS or "pitch_x" not in g: continue
        t = g["team"]
        if dirs[t] is not None: continue
        if 40 < g["pitch_x"] < 80:
            warnings_.append(f"{t} goalkeeper is far from both goals - attack direction not inferred from it"); continue
        dirs[t] = "increasing_x" if g["pitch_x"] < L / 2 else "decreasing_x"; src[t] = "own_goalkeeper"
    for t in TEAMS:                                     # opponent keeper's goal is the target
        o = "blue" if t == "red" else "red"
        if dirs[t] is None and dirs[o] is not None:
            dirs[t] = "decreasing_x" if dirs[o] == "increasing_x" else "increasing_x"; src[t] = "opponent_direction"
    if dirs["red"] and dirs["red"] == dirs["blue"]:
        warnings_.append("both teams got the same attack direction (team/keeper mix-up) - direction cleared")
        dirs = {t: None for t in TEAMS}; src = {t: "unknown" for t in TEAMS}
    return dirs, src


def sgn(direction): return 1.0 if direction == "increasing_x" else -1.0


def label_detections(dets):
    for d in dets: d["label"] = None; d["team_index"] = None
    for t, pre in (("red", "R"), ("blue", "B")):
        mem = sorted([d for d in dets if d["class"] == "player" and d.get("team") == t and "pitch_x" in d],
                     key=lambda d: (d["pitch_x"], d["pitch_y"]))
        for i, d in enumerate(mem, 1): d["label"], d["team_index"] = f"{pre}{i}", i - 1
        for d in dets:
            if d["class"] == "goalkeeper" and d.get("team") == t: d["label"] = f"{pre}-GK"
    for d in dets:
        if d["class"] == "referee": d["label"] = "REF"
        if d["class"] == "ball": d["label"] = "ball"
        if d["label"] is None: d["label"] = "?"


# =====================================================================================================
# 4. PASS GEOMETRY + SCORING
# =====================================================================================================
def point_to_segment(Pp, T, D):
    Pp, T, D = map(lambda a: np.asarray(a, float), (Pp, T, D))
    s = T - Pp; sl = float(s @ s)
    if sl == 0: return float(np.linalg.norm(D - Pp)), 0.0, Pp
    t = float(np.clip((D - Pp) @ s / sl, 0, 1)); c = Pp + t * s
    return float(np.linalg.norm(D - c)), t, c


def lane_check(p, target, opps):
    """Nearest defender to the lane (perpendicular distance, ends excluded)."""
    lo, hi = CONFIG["lane_t_range"]; best = (None, None, None)
    for o in opps:
        d, t, c = point_to_segment(p, target, P(o))
        if t < lo or t > hi: continue
        if best[0] is None or d < best[0]: best = (d, o, c)
    return best


def nearest_opp_dist(pt, opps):
    return min((float(np.linalg.norm(P(o) - pt)) for o in opps), default=None)


def progress(p, target, direction):
    if direction is None: return 0.5
    return clamp01(((target[0] - p[0]) * sgn(direction) + 10.0) / 40.0)


def dist_fit(d, lo=5, hi=30, far=60):
    if d < lo: return 0.5
    if d <= hi: return 1.0
    return clamp01(1 - 0.7 * (d - hi) / (far - hi)) if d < far else 0.3


def opt_ground(p, mate, opps, direction):
    t = P(mate); dist = float(np.linalg.norm(t - p))
    md, nd, cp = lane_check(p, t, opps)
    blocked = md is not None and md < CONFIG["interception_radius_m"]
    clear = 1.0 if md is None else clamp01((md - 0.5) / 3.5)
    ro = nearest_opp_dist(t, opps); space = 1.0 if ro is None else clamp01(ro / 7.0)
    score = 100 * (0.35 * clear + 0.25 * space + 0.25 * progress(p, t, direction) + 0.15 * dist_fit(dist))
    if blocked: score *= 0.25
    return {"type": "ground", "receiver": mate["label"], "receiver_pos": t, "distance_m": r1(dist),
            "min_defender_dist_to_lane_m": r1(md), "blocked": bool(blocked),
            "nearest_defender_pos": None if nd is None else P(nd), "closest_lane_point": cp,
            "receiver_nearest_opponent_m": r1(ro), "score": r1(score), "viable": not blocked}


def opt_lofted(p, mate, opps, direction):
    t = P(mate); dist = float(np.linalg.norm(t - p))
    rad = CONFIG["contest_radius_base_m"] + CONFIG["contest_radius_per_10m"] * max(0.0, dist - 20) / 10
    n = sum(1 for o in opps if np.linalg.norm(P(o) - t) <= rad)
    level = "low" if n == 0 else ("moderate" if n == 1 else "high")
    ro = nearest_opp_dist(t, opps); space = 1.0 if ro is None else clamp01(ro / 7.0)
    lfit = 1.0 if 18 <= dist <= 50 else (0.4 if dist < CONFIG["min_lofted_distance_m"] else (0.7 if dist < 18 else 0.5))
    score = 100 * 0.9 * (0.35 * (1 - clamp01(n / 2)) + 0.25 * space + 0.25 * progress(p, t, direction) + 0.15 * lfit)
    short = dist < CONFIG["min_lofted_distance_m"]
    return {"type": "lofted", "receiver": mate["label"], "receiver_pos": t, "distance_m": r1(dist),
            "landing_pressure": level, "opponents_near_landing": n, "contest_radius_m": r1(rad),
            "receiver_nearest_opponent_m": r1(ro), "too_short": bool(short), "score": r1(score),
            "viable": bool(level != "high" and not short)}


def offside_line(opps_all, passer_pos, direction):
    xs = sorted([o["pitch_x"] for o in opps_all], reverse=(direction == "increasing_x"))
    if not xs: return None, "no_opponents"
    has_gk = any(o["class"] == "goalkeeper" for o in opps_all)
    line = xs[1] if len(xs) >= 2 else xs[0]
    basis = "second_last_defender" if has_gk else "last_visible_defender (keeper not visible)"
    line = max(line, passer_pos[0]) if direction == "increasing_x" else min(line, passer_pos[0])
    return float(line), basis


def through_balls(p, mates, opps, direction):
    if direction is None:
        return {"evaluated": False, "reason": "attack direction unknown", "options": []}
    line, basis = offside_line(opps, p, direction)
    if line is None:
        return {"evaluated": False, "reason": "no opponents visible", "options": []}
    s = sgn(direction); opts = []
    for m in mates:
        if m["class"] != "player": continue
        r = P(m); beyond = (r[0] - line) * s > 0
        target = np.array([line + s * CONFIG["through_target_beyond_line_m"], r[1]])
        if (target[0] - p[0]) * s <= 0: continue                     # not a forward ball
        run = float(np.linalg.norm(r - target)); dist = float(np.linalg.norm(target - p))
        if run > CONFIG["through_max_runner_dist_m"]: continue      # runner too far to reach the space
        md, nd, cp = lane_check(p, target, opps)
        lane_clear = md is None or md > CONFIG["interception_radius_m"]
        no = nearest_opp_dist(target, opps); margin = None if no is None else no - run
        clear = 1.0 if md is None else clamp01((md - 0.5) / 3.5)
        race = 0.7 if margin is None else clamp01((margin + 2) / 8)
        score = 100 * (0.3 * clear + 0.3 * race + 0.25 * progress(p, target, direction) + 0.15 * dist_fit(dist))
        viable = (not beyond) and lane_clear and (margin is None or margin > -1.0)
        if beyond: score = 0.0
        opts.append({"type": "through", "receiver": m["label"], "runner_pos": r, "target_pos": target,
                     "distance_m": r1(dist), "runner_run_m": r1(run), "lane_clear": bool(lane_clear),
                     "min_defender_dist_to_lane_m": r1(md), "race_margin_m": r1(margin),
                     "offside_risk": bool(beyond), "score": r1(score), "viable": bool(viable)})
    return {"evaluated": True, "offside_line_x": line, "offside_line_basis": basis, "options": opts}


def dribble_option(p, opps, direction):
    best = None
    for a in range(0, 360, 15):
        u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]); free = CONFIG["dribble_max_look_m"]
        for o in opps:
            v = P(o) - p; along = float(v @ u)
            if along <= 0: continue
            if abs(float(v[0] * u[1] - v[1] * u[0])) < CONFIG["dribble_corridor_m"]:
                free = min(free, along - 1.0)
        for ax_, lim in ((0, L), (1, W)):                            # keep inside the pitch
            if abs(u[ax_]) > 1e-6:
                tb = ((lim if u[ax_] > 0 else 0.0) - p[ax_]) / u[ax_]
                free = min(free, max(0.0, tb))
        free = max(0.0, free)
        sc = free + (4.0 * float(u[0]) * sgn(direction) if direction else 0.0)
        if best is None or sc > best[0]: best = (sc, a, free, u)
    _, a, free, u = best
    step = min(free, 10.0)
    if direction:
        f = float(u[0]) * sgn(direction)
        head = "forward" if f > 0.5 else ("backward" if f < -0.5 else "sideways")
    else:
        head = "open space"
    return {"angle_deg": a, "free_space_m": r1(free), "target_pos": p + u * step, "heading": head}


def select_passer(team, dets, ball, override, warnings_):
    cands = [d for d in dets if d.get("team") == team and d["class"] in ("player", "goalkeeper") and "pitch_x" in d]
    if not cands: return None, None
    if override and override.get("team") == team and override.get("index") is not None:
        m = [d for d in cands if d.get("team_index") == override["index"]]
        if m: return m[0], {"method": "user_override", "assumed": False}
    if ball is not None:
        b = P(ball); d = min(cands, key=lambda c: np.linalg.norm(P(c) - b))
        gap = float(np.linalg.norm(P(d) - b))
        return d, {"method": "nearest_to_ball", "assumed": False, "ball_distance_m": r1(gap)}
    allp = np.array([P(d) for d in dets if d["class"] in ("player", "goalkeeper") and "pitch_x" in d])
    dens = lambda q: float(np.exp(-((allp - q) ** 2).sum(1) / (2 * 8.0 ** 2)).sum())
    d = max(cands, key=lambda c: dens(P(c)))
    return d, {"method": "densest_cluster", "assumed": True}


def analyze_team(team, dets, ball, direction, override, warnings_):
    passer, how = select_passer(team, dets, ball, override, warnings_)
    if passer is None:
        return {"team": team, "status": "no_players", "passer": None}
    p = P(passer)
    mates = [d for d in dets if d.get("team") == team and d["class"] in ("player", "goalkeeper")
             and "pitch_x" in d and d is not passer]
    opps = [d for d in dets if d.get("team") in TEAMS and d["team"] != team
            and d["class"] in ("player", "goalkeeper") and "pitch_x" in d]
    ground = [opt_ground(p, m, opps, direction) for m in mates]
    lofted = [opt_lofted(p, m, opps, direction) for m in mates]
    thr = through_balls(p, mates, opps, direction)
    allopts = [o for o in ground + lofted + thr["options"] if o["viable"]]
    allopts.sort(key=lambda o: -o["score"])
    rec = None
    if allopts and allopts[0]["score"] >= CONFIG["min_recommend_score"]:
        b = allopts[0]
        if b["type"] == "ground":
            why = f"Ground pass to {b['receiver']}: lane clear" + \
                  (f" (nearest defender {b['min_defender_dist_to_lane_m']} m from the lane)" if b["min_defender_dist_to_lane_m"] is not None else "") + \
                  f", nearest opponent to receiver {b['receiver_nearest_opponent_m']} m."
        elif b["type"] == "lofted":
            why = f"Lofted pass to {b['receiver']}: {b['landing_pressure']} landing pressure ({b['opponents_near_landing']} opponents within {b['contest_radius_m']} m)."
        else:
            why = f"Through ball for {b['receiver']}: lane clear, race margin {b['race_margin_m']} m, onside."
        rec = {"action": "pass", "option": b, "reason": why}
    drib = dribble_option(p, opps, direction)
    if rec is None:
        rec = {"action": "dribble", "dribble": drib,
               "reason": "No pass option reached the minimum score; "
                         f"dribble {drib['heading']} (angle {drib['angle_deg']} deg, {drib['free_space_m']} m free)."}
    return {"team": team, "status": "ok", "passer": {"label": passer["label"], "team_index": passer.get("team_index"),
            "pos": p, **how}, "attack_direction": direction, "ground": ground, "lofted": lofted, "through": thr,
            "ranked_viable_options": allopts[:5], "dribble_option": drib, "recommendation": rec}


# =====================================================================================================
# 5. FORMATION + RATING
# =====================================================================================================
FORMATIONS = {"4-4-2": [4, 4, 2], "4-3-3": [4, 3, 3], "4-2-3-1": [4, 2, 3, 1], "3-5-2": [3, 5, 2],
              "3-4-3": [3, 4, 3], "5-3-2": [5, 3, 2], "5-4-1": [5, 4, 1], "4-5-1": [4, 5, 1],
              "4-1-4-1": [4, 1, 4, 1], "4-4-1-1": [4, 4, 1, 1], "4-3-2-1": [4, 3, 2, 1], "3-4-1-2": [3, 4, 1, 2]}


def scale_counts(tpl, n):
    raw = np.array(tpl, float) * n / sum(tpl); c = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - c))[: n - c.sum()]: c[i] += 1
    return c


def rate_formation(team, dets, direction):
    out = [d for d in dets if d["class"] == "player" and d.get("team") == team and "pitch_x" in d]
    n = len(out); res = {"team": team, "outfield_visible": n, "reliable": n >= CONFIG["formation_reliable_outfield"]}
    if n < CONFIG["formation_min_outfield"]:
        return {**res, "formation": None, "rating": None, "note": f"only {n} outfield players visible"}
    xs = np.array([d["pitch_x"] for d in out]); ys = np.array([d["pitch_y"] for d in out])
    best = None
    for k in range(2, min(5, n) + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(xs.reshape(-1, 1))
        order = np.argsort(km.cluster_centers_.ravel()); rank = {c: i for i, c in enumerate(order)}
        lab = np.array([rank[c] for c in km.labels_]); counts = [int((lab == i).sum()) for i in range(k)]
        centers = [float(xs[lab == i].mean()) for i in range(k)]
        within = float(np.mean([xs[lab == i].std() for i in range(k)]))
        gap = float(np.min(np.diff(centers))) if k > 1 else 10.0
        q = clamp01(1 - within / max(gap, 1.0))
        for name, tpl in FORMATIONS.items():
            if len(tpl) != k: continue
            for rev in (False, True):
                cnt = counts[::-1] if rev else counts
                fit = clamp01(1 - float(np.abs(np.array(cnt) - scale_counts(tpl, n)).sum()) / n)
                comb = 0.7 * fit + 0.3 * q
                if best is None or comb > best[0]:
                    best = (comb, k, name, fit, rev, lab, counts, centers, within)
    if best is None:
        return {**res, "formation": None, "rating": None, "note": "could not group players into lines"}
    _, k, name, fit, rev, lab, counts, centers, within = best
    if direction is not None:                       # defenders = lines nearest own goal
        rev = (direction == "decreasing_x"); orient = "from_attack_direction"
    else:
        orient = "assumed_best_template_match"
    lines = [{"count": counts[i], "center_x": r1(centers[i]), "std_m": r1(xs[lab == i].std()),
              "members": [out[j]["label"] for j in range(n) if lab[j] == i]} for i in range(k)]
    if rev: lines = lines[::-1]
    fstr = "-".join(str(l["count"]) for l in lines)
    compact = clamp01(1 - (within - 2.5) / 5.5)
    width = clamp01((ys.max() - ys.min()) / (0.6 * W))
    gaps = np.abs(np.diff(sorted(centers)))
    sp = float(np.mean([1.0 if 8 <= g <= 20 else (g / 8 if g < 8 else max(0.0, 1 - (g - 20) / 20)) for g in gaps]))
    rating = 10 * (0.40 * fit + 0.25 * compact + 0.20 * width + 0.15 * sp)
    word = "excellent" if rating >= 8 else "good" if rating >= 6.5 else "fair" if rating >= 5 else "poor"
    return {**res, "formation": fstr, "closest_template": name, "orientation": orient, "lines": lines,
            "rating": r1(rating), "rating_label": word,
            "components": {"template_match": r1(fit), "compactness": r1(compact), "width": r1(width), "spacing": r1(sp)},
            "note": "geometric estimate, not a coach's judgement" + ("" if n >= CONFIG["formation_reliable_outfield"]
                                                                   else " - PARTIAL VIEW, low reliability")}


# =====================================================================================================
# 6. RENDERING (every PNG has the same pixel size)
# =====================================================================================================
def save_fig(fig, path):
    fig.savefig(path, dpi=CONFIG["dpi"], facecolor=fig.get_facecolor()); plt.close(fig)


def photo_page(img, title, subtitle=""):
    fig = plt.figure(figsize=CONFIG["fig_size"], facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 0.93]); ax.imshow(img); ax.axis("off"); ax.set_facecolor(BG)
    fig.text(0.01, 0.965, title, color="white", fontsize=13, weight="bold", va="center")
    if subtitle: fig.text(0.99, 0.965, subtitle, color="#c9d1d9", fontsize=9, ha="right", va="center")
    return fig, ax


def pitch_polylines():
    lines = [np.array([[0, 0], [L, 0], [L, W], [0, W], [0, 0]], float), np.array([[L/2, 0], [L/2, W]], float)]
    for x0, s in [(0, 1), (L, -1)]:
        bx, gx = x0 + s * PB_L, x0 + s * GB_L
        lines.append(np.array([[x0, (W-PB_W)/2], [bx, (W-PB_W)/2], [bx, (W+PB_W)/2], [x0, (W+PB_W)/2]], float))
        lines.append(np.array([[x0, (W-GB_W)/2], [gx, (W-GB_W)/2], [gx, (W+GB_W)/2], [x0, (W+GB_W)/2]], float))
    t = np.linspace(0, 2 * np.pi, 90)
    lines.append(np.c_[L/2 + CC_R * np.cos(t), W/2 + CC_R * np.sin(t)])
    return lines


def densify(pl, step=1.0):
    out = []
    for a, b in zip(pl[:-1], pl[1:]):
        out.append(np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / step)), endpoint=False))
    out.append(pl[-1:]); return np.vstack(out)


def draw_pitch_on_image(ax, H, shape, anchors=None):
    ih, iw = shape[:2]; Hi = np.linalg.inv(H)
    if anchors is not None and len(anchors):
        a = np.asarray(anchors, float); ref = np.sign(np.median((Hi @ np.c_[a, np.ones(len(a))].T)[2])) or 1.0
    else:
        c = H @ np.array([iw / 2, ih / 2, 1.0]); c = c[:2] / c[2]
        ref = np.sign((Hi @ np.array([c[0], c[1], 1.0]))[2]) or 1.0
    for pl in pitch_polylines():
        pts = densify(pl); hp = (Hi @ np.c_[pts, np.ones(len(pts))].T).T
        ok = (hp[:, 2] * ref) > 1e-6
        xy = hp[:, :2] / np.where(ok, hp[:, 2], 1.0)[:, None]
        ok &= (xy[:, 0] > -iw) & (xy[:, 0] < 2 * iw) & (xy[:, 1] > -ih) & (xy[:, 1] < 2 * ih)
        xy[~ok] = np.nan
        ax.plot(xy[:, 0], xy[:, 1], "-", color="cyan", lw=1.4, alpha=0.9)
    ax.set_xlim(0, iw); ax.set_ylim(ih, 0)


def bounds_for(dets):
    pts = np.array([[d["pitch_x"], d["pitch_y"]] for d in dets if "pitch_x" in d and d["class"] in PERSON + ("ball",)])
    if len(pts) == 0: return (-5, L + 5, W + 5, -5)
    lo, hi = pts.min(0) - 12, pts.max(0) + 12
    if hi[0] - lo[0] < 45: c = (hi[0] + lo[0]) / 2; lo[0], hi[0] = c - 22.5, c + 22.5
    if hi[1] - lo[1] < 28: c = (hi[1] + lo[1]) / 2; lo[1], hi[1] = c - 14, c + 14
    return (lo[0], hi[0], hi[1], lo[1])


def bird_page(title, subtitle, dets, bnd, panel_lines=None, carrier=None, ball=None):
    fig = plt.figure(figsize=CONFIG["fig_size"], facecolor=BG)
    ax = fig.add_axes([0.02, 0.05, 0.68, 0.86]); pn = fig.add_axes([0.71, 0.05, 0.28, 0.86])
    ax.set_facecolor("#1b5e34")
    for pl in pitch_polylines(): ax.plot(pl[:, 0], pl[:, 1], color="white", lw=1.4, zorder=1)
    ax.set_xlim(bnd[0], bnd[1]); ax.set_ylim(bnd[2], bnd[3]); ax.set_aspect("equal", adjustable="box"); ax.axis("off")
    for d in dets:
        if "pitch_x" not in d or d["class"] == "ball": continue
        c = "#ffd60a" if d["class"] == "referee" else TEAM_COL.get(d.get("team"), TEAM_COL["unknown"])
        mk = "^" if d["class"] == "referee" else ("s" if d["class"] == "goalkeeper" else "o")
        ax.scatter(d["pitch_x"], d["pitch_y"], s=150, c=c, marker=mk, edgecolors="white", linewidths=1.2, zorder=5)
        ax.text(d["pitch_x"], d["pitch_y"] - 1.9, d["label"], color="white", fontsize=7.5, ha="center", zorder=6)
    if ball is not None:
        ax.scatter(*P(ball), s=170, c="white", marker="*", edgecolors="black", zorder=7)
    if carrier is not None:
        ax.scatter(*carrier["pos"], s=420, facecolors="none", edgecolors="#ffd60a", linewidths=3, zorder=8)
    fig.text(0.01, 0.965, title, color="white", fontsize=13, weight="bold", va="center")
    fig.text(0.99, 0.965, subtitle, color="#c9d1d9", fontsize=9, ha="right", va="center")
    pn.set_facecolor("#161b22"); pn.set_xticks([]); pn.set_yticks([])
    for s in pn.spines.values(): s.set_color("#30363d")
    y = 0.98
    for ln in (panel_lines or []):
        bold = ln.startswith("#")
        pn.text(0.03, y, ln.lstrip("#"), color="white" if bold else "#c9d1d9", fontsize=8.6 if bold else 8,
                family="monospace", weight="bold" if bold else "normal", va="top", transform=pn.transAxes)
        y -= 0.036
    return fig, ax


def arc(a, b, bow=0.18):
    a, b = np.asarray(a, float), np.asarray(b, float); m = (a + b) / 2; d = b - a
    n = np.array([-d[1], d[0]]) * bow; c = m + n; t = np.linspace(0, 1, 30)[:, None]
    return (1 - t) ** 2 * a + 2 * (1 - t) * t * c + t ** 2 * b


def render_all(out, img, dets, calib, overlay, ball, carriers, reports, forms, dirs, meta):
    os.makedirs(out, exist_ok=True); files = {}
    sub = f"calibration: {calib['method']} ({calib['quality']})" if calib.get("method") else "calibration failed"
    # 01
    fig, ax = photo_page(img, "01  Calibration check", sub + "  |  cyan lines should sit on the real pitch lines")
    if overlay.get("H") is not None:
        anchors = [PITCH_VERTICES[i] for i in (overlay.get("ids") or [])] or None
        draw_pitch_on_image(ax, overlay["H"], img.shape, anchors)
        for i in (overlay.get("ids") or []):
            ax.scatter(*overlay["kps"][i][:2], s=30, c="#ffd60a", edgecolors="black", zorder=5)
    else:
        ax.text(0.5, 0.5, "NO CALIBRATION\n" + (meta.get("skip_reason") or ""), color="#ff6b6b", fontsize=18,
                weight="bold", ha="center", va="center", transform=ax.transAxes,
                bbox=dict(facecolor="black", alpha=0.7))
    save_fig(fig, os.path.join(out, "01_calibration_check.png")); files["calibration_check"] = "01_calibration_check.png"
    # 02
    fig, ax = photo_page(img, "02  Detections", "boxes coloured by team  |  white = goalkeeper  |  yellow = referee")
    for d in dets:
        x1, y1, x2, y2 = d["bbox_px"]
        c = "white" if d["class"] == "ball" else ("#ffd60a" if d["class"] == "referee" else TEAM_COL.get(d.get("team"), TEAM_COL["unknown"]))
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=c,
                                        linewidth=3.2 if d["class"] == "goalkeeper" else 2))
        tag = "GK" if d["class"] == "goalkeeper" else ("REF" if d["class"] == "referee" else ("ball" if d["class"] == "ball" else d["label"]))
        ax.text(x1, y1 - 3, tag, color="white", fontsize=7.5, bbox=dict(facecolor=c, alpha=0.75, pad=1, edgecolor="none"))
    if ball is None:
        ax.text(0.01, 0.02, "ball not detected", color="#ffd60a", fontsize=11, transform=ax.transAxes,
                bbox=dict(facecolor="black", alpha=0.6))
    save_fig(fig, os.path.join(out, "02_detections.png")); files["detections"] = "02_detections.png"
    if overlay.get("H") is None:
        return files
    bnd = bounds_for(dets)
    # 03
    n = {t: sum(1 for d in dets if d["class"] == "player" and d.get("team") == t) for t in TEAMS}
    pl = ["#PLAYERS", f"red   {n['red']}", f"blue  {n['blue']}", f"ref   {sum(d['class']=='referee' for d in dets)}",
          f"ball  {'found' if ball is not None else 'NOT detected'}", "",
          "#ATTACK DIRECTION", f"red : {dirs['red'] or 'unknown'}", f"blue: {dirs['blue'] or 'unknown'}", ""]
    if carriers:
        pl.append("#BALL CARRIER")
        for t, c in carriers.items():
            pl.append(f"{t}: {c['label']}" + ("  (ASSUMED)" if c["assumed"] else ""))
    fig, ax = bird_page("03  Bird's-eye view", "yellow ring = ball carrier", dets, bnd, pl, ball=ball)
    for t, c in carriers.items(): ax.scatter(*c["pos"], s=420, facecolors="none", edgecolors="#ffd60a", linewidths=3, zorder=8)
    save_fig(fig, os.path.join(out, "03_birdseye.png")); files["birdseye"] = "03_birdseye.png"

    def lines_for(rep, kind):
        L_ = [f"#{rep['team'].upper()}  passer {rep['passer']['label']}" + (" (ASSUMED)" if rep["passer"]["assumed"] else "")]
        if kind == "ground":
            for o in sorted(rep["ground"], key=lambda o: -o["score"]):
                d = o["min_defender_dist_to_lane_m"]
                L_.append(f"{o['receiver']:>5} {'BLOCKED' if o['blocked'] else 'open   '} lane {('%.1fm' % d) if d is not None else 'n/a':>6} s{o['score']:.0f}")
        elif kind == "lofted":
            for o in sorted(rep["lofted"], key=lambda o: -o["score"]):
                L_.append(f"{o['receiver']:>5} {o['landing_pressure']:<8} n{o['opponents_near_landing']} {o['distance_m']:.0f}m s{o['score']:.0f}" + (" short" if o["too_short"] else ""))
        elif kind == "through":
            th = rep["through"]
            if not th["evaluated"]: L_.append(f"not evaluated: {th['reason']}")
            else:
                L_.append(f"offside line x={th['offside_line_x']:.1f}")
                for o in sorted(th["options"], key=lambda o: -o["score"]):
                    L_.append(f"{o['receiver']:>5} {'OFFSIDE' if o['offside_risk'] else ('ok' if o['viable'] else 'no'):<7} m{('%+.1f' % o['race_margin_m']) if o['race_margin_m'] is not None else 'n/a'} s{o['score']:.0f}")
            if th["evaluated"] and not th["options"]: L_.append("no runner in range")
        return L_[:26]

    for t, rep in reports.items():
        if rep.get("status") != "ok": continue
        p = np.array(rep["passer"]["pos"]); sub = f"team {t}" + (" | carrier ASSUMED" if rep["passer"]["assumed"] else "")
        # 04 ground
        fig, ax = bird_page(f"04  Ground passes ({t})", sub + " | green=open red=blocked | grey = defender's perpendicular distance",
                            dets, bnd, lines_for(rep, "ground"), ball=ball)
        ax.scatter(*p, s=420, facecolors="none", edgecolors="#ffd60a", linewidths=3, zorder=8)
        for o in rep["ground"]:
            c = "#ff4d4d" if o["blocked"] else "#3ddc84"; q = np.array(o["receiver_pos"])
            ax.plot([p[0], q[0]], [p[1], q[1]], color=c, lw=2.2, ls="--" if o["blocked"] else "-", alpha=.9, zorder=3)
            if o["nearest_defender_pos"] is not None:
                ax.plot([o["nearest_defender_pos"][0], o["closest_lane_point"][0]], [o["nearest_defender_pos"][1], o["closest_lane_point"][1]],
                        color="#c9d1d9", lw=1, ls=":", zorder=2)
                m = (p + q) / 2; ax.text(m[0], m[1], f"{o['min_defender_dist_to_lane_m']}m", color="white", fontsize=7,
                                         bbox=dict(facecolor=c, alpha=.7, pad=1, edgecolor="none"), zorder=6)
        save_fig(fig, os.path.join(out, f"04_ground_passes_{t}.png"))
        # 05 lofted
        fig, ax = bird_page(f"05  Lofted passes ({t})", sub + " | ring = landing zone (green low, orange moderate, red high pressure)",
                            dets, bnd, lines_for(rep, "lofted"), ball=ball)
        ax.scatter(*p, s=420, facecolors="none", edgecolors="#ffd60a", linewidths=3, zorder=8)
        for o in rep["lofted"]:
            c = {"low": "#3ddc84", "moderate": "#ffa62b", "high": "#ff4d4d"}[o["landing_pressure"]]; q = np.array(o["receiver_pos"])
            a = arc(p, q); ax.plot(a[:, 0], a[:, 1], color=c, lw=2, alpha=.9, zorder=3)
            ax.add_patch(mpatches.Circle(q, o["contest_radius_m"], fill=True, facecolor=c, alpha=.15, edgecolor=c, lw=1.5, zorder=2))
        save_fig(fig, os.path.join(out, f"05_lofted_passes_{t}.png"))
        # 06 through
        th = rep["through"]
        fig, ax = bird_page(f"06  Through balls ({t})", sub + " | dashed vertical = offside line", dets, bnd, lines_for(rep, "through"), ball=ball)
        ax.scatter(*p, s=420, facecolors="none", edgecolors="#ffd60a", linewidths=3, zorder=8)
        if th["evaluated"]:
            ax.axvline(th["offside_line_x"], color="#ffd60a", ls="--", lw=1.6, zorder=2)
            for o in th["options"]:
                c = "#ff4d4d" if (o["offside_risk"] or not o["viable"]) else "#3ddc84"
                tp, rp = np.array(o["target_pos"]), np.array(o["runner_pos"])
                ax.annotate("", xy=tp, xytext=p, arrowprops=dict(arrowstyle="->", color=c, lw=2, alpha=.9), zorder=3)
                ax.plot([rp[0], tp[0]], [rp[1], tp[1]], color=c, ls=":", lw=1.3, zorder=3)
                ax.scatter(*tp, marker="*", s=130, c=c, edgecolors="white", zorder=6)
        else:
            ax.text(0.5, 0.5, "attack direction unknown\nthrough balls not evaluated", transform=ax.transAxes, ha="center",
                    va="center", color="white", fontsize=14, bbox=dict(facecolor="black", alpha=.7))
        save_fig(fig, os.path.join(out, f"06_through_balls_{t}.png"))
        # 08 recommendation
        rec = rep["recommendation"]
        head = ("PASS: " + f"{rec['option']['type']} to {rec['option']['receiver']} (score {rec['option']['score']:.0f})") \
            if rec["action"] == "pass" else "DRIBBLE"
        pl = [f"#{head}", ""] + [rec["reason"][i:i + 40] for i in range(0, len(rec["reason"]), 40)] + ["", "#TOP OPTIONS"]
        for o in rep["ranked_viable_options"][:5]: pl.append(f"{o['type']:<7}{o['receiver']:>5}  s{o['score']:.0f}")
        fig, ax = bird_page(f"08  Recommendation ({t})", sub, dets, bnd, pl, ball=ball)
        ax.scatter(*p, s=420, facecolors="none", edgecolors="#ffd60a", linewidths=3, zorder=8)
        if rec["action"] == "pass":
            o = rec["option"]; q = np.array(o.get("target_pos", o.get("receiver_pos")))
            if o["type"] == "lofted":
                a = arc(p, q); ax.plot(a[:, 0], a[:, 1], color="#3ddc84", lw=4, zorder=4)
                ax.annotate("", xy=a[-1], xytext=a[-3], arrowprops=dict(arrowstyle="-|>", color="#3ddc84", lw=3))
            else:
                ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#3ddc84", lw=4), zorder=4)
        else:
            q = np.array(rec["dribble"]["target_pos"])
            ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle="-|>", color="#ffa62b", lw=4), zorder=4)
        save_fig(fig, os.path.join(out, f"08_recommendation_{t}.png"))
        files[t] = {"ground_passes": f"04_ground_passes_{t}.png", "lofted_passes": f"05_lofted_passes_{t}.png",
                    "through_balls": f"06_through_balls_{t}.png", "recommendation": f"08_recommendation_{t}.png"}
    # 07 formation (each team that has a report, else each team with players)
    for t, fm in forms.items():
        pl = [f"#{t.upper()} FORMATION"]
        if fm.get("formation"):
            pl += [f"{fm['formation']}   rating {fm['rating']}/10 ({fm['rating_label']})",
                   f"closest: {fm['closest_template']}", f"outfield visible: {fm['outfield_visible']}",
                   "RELIABLE" if fm["reliable"] else "PARTIAL VIEW", f"orientation: {fm['orientation']}", "", "#COMPONENTS"]
            pl += [f"{k:<14}{v:.2f}" for k, v in fm["components"].items()] + ["", "#LINES"]
            pl += [f"L{i+1}: {l['count']}  {' '.join(l['members'])}"[:38] for i, l in enumerate(fm["lines"])]
        else:
            pl += [fm.get("note", "not enough players")]
        fig, ax = bird_page(f"07  Formation ({t})", "geometric estimate", dets, bnd, pl)
        if fm.get("formation"):
            for l in fm["lines"]:
                ax.axvline(l["center_x"], color=TEAM_COL[t], ls="--", lw=1.4, alpha=.8, zorder=2)
                ax.text(l["center_x"], bnd[3] + 1.5, str(l["count"]), color="white", fontsize=11, weight="bold", ha="center", zorder=9)
            ax.text(0.5, 0.03, f"{fm['formation']}   {fm['rating']}/10", transform=ax.transAxes, ha="center", color="white",
                    fontsize=16, weight="bold", bbox=dict(facecolor="black", alpha=.7))
        save_fig(fig, os.path.join(out, f"07_formation_{t}.png"))
        files.setdefault(t, {})["formation"] = f"07_formation_{t}.png"
    return files


# =====================================================================================================
# 7. PIPELINE
# =====================================================================================================
def analyze_core(img, dets, H, calib, overlay, out_dir, image_name="image", passer=None, attack_direction=None):
    """Everything after detection + calibration. Pure function of (image, detections, H) - easy to test."""
    warnings_ = list(calib.get("warnings", []))
    meta = {"image": image_name, "skipped": False, "skip_reason": None}
    if H is None:
        why = "; ".join(str(a.get("why")) for a in calib.get("attempts", []) if a.get("why")) or "no usable pitch keypoints"
        meta.update(skipped=True, skip_reason="calibration_failed: " + why)
        for d in dets: d.setdefault("team", "unknown"); d["label"] = d["class"]
        files = render_all(out_dir, img, dets, calib, overlay, None, {}, {}, {}, {t: None for t in TEAMS}, meta)
        return _finish(out_dir, meta, calib, dets, None, {}, {}, {}, {t: None for t in TEAMS}, {t: "unknown" for t in TEAMS}, files, warnings_)
    set_pitch_positions(dets, H)
    assign_outfield_teams(img, dets, warnings_)
    assign_keepers(dets, warnings_)
    label_detections(dets)
    teamed = {t: sum(1 for d in dets if d["class"] == "player" and d.get("team") == t and "pitch_x" in d) for t in TEAMS}
    if min(teamed.values()) < 1:
        meta.update(skipped=True, skip_reason=f"cannot_split_teams (players on pitch: {teamed})")
        files = render_all(out_dir, img, dets, calib, overlay, None, {}, {}, {}, {t: None for t in TEAMS}, meta)
        return _finish(out_dir, meta, calib, dets, None, {}, {}, {}, {t: None for t in TEAMS}, {t: "unknown" for t in TEAMS}, files, warnings_)
    dirs, dsrc = infer_attack_directions(dets, attack_direction, warnings_)
    for t in TEAMS:
        if dirs[t] is None: warnings_.append(f"{t}: attack direction unknown - through balls skipped, forward progress neutral")
    balls = sorted([d for d in dets if d["class"] == "ball" and "pitch_x" in d], key=lambda d: -d["confidence"])
    ball = balls[0] if balls else None
    if ball is None: warnings_.append("ball not detected - carrier is ASSUMED from player clustering; both teams reported")
    if passer is not None:                                   # user override -> that team's report only
        teams = [passer["team"]]
    elif ball is not None:
        near = min([d for d in dets if d.get("team") in TEAMS and d["class"] in ("player", "goalkeeper") and "pitch_x" in d],
                   key=lambda d: np.linalg.norm(P(d) - P(ball)))
        teams = [near["team"]]
    else:
        teams = list(TEAMS)
    reports, carriers = {}, {}
    for t in teams:
        reports[t] = analyze_team(t, dets, ball, dirs[t], passer, warnings_)
        if reports[t].get("status") == "ok":
            carriers[t] = {"label": reports[t]["passer"]["label"], "pos": reports[t]["passer"]["pos"],
                           "assumed": reports[t]["passer"]["assumed"]}
            if reports[t]["passer"].get("ball_distance_m", 0) and reports[t]["passer"]["ball_distance_m"] > 6:
                warnings_.append(f"nearest player is {reports[t]['passer']['ball_distance_m']} m from the ball (in the air?)")
    forms = {t: rate_formation(t, dets, dirs[t]) for t in TEAMS}
    files = render_all(out_dir, img, dets, calib, overlay, ball, carriers, reports, forms, dirs, meta)
    return _finish(out_dir, meta, calib, dets, ball, reports, forms, carriers, dirs, dsrc, files, warnings_)


def _finish(out_dir, meta, calib, dets, ball, reports, forms, carriers, dirs, dsrc, files, warnings_):
    res = {**meta, "calibration": calib,
           "ball": {"detected": ball is not None, "pitch_pos": None if ball is None else P(ball)},
           "possession": {"teams_reported": list(reports.keys()),
                          "method": "ball" if ball is not None else "unknown_both_teams_reported"},
           "attack_direction": {"red": dirs["red"], "blue": dirs["blue"], "source": dsrc},
           "detections": [{k: d.get(k) for k in ("label", "class", "team", "confidence", "bbox_px", "pitch_x", "pitch_y")} for d in dets],
           "reports": reports, "formations": forms, "files": files, "warnings": sorted(set(warnings_)),
           "limits": ["single frame: no speeds - race margins and interception are distance-based estimates",
                      "formation is a geometric estimate; only reliable with >= 8 visible outfield players"]}
    with open(os.path.join(out_dir, "analysis.json"), "w") as f:
        json.dump(res, f, indent=2, default=to_py)
    return res


def analyze_image(image_path, out_dir, checkpoint=None, passer=None, attack_direction=None):
    """passer = {"team": "red", "index": 3} (optional override).  attack_direction = {"red": "increasing_x"} (optional)."""
    os.makedirs(out_dir, exist_ok=True)
    img_pil = Image.open(image_path).convert("RGB"); img = np.array(img_pil)
    ck = checkpoint or CONFIG["checkpoint_path"]
    if not ck: raise RuntimeError("no detector checkpoint (--checkpoint or FOOTBALL_CKPT)")
    dets = run_detection(img_pil, ck)
    name = os.path.basename(image_path)
    if sum(1 for d in dets if d["class"] in PERSON) < CONFIG["min_person_count"]:
        meta = {"image": name, "skipped": True, "skip_reason": f"too_few_people (< {CONFIG['min_person_count']})"}
        for d in dets: d["team"] = "unknown"; d["label"] = d["class"]
        files = render_all(out_dir, img, dets, {"method": None}, {}, None, {}, {}, {}, {t: None for t in TEAMS}, meta)
        return _finish(out_dir, meta, {"method": None, "quality": None, "warnings": []}, dets, None, {}, {}, {},
                       {t: None for t in TEAMS}, {t: "unknown" for t in TEAMS}, files, [])
    H, calib, overlay = calibrate(image_path, img, dets)
    return analyze_core(img, dets, H, calib, overlay, out_dir, name, passer, attack_direction)


# =====================================================================================================
# 8. CLI + FASTAPI
# =====================================================================================================
def create_app(checkpoint, root="./api_results"):
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    os.makedirs(root, exist_ok=True)
    app = FastAPI(title="Football image analyzer")
    app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
                       allow_methods=["*"], allow_headers=["*"])
    app.mount("/results", StaticFiles(directory=root), name="results")

    @app.get("/health")
    def health():
        ck = checkpoint or CONFIG["checkpoint_path"]
        return {
            "ok": True,
            "status": "healthy",
            "service": "football_analyzer",
            "detector": "DINOv2 + Faster R-CNN",
            "checkpoint": ck,
            "checkpoint_exists": os.path.exists(ck) if ck else False
        }

    @app.post("/analyze")
    def analyze(file: UploadFile = File(...), passer_team: str = Form(None), passer_index: int = Form(None),
                attack_red: str = Form(None), attack_blue: str = Form(None)):
        job = uuid.uuid4().hex[:12]; folder = os.path.join(root, job); os.makedirs(folder)
        src = os.path.join(folder, "input" + os.path.splitext(file.filename or ".jpg")[1])
        with open(src, "wb") as f: shutil.copyfileobj(file.file, f)
        ov = {"team": passer_team, "index": passer_index} if passer_team else None
        ad = {k: v for k, v in (("red", attack_red), ("blue", attack_blue)) if v}
        res = analyze_image(src, folder, checkpoint, ov, ad or None)
        base = f"/results/{job}/"
        def url(x): return {k: url(v) for k, v in x.items()} if isinstance(x, dict) else (base + x if isinstance(x, str) else x)
        res["files"] = url(res["files"]); res["json_url"] = base + "analysis.json"
        return json.loads(json.dumps(res, default=to_py))
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*"); ap.add_argument("--out", default="./results")
    ap.add_argument("--checkpoint", default=CONFIG["checkpoint_path"]); ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    if a.serve:
        import uvicorn; uvicorn.run(create_app(a.checkpoint), host="0.0.0.0", port=a.port); return
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp"); paths = []
    for i in a.inputs:
        paths += [os.path.join(i, f) for f in sorted(os.listdir(i)) if f.lower().endswith(exts)] if os.path.isdir(i) else [i]
    for pth in paths:
        stem = os.path.splitext(os.path.basename(pth))[0]
        try:
            r = analyze_image(pth, os.path.join(a.out, stem), a.checkpoint)
            rec = {t: v.get("recommendation", {}).get("reason") for t, v in r["reports"].items()}
            print(f"{stem}: {'SKIPPED - ' + r['skip_reason'] if r['skipped'] else 'ok'} {rec if rec else ''}")
        except RuntimeError as e:
            print("STOPPING:", e); break
        except Exception as e:
            print(f"{stem}: error {e}")


if __name__ == "__main__":
    main()
