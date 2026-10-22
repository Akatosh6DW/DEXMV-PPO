"""Run fresh RGB hand inference; retain missing frames instead of inventing poses.

Run in the isolated vision environment, not aero_rl. This stage deliberately
does not import any simulator, old reference, or task policy checkpoint.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
               (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17),
               (0, 17), (17, 18), (18, 19), (19, 20)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(args.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = (int(capture.get(p)) for p in
                     (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    if not capture.isOpened() or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Invalid video")
    writer = cv2.VideoWriter(str(args.output / "overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open overlay video encoder")
    vision = mp.tasks.vision
    options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
        min_tracking_confidence=.5)
    normalized, world, handedness, handedness_score, valid = [], [], [], [], []
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                index = len(valid)
                result = detector.detect_for_video(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), round(index*1000/fps))
                found = len(result.hand_landmarks) == 1
                valid.append(found)
                xyz = np.full((21, 3), np.nan, dtype=np.float32)
                metric = xyz.copy()
                label, score = "missing", 0.
                if found:
                    xyz = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
                    metric = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], dtype=np.float32)
                    category = result.handedness[0][0]
                    label, score = category.category_name, category.score
                    points = np.rint(xyz[:, :2] * [width, height]).astype(int)
                    for a, b in CONNECTIONS:
                        cv2.line(frame, tuple(points[a]), tuple(points[b]), (0, 220, 0), 1)
                    for point in points:
                        cv2.circle(frame, tuple(point), 2, (0, 0, 255), -1)
                cv2.putText(frame, f"frame {index} / {label} (inference, not GT)",
                            (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                writer.write(frame)
                normalized.append(xyz)
                world.append(metric)
                handedness.append(label)
                handedness_score.append(score)
    finally:
        capture.release()
        writer.release()
    if not valid:
        raise ValueError("No decoded frames")
    np.savez_compressed(args.output / "hand_landmarks.npz",
                        image_landmarks=normalized, world_landmarks=world,
                        valid=valid, handedness=handedness,
                        handedness_score=handedness_score,
                        timestamps=np.arange(len(valid))/fps, fps=fps,
                        image_size=[width, height])
    manifest = dict(source_video=str(args.video.resolve()),
                    source_video_sha256=digest(args.video),
                    perception_model_sha256=digest(args.model),
                    script_sha256=digest(__file__),
                    output_sha256=digest(args.output / "hand_landmarks.npz"),
                    versions={k: importlib.metadata.version(k) for k in
                              ("mediapipe", "opencv-contrib-python", "numpy")},
                    frames=len(valid), detected_frames=int(sum(valid)),
                    detected_fraction=float(np.mean(valid)), fps=fps,
                    missing_indices=np.flatnonzero(~np.asarray(valid)).tolist(),
                    source_type="fresh RGB inference; no published motion fits used",
                    limitations=["World landmarks are hand-centered monocular estimates, not calibrated camera coordinates.",
                                 "Handedness score is not joint accuracy or tracking confidence.",
                                 "Occluded joints are predictions, not measurements.",
                                 "No object pose is produced at this stage."])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
