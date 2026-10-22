"""Weak-perspective cylinder/hand alignment from RGB and fresh hand predictions.

This is explicitly NOT calibrated six-DoF ground truth. The can's visible white
lid supplies a scale/axis estimate; fingertip surface proximity resolves one
otherwise unobservable hand-to-can depth translation at the grasp anchor.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from extract_raw_video_hand import digest


def cylinder_distance(points, center, axis, radius, half_height):
    delta = points - center
    axial = delta @ axis
    radial = np.linalg.norm(delta - axial[..., None]*axis, axis=-1)
    d = np.stack([radial-radius, np.abs(axial)-half_height], axis=-1)
    return np.minimum(d.max(-1), 0) + np.linalg.norm(np.maximum(d, 0), axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--hand", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diameter", type=float, default=.069644)
    parser.add_argument("--height", type=float, default=.103637)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    hand = dict(np.load(args.hand, allow_pickle=False))
    if not hand["valid"].all():
        raise ValueError("Missing hand frames require an explicit quality decision")
    cap = cv2.VideoCapture(str(args.video))
    width, height = map(int, hand["image_size"])
    writer = cv2.VideoWriter(str(args.output / "overlay.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), float(hand["fps"]), (width, height))
    if not writer.isOpened():
        raise RuntimeError("Cannot open video encoder")
    previous = np.array([426., 258.])  # One manually identified object ROI, no motion labels.
    ellipses, centers, axes, joints, scales = [], [], [], [], []
    try:
        for i, xyz in enumerate(hand["image_landmarks"]):
            ok, image = cap.read()
            if not ok:
                raise ValueError("Video/hand frame count differs")
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Reject bright skin touching the lid; its saturation is much higher.
            mask = cv2.inRange(hsv, (0, 0, 220), (180, 40, 255))
            yy, xx = np.indices(mask.shape)
            mask[(abs(xx-previous[0]) > 55) | (abs(yy-previous[1]) > 55)] = 0
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            candidates = []
            for contour in contours:
                if len(contour) < 5 or not 300 < cv2.contourArea(contour) < 3500:
                    continue
                ellipse = cv2.fitEllipse(contour)
                point, size, angle = ellipse
                if min(size)/max(size) < .35 or not 25 < max(size) < 95:
                    continue
                filled_ratio = cv2.contourArea(contour)/(np.pi*size[0]*size[1]/4)
                if not .8 < filled_ratio < 1.2:
                    continue
                candidates.append((np.linalg.norm(np.asarray(point)-previous), ellipse))
            if not candidates:
                raise ValueError(f"No valid lid ellipse at frame {i}; do not fabricate object pose")
            ellipse = min(candidates, key=lambda pair: pair[0])[1]
            point, size, angle = ellipse
            previous = np.asarray(point)
            minor, major = sorted(size)
            theta = np.deg2rad(angle if size[0] <= size[1] else angle+90)
            normal_xy = np.array([np.cos(theta), np.sin(theta)])
            # The visible top faces the camera and projects upwards in this clip.
            if normal_xy[1] > 0:
                normal_xy *= -1
            ratio = minor/major
            axis = np.r_[normal_xy*np.sqrt(1-ratio**2), -ratio]
            scale = major/args.diameter
            center = -.5*args.height*axis
            pixels = xyz[:, :2]*[width, height]
            world = hand["world_landmarks"][i]
            pixel_relative = pixels-pixels[0]
            world_relative = world-world[0]
            hand_scale = np.linalg.norm(pixel_relative)/max(np.linalg.norm(world_relative[:, :2]), 1e-8)
            points = np.column_stack([(pixels-previous)/scale, world_relative[:, 2]*hand_scale/scale])
            ellipses.append([*point, *size, angle])
            centers.append(center)
            axes.append(axis)
            joints.append(points)
            scales.append(scale)
            cv2.ellipse(image, ellipse, (0, 255, 255), 2)
            cv2.putText(image, f"RGB lid fit {i}: scale {scale:.0f} px/m (approx.)",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
            writer.write(image)
    finally:
        cap.release()
        writer.release()
    joints, centers, axes = map(np.asarray, (joints, centers, axes))
    # Select by visible event timing, BEFORE any simulator execution. No success search.
    anchor = 60
    contact_frames = np.arange(52, 76)
    tips = np.array([4, 8, 12, 16, 20])
    def depth_error(z):
        return np.concatenate([cylinder_distance(joints[i, tips]+[0, 0, z[0]],
                               centers[i], axes[i], args.diameter/2, args.height/2)
                               for i in contact_frames])
    fits = [least_squares(depth_error, [z], bounds=([-.2], [.2]),
                          loss="soft_l1", f_scale=.005) for z in (-.08, .08)]
    fit = min(fits, key=lambda item: item.cost)
    surface_error = np.abs(depth_error(fit.x))
    joints[:, :, 2] += fit.x[0]
    np.savez_compressed(args.output / "hand_object.npz", hand_joints=joints,
                        object_centers=centers, object_axes=axes, lid_ellipses=ellipses,
                        pixel_scale=scales, time=hand["timestamps"], anchor_frame=anchor,
                        selected_frames=np.arange(30, 61), diameter=args.diameter,
                        height=args.height, handedness=hand["handedness"])
    manifest = dict(video_sha256=digest(args.video), hand_sha256=digest(args.hand),
                    source_sha256=digest(__file__), output_sha256=digest(args.output / "hand_object.npz"),
                    frames=len(joints), grasp_anchor_frame=anchor, retarget_frames=[30, 60],
                    manually_identified_first_lid_center_px=[426, 258],
                    diameter_m=args.diameter, height_m=args.height,
                    dimensions_source="Bounding extents of DexMV YCB 005_tomato_soup_can collision STL; no rescaling",
                    depth_offset_m=float(fit.x[0]),
                    median_tip_surface_error_m=float(np.median(surface_error)),
                    p95_tip_surface_error_m=float(np.quantile(surface_error, .95)),
                    limitations=["Single-view weak perspective, unknown camera calibration, no measured depth",
                                 "White lid interior is an approximate outer-radius proxy",
                                 "Rotation around the symmetry axis is unobservable and unnecessary for an analytic cylinder",
                                 "Common hand depth fitted using assumed fingertip contact; not independent ground truth",
                                 "Fixed-wrist task removes human arm translation and freezes object placement at grasp anchor",
                                 "No robot reward or success score used in any fit"])
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
