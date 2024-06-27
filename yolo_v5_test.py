import sys
import os
sys.path.insert(0, os.path.abspath('D:/Neil/applepick_v2/yolov5'))  # Use forward slashes

import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.torch_utils import select_device
import supervision as sv
from realsense_camera import RealsenseCamera

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv5 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def get_depth_information(detections, depth_frame_in):
    depth_frame = np.asanyarray(depth_frame_in.get_data())
    depth_info = []

    for det in detections:
        bbox = det[:4]  # Bounding box coordinates
        try:
            x_min, y_min, x_max, y_max = map(int, bbox)  # Unpack bounding box and convert to int
        except Exception as e:
            print(f"Error processing bounding box {bbox}: {e}")
            continue  # Skip this detection if there is an error with converting

        depth_values = depth_frame[y_min:y_max, x_min:x_max]
        avg_depth = np.mean(depth_values) if depth_values.size > 0 else 0  # Ensure depth_values isn't empty before averaging
        rounded_depth = round(avg_depth, 2)
        depth_info.append(rounded_depth)

    return depth_info

def get_real_world_coordinates(detections, depth_info, intrinsics):
    object_coordinates = []

    for bbox, depth in zip(detections, depth_info):
        x_center = (bbox[2] + bbox[0]) // 2
        y_center = (bbox[3] + bbox[1]) // 2

        depth_meters = depth * 0.001

        X = depth_meters * (x_center - intrinsics.ppx) / intrinsics.fx
        Y = depth_meters * (y_center - intrinsics.ppy) / intrinsics.fy

        object_coordinates.append((X, Y))

    return object_coordinates

def main():
    args = parse_arguments()
    realsense_cam = RealsenseCamera()  # Replace this with your camera initialization

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.webcam_resolution[0], args.webcam_resolution[1], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, args.webcam_resolution[0], args.webcam_resolution[1], rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)

    device = select_device('')
    model = attempt_load('yolov5_weights.pt', device)
    model.to(device).eval()
    names = model.module.names if hasattr(model, 'module') else model.names
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=1, text_scale=0.5)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        bgr_frame = np.asanyarray(color_frame.get_data())
        img = cv2.resize(bgr_frame, (640, 640))
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)
        detections = pred[0].cpu().numpy()

        if detections.shape[0] > 0:  # Ensure there are detections
            xyxy = np.array([det[:4] for det in detections])
            confidence = np.array([det[4] for det in detections])
            class_id = np.array([int(det[5]) for det in detections])
            detections_formatted = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
        else:
            xyxy = np.empty((0, 4), dtype=np.float32)
            confidence = np.empty((0,), dtype=np.float32)
            class_id = np.empty((0,), dtype=np.int32)
            detections_formatted = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)

        labels = [f"{names[int(cls)]} {conf:.2f}" for *xyxy, conf, cls in detections]

        depth_info = get_depth_information(detections, depth_frame)
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        object_coordinates = get_real_world_coordinates(detections, depth_info, intrinsics)

        annotated_frame = box_annotator.annotate(
            scene=bgr_frame,
            detections=detections_formatted,  # Use formatted detections
            labels=[
                f"{label} - Depth: {depth_info[i] if i < len(depth_info) else 'N/A'}, Coordinates: {f'({x_coord:.3f}, {y_coord:.3f}) meters' if x_coord != 'N/A' and y_coord != 'N/A' else 'N/A'}"
                for i, (label, (x_coord, y_coord)) in enumerate(zip(labels, object_coordinates))
            ]
        )

        for label, depth, (x_coord, y_coord) in zip(labels, depth_info, object_coordinates):
            print(f"Object: "potted plant", Depth: {depth} mm, Coordinates: ({x_coord:.2f}, {y_coord:.2f}) meters")

        cv2.imshow("yolov5", annotated_frame)

        if cv2.waitKey(30) == 27:  # Escape key
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
