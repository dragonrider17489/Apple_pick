import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import supervision as sv
from realsense_camera import RealsenseCamera  # Assuming the corrected name for the camera class

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv10 live")
    parser.add_argument("--webcam-resolution", default=[1280, 720], nargs=2, type=int)
    args = parser.parse_args()
    return args

def get_depth_information(detections, depth_frame_in, realsense_cam):
    depth_frame = np.asanyarray(depth_frame_in.get_data())
    depth_info = []
    xyxy = detections.xyxy

    for point in xyxy:
        x_min, y_min, x_max, y_max = map(int, point)  # Convert coordinates to integers
        # Calculate the average depth within the bounding box region
        depth_values = depth_frame[y_min:y_max, x_min:x_max]
        avg_depth = np.mean(depth_values) if len(depth_values) > 0 else 0
        rounded_depth = round(avg_depth, 2)  # Round to two decimal places
        depth_info.append(rounded_depth)      

    return depth_info



def get_real_world_coordinates(detections, depth_info, depth_frame, intrinsics):
    object_coordinates = [] 

    for point, depth in zip(detections.xyxy, depth_info):
        x_center = (point[2] + point[0]) // 2  # Calculating x-coordinate of the center
        y_center = (point[3] + point[1]) // 2  # Calculating y-coordinate of the center

        depth_meters = depth * 0.001

        # Compute real-world (x, y) position using intrinsics
        X = depth_meters * (x_center - intrinsics.ppx) / intrinsics.fx
        Y = depth_meters * (y_center - intrinsics.ppy) / intrinsics.fy

        object_coordinates.append((X, Y))

    return object_coordinates


def main():
    realsense_cam = RealsenseCamera()  # Replace this with your camera initialization
    args = parse_arguments()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.webcam_resolution[0], args.webcam_resolution[1], rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, args.webcam_resolution[0], args.webcam_resolution[1], rs.format.z16, 30)
    pipeline.start(config)

    align = rs.align(rs.stream.color)  # Initialize the alignment object

    model = YOLO("yolov10_weights.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    detected_objects = []  

    while True:
        frames = pipeline.wait_for_frames()
        
        aligned_frames = align.process(frames)  # Align the frames
        
        color_frame = aligned_frames.first(rs.stream.color)
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        bgr_frame = np.asanyarray(color_frame.get_data())

        result = model(bgr_frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[int(class_id)]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
  
        depth_info = get_depth_information(detections, depth_frame, realsense_cam)
        intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        object_coordinates = get_real_world_coordinates(detections, depth_info, depth_frame, intrinsics)

        annotated_frame = box_annotator.annotate(
            scene=bgr_frame,
            detections=detections,
            labels=[
                f"{label} - Depth: {depth_info[i] if i < len(depth_info) else 'N/A'}, Coordinates: {f'({x_coord:.3f}, {y_coord:.3f}) meters' if x_coord != 'N/A' and y_coord != 'N/A' else 'N/A'}"
                for i, ((_, confidence, class_id, _), label, (x_coord, y_coord)) in enumerate(zip(detections, labels, object_coordinates))
            ]
        )

        # Display object names and their depths in real-time
        for label, depth, (x_coord, y_coord) in zip(labels, depth_info, object_coordinates):
            label='potted plant'
            print(f"Object: {label}, Depth: {depth} mm, Coordinates: ({x_coord:.2f}, {y_coord:.2f}) meters")

        cv2.imshow("yolov8", annotated_frame)

        if (cv2.waitKey(30) == 27):
            break

    pipeline.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
