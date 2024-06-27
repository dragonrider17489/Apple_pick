import pyrealsense2 as rs
import numpy as np
import cv2

def main():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    try:
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # Display the color image
            cv2.namedWindow('RealSense Color', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Color', color_image)

            # Display the depth colormap image
            cv2.namedWindow('RealSense Depth', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense Depth', depth_colormap)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit the program
                break
            elif key == ord('c'):  # Take a photo when the 'c' key is pressed
                # Save the color and depth images
                cv2.imwrite('color_image.png', color_image)
                cv2.imwrite('depth_image.png', depth_colormap)
                print('Images saved.')

    finally:
        # Stop streaming
        pipeline.stop()

if __name__ == "__main__":
    main()
