import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import subprocess

import subprocess

def set_logitech_camera_settings(camera, brightness=100, contrast=100, saturation=100, sharpness=100, exposure=0):
    # Set brightness
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c brightness=150", shell=True)  # Changed device path to /dev/video4

    # Set contrast
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c contrast=120", shell=True)

    # Set saturation
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c saturation={saturation}", shell=True)

    # Set sharpness
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c sharpness={sharpness}", shell=True)

    # Set exposure
    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c white_balance_automatic=0", shell=True)  # Set exposure_auto_priority to 0 for manual exposure

    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c zoom_absolute=140", shell=True)  # Set exposure_auto_priority to 0 for manual exposure

    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c auto_exposure=1 -c exposure_time_absolute=80", shell=True)
    # subprocess.run(f"v4l2-ctl -d /dev/video4 -c auto_exposure=0 -c exposure_time_absolute=100", shell=True)

    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c focus_automatic_continuous=0", shell=True)

    subprocess.run(f"v4l2-ctl -d /dev/video{camera} -c exposure_dynamic_framerate=1", shell=True)

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/raw_image', 1)
        self.bridge = CvBridge()
        camera = 0
        set_logitech_camera_settings(camera)
        self.cap = cv2.VideoCapture(camera)

        # Retrieve current resolution
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f'Original camera resolution: {int(width)}x{int(height)}')

        # Set doubled resolution
        doubled_width = int(640 * 8)
        doubled_height = int(480 * 8)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, doubled_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, doubled_height)

        # Confirm the new resolution
        new_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        new_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f'Doubled camera resolution: {int(new_width)}x{int(new_height)}')

        self.publish_images()

    def publish_images(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                self.publisher_.publish(msg)
            else:
                self.get_logger().error('Failed to read frame from camera!')
                continue

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    try:
        rclpy.spin(image_publisher)
    except KeyboardInterrupt:
        pass
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
