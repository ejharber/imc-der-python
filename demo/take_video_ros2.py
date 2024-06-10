import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import subprocess
from display_video import set_logitech_camera_settings

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/raw_image', 1)
        self.bridge = CvBridge()
        # Replace '0' with the index of your camera, or provide the file path if reading from a file
        camera = 0
        set_logitech_camera_settings(0)
        self.cap = cv2.VideoCapture(camera)
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
