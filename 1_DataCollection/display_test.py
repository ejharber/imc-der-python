#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageDisplayNode(Node):
    def __init__(self):
        super().__init__('image_display_node')

        # Create a CvBridge instance
        self.bridge = CvBridge()

        # Create a subscription to the image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/raw_image',  # Change this to your image topic
            self.image_callback,
            10
        )
        
        # Create a window to display the image
        cv2.namedWindow("Image", cv2.WINDOW_AUTOSIZE)

    def image_callback(self, msg):
        try:
            # Convert the ROS Image message to a CV2 image
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Display the image
            cv2.imshow("Image", img)
            cv2.waitKey(1)  # Wait for 1 ms to display the image
        except Exception as e:
            self.get_logger().error(f"Error converting image: {str(e)}")

    def destroy_node(self):
        # Destroy the OpenCV window when shutting down
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)

    image_display_node = ImageDisplayNode()

    try:
        rclpy.spin(image_display_node)
    except KeyboardInterrupt:
        pass
    finally:
        image_display_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
