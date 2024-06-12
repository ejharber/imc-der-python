import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from mocap4r2_msgs.msg import Markers
from mocap4r2_msgs.msg import RigidBodies

from cv_bridge import CvBridge
import cv2
import numpy as np 

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from geometry_msgs.msg import PoseStamped

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(Image, '/camera/raw_image', self.listener_callback, 1)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        self.mocap_subscription = self.create_subscription(PoseStamped, '/camera/goal', self.goal_callback, 1)
        self.mocap_goal = None

        self.mocap_subscription = self.create_subscription(PoseStamped, '/camera/actual', self.actual_callback, 1)
        self.mocap_actual = None

        self.calibration = np.load("calibration_data.npz")

    def goal_callback(self, msg):
        self.mocap_goal = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def actual_callback(self, msg):
        self.mocap_actual = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])

    def listener_callback(self, msg):

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if self.mocap_goal is not None:
                
                imgpoints, _ = cv2.projectPoints(self.mocap_goal, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
                for i in range(imgpoints.shape[0]):
                    # print((int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])))
                    # print(img.shape)
                    if imgpoints[i, 0, 0] < 0 or imgpoints[i, 0, 1] < 0: continue 
                    if imgpoints[i, 0, 0] > img.shape[0] or imgpoints[i, 0, 1] > img.shape[1]: continue 
                    img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 20, (0,0,255), -1)

            if self.mocap_actual is not None:
                
                imgpoints, _ = cv2.projectPoints(self.mocap_actual, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
                for i in range(imgpoints.shape[0]):
                    # print((int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])))
                    # print(img.shape)
                    if imgpoints[i, 0, 0] < 0 or imgpoints[i, 0, 1] < 0: continue 
                    if imgpoints[i, 0, 0] > img.shape[0] or imgpoints[i, 0, 1] > img.shape[1]: continue 
                    img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 20, (255,0,0), -1)

            cv2.imshow("Image", img)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error("Error converting image: %s" % str(e))

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    try:
        rclpy.spin(image_subscriber)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
