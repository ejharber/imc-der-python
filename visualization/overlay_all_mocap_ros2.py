import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from mocap4r2_msgs.msg import Markers
from mocap4r2_msgs.msg import RigidBodies

from cv_bridge import CvBridge
import cv2
import numpy as np 
from geometry_msgs.msg import PoseStamped

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(Image, '/camera/raw_image', self.listener_callback, 1)
        self.subscription  # prevent unused variable warning
        self.bridge = CvBridge()

        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 1)
        self.mocap_subscription

        self.mocap_history = []
        self.mocap_delay = 14

        self.calibration = np.load("calibration_data.npz")

    def mocap_callback(self, msg):
        print('new data')

        markers = []
        for rigidbody in msg.rigidbodies:
            for marker in rigidbody.markers:
                markers.append(marker)
        self.mocap_history.append(markers)
        if len(self.mocap_history) > self.mocap_delay:
            self.mocap_history = self.mocap_history[1:]

    def listener_callback(self, msg):

        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if len(self.mocap_history) > 0:
                mocap_data = []
                for marker in self.mocap_history[0]:
                    mocap_data.append([marker.translation.x, marker.translation.y, marker.translation.z])
                mocap_data = np.array(mocap_data)
                # print(mocap_data)

                imgpoints, _ = cv2.projectPoints(mocap_data, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
                # imgpoints = imgpoints 
                for i in range(imgpoints.shape[0]):
                    # print((int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])))
                    # print(img.shape)
                    if imgpoints[i, 0, 0] < 0 or imgpoints[i, 0, 1] < 0: continue 
                    if imgpoints[i, 0, 0] > img.shape[1] or imgpoints[i, 0, 1] > img.shape[0]: continue 
                    img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 3, (0,0,255), -1)

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
