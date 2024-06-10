import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from mocap4r2_msgs.msg import Markers
from mocap4r2_msgs.msg import RigidBodies

from cv_bridge import CvBridge
import cv2
import numpy as np 

class CollectData(Node):

    def __init__(self):
        super().__init__('collect_data')
        self.bridge = CvBridge()

        self.real_sense_subscription = self.create_subscription(Image, '/camera/raw_image', self.real_sense_callback, 1)
        self.real_sense_subscription  # prevent unused variable warning
        self.real_sense_last_msg = None

        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 1)
        self.mocap_subscription

        self.mocap_history = []
        self.mocap_delay = 4

        self.publisher = self.create_publisher(Image, '/mocapRGB', 1)
        self.calibration = np.load("calibration_data.npz")
        timer_period = 0.01  # run as fast as possible (its a camera so max 100 hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):

        if self.real_sense_last_msg is None:
            return 

        img = self.bridge.imgmsg_to_cv2(self.real_sense_last_msg, desired_encoding='bgr8')

        if len(self.mocap_history) > 0:
            mocap_data = []
            for marker in self.mocap_history[0]:
                mocap_data.append([marker.translation.x, marker.translation.y, marker.translation.z])
            mocap_data = np.array(mocap_data)
            # print(mocap_data)

            imgpoints, _ = cv2.projectPoints(mocap_data, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
            for i in range(imgpoints.shape[0]):
                # print((int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])))
                # print(img.shape)
                if imgpoints[i, 0, 0] < 0 or imgpoints[i, 0, 1] < 0: continue 
                if imgpoints[i, 0, 0] > img.shape[0] or imgpoints[i, 0, 1] > img.shape[1]: continue 
                img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 3, (0,0,255), -1)

        msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")

        self.publisher.publish(msg)

        self.real_sense_last_msg = None

    def real_sense_callback(self, msg):
        self.real_sense_last_msg = msg 

    def mocap_callback(self, msg):
        print('new data')
        self.mocap_history.append(msg.markers)
        if len(self.mocap_history) > self.mocap_delay:
            self.mocap_history = self.mocap_history[1:]

def main(args=None):
    rclpy.init(args=args)

    collect_data = CollectData()

    rclpy.spin(collect_data)

    collect_data.destroy_node() # this line is optional 
    rclpy.shutdown()


if __name__ == '__main__':
    main()