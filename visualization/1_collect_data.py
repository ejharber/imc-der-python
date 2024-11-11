import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from mocap4r2_msgs.msg import RigidBodies

from cv_bridge import CvBridge
import cv2
import numpy as np 

class CollectData(Node):

    def __init__(self):
        super().__init__('collect_data')
        self.bridge = CvBridge()

        self.camera_subscription = self.create_subscription(Image, '/camera/raw_image', self.camera_callback, 1)
        self.camera_subscription  # prevent unused variable warning
        self.camera_last_msg = None

        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 1)
        self.mocap_subscription
        self.mocap_last_msg = None

        self.trigger_subscription = self.create_subscription(Empty, '/trigger', self.trigger_callback, 1)
        self.count = 0

    def camera_callback(self, msg):
        self.camera_last_msg = msg 

    def mocap_callback(self, msg):
        for rb in msg.rigidbodies:
            # print(rb.rigid_body_name)
            if 'calib_board.calib_board' == rb.rigid_body_name:
                self.mocap_last_msg = rb

    def trigger_callback(self, msg):

        # ros2 launch mocap_vicon_driver mocap_vicon_driver_launch.py 
        # ros2 launch realsense2_camera rs_launch.py

        # ros2 topic pub --rate 1 /trigger std_msgs/msg/Empty
        # ros2 topic pub --once /trigger std_msgs/msg/Empty

        markers = np.zeros((3, 3))
        cv_image = None

        if self.camera_last_msg is None:
            print("NO NEW REAL SENSE DATA")
        else:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_last_msg, desired_encoding='rgb8')
            cv_image = cv_image[:, :, [2, 1, 0]]
            self.camera_last_msg = None

        if self.mocap_last_msg is None:
            print("NO NEW MOCAP DATA")
        else:
            for i, marker in enumerate(self.mocap_last_msg.markers):
                print(marker.marker_name, marker.translation)
                if marker.translation.x == 0 or marker.translation.y == 0 or marker.translation.z == 0:
                    break
                if marker.marker_name == "calib_board1":
                    markers[0, 0] = marker.translation.x
                    markers[0, 1] = marker.translation.y
                    markers[0, 2] = marker.translation.z

                if marker.marker_name == "calib_board2":
                    markers[1, 0] = marker.translation.x
                    markers[1, 1] = marker.translation.y
                    markers[1, 2] = marker.translation.z

                if marker.marker_name == "calib_board3":
                    markers[2, 0] = marker.translation.x
                    markers[2, 1] = marker.translation.y
                    markers[2, 2] = marker.translation.z

            print(markers)

            self.mocap_last_msg = None

            if np.any(markers == 0):             
                print("NO or INCOMPLETE MOCAP DATA")

        if not np.any(markers == 0) and cv_image is not None:
            cv2.imwrite('raw_data/' + str(self.count) + '.png', cv_image)
            np.save('raw_data/' + str(self.count), markers)
 
            self.count += 1

def main(args=None):
    rclpy.init(args=args)

    collect_data = CollectData()

    rclpy.spin(collect_data)

    collect_data.destroy_node() # this line is optional 
    rclpy.shutdown()


if __name__ == '__main__':
    main()