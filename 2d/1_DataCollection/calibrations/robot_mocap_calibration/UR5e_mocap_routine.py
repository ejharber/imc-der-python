import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

import rtde_control
import numpy as np

from mocap4r2_msgs.msg import RigidBodies
import time 

class UR5EMocapCalibration(Node):

    def __init__(self):
        super().__init__('kinova_mocap_calibration')

        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 1)
        self.mocap_subscription  # prevent unused variable warning
        self.mocap_last_msg = None

        self.tool_pose_subscription = self.create_subscription(PoseStamped, '/ur5e/feedback/end_effector', self.tool_pose_callback, 1)
        self.tool_pose_subscription  # prevent unused variable warning
        self.tool_pose_last_msg = None
        self.current_pose = None

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.10.60")
        self.home_joint_pose = [180, -110, -140, -30, 90, 0]

        self.mocap_base_data = []
        self.mocap_ee_data = []
        self.ur5e_ee_data = []

        self.calibration_routine()

        np.savez("calibration_data", mocap_base_data = self.mocap_base_data, mocap_ee_data = self.mocap_ee_data, ur5e_ee_data = self.ur5e_ee_data)

    def take_data(self):
        self.mocap_last_msg = None
        self.tool_pose_last_msg = None

        for _ in range(10):
            rclpy.spin_once(self)

        if self.mocap_last_msg is None:
            print('data incomplete: no mocap data')
            return 

        if self.tool_pose_last_msg is None:
            print('data incomplete: no tool data data')

        if not len(self.mocap_last_msg.rigidbodies) == 2:
            print('data incomplete: wrong number of rigid bodies ' + str(len(self.mocap_last_msg.rigidbodies)))
            return 

        print(self.mocap_last_msg.rigidbodies[0].rigid_body_name)
        print(self.mocap_last_msg.rigidbodies[1].rigid_body_name)

        pose_rb_1 = [self.mocap_last_msg.rigidbodies[0].pose.position.x,
                     self.mocap_last_msg.rigidbodies[0].pose.position.y,
                     self.mocap_last_msg.rigidbodies[0].pose.position.z,
                     self.mocap_last_msg.rigidbodies[0].pose.orientation.w,
                     self.mocap_last_msg.rigidbodies[0].pose.orientation.x,
                     self.mocap_last_msg.rigidbodies[0].pose.orientation.y,
                     self.mocap_last_msg.rigidbodies[0].pose.orientation.z]
        pose_rb_2 = [self.mocap_last_msg.rigidbodies[1].pose.position.x,
                     self.mocap_last_msg.rigidbodies[1].pose.position.y,
                     self.mocap_last_msg.rigidbodies[1].pose.position.z,
                     self.mocap_last_msg.rigidbodies[1].pose.orientation.w,
                     self.mocap_last_msg.rigidbodies[1].pose.orientation.x,
                     self.mocap_last_msg.rigidbodies[1].pose.orientation.y,
                     self.mocap_last_msg.rigidbodies[1].pose.orientation.z]

        if np.any(np.array(pose_rb_1) == 0) or np.any(np.array(pose_rb_2) == 0):
            print('data incomplete')
            return 

        self.mocap_base_data.append(pose_rb_1)

        self.mocap_ee_data.append(pose_rb_2)

        self.ur5e_ee_data.append([self.tool_pose_last_msg.pose.position.x,
                                    self.tool_pose_last_msg.pose.position.y,
                                    self.tool_pose_last_msg.pose.position.z,
                                    self.tool_pose_last_msg.pose.orientation.w,
                                    self.tool_pose_last_msg.pose.orientation.x,
                                    self.tool_pose_last_msg.pose.orientation.y,
                                    self.tool_pose_last_msg.pose.orientation.z])

        self.mocap_last_msg = None
        self.tool_pose_last_msg = None

        print(len(self.mocap_base_data))

    def calibration_routine(self):
        self.go_to_home()
        
        for q1 in np.linspace(-110, -60, 5):
            for q2 in np.linspace(-140, -80, 5):
                for q3 in np.linspace(-30, -30 + 90, 5):

                    time.sleep(1)

                    q = [180, q1 ,q2, q3, 90, 0]
                    q = np.array(q) * np.pi / 180.0
                    self.rtde_c.moveJ(q, 1, 1)

                    time.sleep(1)
                    for _ in range(10):
                        self.take_data()

        self.go_to_home()

    def tool_pose_callback(self, msg):
        self.tool_pose_last_msg = msg
        last_pose = self.current_pose
        self.current_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z])

    def mocap_callback(self, msg):
        self.mocap_last_msg = msg

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, 0.5, 0.5)

def main(args=None):
    rclpy.init(args=args)

    ur5e_mocap_calibration = UR5EMocapCalibration()

    ur5e_mocap_calibration.destroy_node() # this line is optional 
    rclpy.shutdown()


if __name__ == '__main__':
    main()