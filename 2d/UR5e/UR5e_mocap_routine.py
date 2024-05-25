import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState

import rtde_control, rtde_receive
import numpy as np

from mocap4r2_msgs.msg import RigidBodies
import time 

class UR5EMocapCalibration(Node):

    def __init__(self):
        super().__init__('kinova_mocap_calibration')

        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 1)
        self.mocap_subscription  # prevent unused variable warning
        self.mocap_data = None

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

        self.home_joint_pose = [0, -54, 134, -167, -90, 0]
        self.home_cart_pose = None

        self.mocap_base_data = []
        self.mocap_ee_data = []
        self.ur5e_ee_data = []
        self.ur5e_joint_data = []

        self.calibration_routine()

        np.savez("calibration_data", ur5e_joint_data = self.ur5e_joint_data, mocap_base_data = self.mocap_base_data, mocap_ee_data = self.mocap_ee_data, ur5e_ee_data = self.ur5e_ee_data)

    def take_data(self):
        self.mocap_last_msg = None
        self.tool_pose_last_msg = None

        for _ in range(10):
            rclpy.spin_once(self)

        if self.mocap_data is None:
            print('data incomplete: no mocap data')
            return 

        if np.any(self.mocap_data == 0):
            print('data incomplete: wrong number of rigid bodies ')
            # return 

        pose_rb_1 = self.mocap_data[:, 0]
        pose_rb_2 = self.mocap_data[:, 1]

        self.mocap_base_data.append(pose_rb_1)

        self.mocap_ee_data.append(pose_rb_2)

        self.ur5e_ee_data.append(self.rtde_r.getActualTCPPose())

        self.ur5e_joint_data.append(self.rtde_r.getActualQ())

        self.mocap_data = None

    def calibration_routine(self):
        self.go_to_home()
        
        N = 2

        for dq0 in np.linspace(-20, 20, N):
            for dq1 in np.linspace(0, -40, N):
                for dq2 in np.linspace(0, -40, N):
                    for dq3 in np.linspace(-20, 20, N):
                        for dq4 in np.linspace(-20, 40, N):

                            time.sleep(1)

                            q = [dq0 + self.home_joint_pose[0], dq1 + self.home_joint_pose[1], dq2 + self.home_joint_pose[2], 
                                 dq3 + self.home_joint_pose[3], dq4 + self.home_joint_pose[4], 0]
                            q = np.array(q) * np.pi / 180.0
                            self.rtde_c.moveJ(q, 1, 1)

                            time.sleep(1)
                            for _ in range(10):
                                self.take_data()
                                time.sleep(0.01)

        self.go_to_home()

    def mocap_callback(self, msg):
        self.mocap_data = np.zeros((7, 2))
        for rigid_body in msg.rigidbodies:

            i = None

            if rigid_body.rigid_body_name == "UR_base.UR_base":
                i = 0

            if rigid_body.rigid_body_name == "rope_base.rope_base":
                i = 1

            if i is None: continue

            self.mocap_data[0, i] = rigid_body.pose.position.x    
            self.mocap_data[1, i] = rigid_body.pose.position.y    
            self.mocap_data[2, i] = rigid_body.pose.position.z    

            self.mocap_data[3, i] = rigid_body.pose.orientation.w    
            self.mocap_data[4, i] = rigid_body.pose.orientation.x    
            self.mocap_data[5, i] = rigid_body.pose.orientation.y  
            self.mocap_data[6, i] = rigid_body.pose.orientation.z

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q)
        self.home_cart_pose = self.rtde_r.getActualTCPPose()

def main(args=None):
    rclpy.init(args=args)

    ur5e_mocap_calibration = UR5EMocapCalibration()

    ur5e_mocap_calibration.destroy_node() # this line is optional 
    rclpy.shutdown()


if __name__ == '__main__':
    main()