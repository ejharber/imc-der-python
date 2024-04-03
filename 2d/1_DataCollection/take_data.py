#!/usr/bin/env python3

import rtde_control
import numpy as np

# ros packages
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from mocap4r2_msgs.msg import Markers
import matplotlib.pyplot as plt

import numpy as np
import time

def create_trajectory(weighpoints):
    def quintic_func(q0, qf, T, qd0=0, qdf=0):
        X = [
            [ 0.0,          0.0,         0.0,        0.0,     0.0,  1.0],
            [ T**5,         T**4,        T**3,       T**2,    T,    1.0],
            [ 0.0,          0.0,         0.0,        0.0,     1.0,  0.0],
            [ 5.0 * T**4,   4.0 * T**3,  3.0 * T**2, 2.0 * T, 1.0,  0.0],
            [ 0.0,          0.0,         0.0,        2.0,     0.0,  0.0],
            [20.0 * T**3,  12.0 * T**2,  6.0 * T,    2.0,     0.0,  0.0],
        ]
        # fmt: on
        coeffs, resid, rank, s = np.linalg.lstsq(
            X, np.r_[q0, qf, qd0, qdf, 0, 0], rcond=None
        )

        # coefficients of derivatives
        coeffs_d = coeffs[0:5] * np.arange(5, 0, -1)
        coeffs_dd = coeffs_d[0:4] * np.arange(4, 0, -1)

        return lambda x: (
            np.polyval(coeffs, x),
            np.polyval(coeffs_d, x),
            np.polyval(coeffs_dd, x),
        )

    def quintic(q0, qf, t, qd0=0, qdf=0):
        tf = max(t)

        polyfunc = quintic_func(q0, qf, tf, qd0, qdf)

        # evaluate the polynomials
        traj = polyfunc(t)
        p = traj[0]
        pd = traj[1]
        pdd = traj[2]

        return p

    traj = []
    time = 0.5 # set sim time to be 0.5 seconds

    for dim in range(weighpoints.shape[0]):
        x_1_traj = [0, time]
        x_2_traj = [weighpoints[dim, 0], weighpoints[dim, 1]]
        t = np.linspace(0, time, round(time*500)) # can send data at 500 hz
        traj.append(quintic(x_2_traj[0], x_2_traj[1], t))

    traj = np.array(traj)

    return traj


class UR5E(Node):
    def __init__(self):

        super().__init__('collect_rope_data')

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.10.60")

        self.home_joint_pose = [180, -73, 108, -218, 266, 0]

        # ati cb
        self.ati_subscription = self.create_subscription(WrenchStamped, '/ati', self.ati_callback, 1)
        self.ati_data = None
        self.ati_data_zero = None

        # mocap cb
        self.mocap_subscription = self.create_subscription(Markers, '/markers', self.mocap_callback, 1)
        self.mocap_data = None

        # commanded cb
        self.ur5e_cmd_subscription = self.create_subscription(JointState, '/ur5e/command/joint_states', self.ur5e_cmd_callback, 1)
        self.ur5e_cmd_data = None

        # tool cb
        self.ur5e_tool_subscription = self.create_subscription(PoseStamped, '/ur5e/feedback/end_effector', self.ur5e_tool_callback, 1)
        self.ur5e_tool_data = None

        # joint measured cb
        self.ur5e_jointstate_subscription = self.create_subscription(JointState, '/ur5e/feedback/joint_states', self.ur5e_jointstate_callback, 1)
        self.ur5e_jointstate_data = None

        timer_period = 0.001  # run as fast as possible (its a camera so max 100 hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.q_save = []
        self.ati_data_save = []
        self.mocap_data_save = []
        self.ur5e_cmd_data_save = []
        self.ur5e_tool_data_save = []
        self.ur5e_jointstate_data_save = []

        time.sleep(1)
        self.go_to_home()

    def timer_callback(self):
        pass

    def ati_callback(self, msg):
        self.ati_data = [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z]

    def mocap_callback(self, msg):
        self.mocap_data = np.zeros((25, 3))
        for i, marker in enumerate(msg.markers):
            self.mocap_data[i, :] = np.array([marker.translation.x, marker.translation.y, marker.translation.z])

    def ur5e_cmd_callback(self, msg):
        self.ur5e_cmd_data = msg.position

    def ur5e_tool_callback(self, msg):
        self.ur5e_tool_data = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.w, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]

    def ur5e_jointstate_callback(self, msg):
        self.ur5e_jointstate_data = msg.position

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q)

    def take_data(self, q, i):
        rclpy.spin_once(self)

        if i%5 == 0:
            self.q_save.append(q)
            self.ati_data_save.append(self.ati_data - self.ati_data_zero)
            self.mocap_data_save.append(self.mocap_data)
            self.ur5e_cmd_data_save.append(self.ur5e_cmd_data)
            self.ur5e_tool_data_save.append(self.ur5e_tool_data)
            self.ur5e_jointstate_data_save.append(self.ur5e_jointstate_data)

        # print(np.array(self.mocap_data).shape)

    def map_work_space_servoj(self):
        print('start mapping')

        for q2 in np.linspace(-140, -80, 10):
            for q3 in np.linspace(-30, -30 + 90, 10):

                self.go_to_home()
                self.reset_rope()
                self.go_to_home()

                q = [180, -110 ,q2, q3, 90, 0]
                # q = [180, -90, -100, 0, 90, 0]

                weighpoints = [[self.home_joint_pose[i], q[i]] for i in range(6)]
                weighpoints = np.array(weighpoints)
                traj = create_trajectory(weighpoints)

                # Parameters
                velocity = 3
                acceleration = 5
                dt = 1.0/500  # 2ms
                lookahead_time = 0.05
                gain = 1000

                for i in range(traj.shape[1]):
                    q = traj[:, i]

                    q = q * np.pi / 180
                    t_start = self.rtde_c.initPeriod()
                    self.rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)
                    self.take_data(q, i)
                    self.rtde_c.waitPeriod(t_start)
                    self.take_data(q, i)


                plt.plot(self.q_save, 'r-')
                plt.plot(self.ur5e_cmd_data, 'b-')
                plt.plot(self.ur5e_jointstate_data, 'k-')
                plt.show()

                self.rtde_c.servoStop()

                # np.array(self.q_save)
                # np.array(self.ati_data_save)
                # np.array(self.mocap_data_save)
                # np.array(self.ur5e_cmd_data_save)
                # np.array(self.ur5e_tool_data_save)
                # np.array(self.ur5e_jointstate_data_save)

                np.savez('first', q_save = self.q_save,
                                  ati_data = self.ati_data_save,
                                  mocap_data = self.mocap_data_save,
                                  ur5e_cmd_data = self.ur5e_cmd_data_save,
                                  ur5e_tool_data = self.ur5e_tool_data_save,
                                  ur5e_jointstate_data = self.ur5e_jointstate_data_save)

        self.go_to_home()

    def test_rope_swing(self):
        print('start mapping')

        q2 = -100
        q3 = 0

        self.go_to_home()
        self.reset_rope()
        self.go_to_home()

        return

        q = [180, -110 ,q2, q3, 90, 0]
        # q = [180, -90, -100, 0, 90, 0]

        weighpoints = [[self.home_joint_pose[i], q[i]] for i in range(6)]
        weighpoints = np.array(weighpoints)
        traj = create_trajectory(weighpoints)

        # Parameters
        velocity = 3
        acceleration = 5
        dt = 1.0/500  # 2ms
        lookahead_time = 0.04
        gain = 1000

        tick = time.time()
        for i in range(traj.shape[1]):
            t_start = self.rtde_c.initPeriod()
            q = traj[:, i]
            q = q * np.pi / 180

            self.rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)

            self.take_data(q, i)

            self.rtde_c.waitPeriod(t_start)
        
        self.q_save = np.array(self.q_save)
        self.ur5e_cmd_data_save = np.array(self.ur5e_cmd_data_save)
        self.ur5e_jointstate_data_save = np.array(self.ur5e_jointstate_data_save)
        self.ati_data_save = np.array(self.ati_data_save)

        plt.figure('ATI Data')
        plt.plot(self.ati_data_save)

        plt.figure('Robot Trajectory')
        plt.plot(np.linspace(0, 0.25,  self.q_save[0:-20,:].shape[0]), self.q_save[0:-20,:], 'r.')
        plt.plot(np.linspace(0, 0.25,  self.q_save[0:-20,:].shape[0]), self.q_save[0:-20,0], 'r.', label = 'predicted trajectory')
        # plt.plot(self.ur5e_cmd_data_save[20:,:], 'b-')
        plt.plot(np.linspace(0, 0.25,  self.q_save[0:-20,:].shape[0]), self.ur5e_jointstate_data_save[20:,:], 'k-')
        plt.plot(np.linspace(0, 0.25,  self.q_save[0:-20,:].shape[0]), self.ur5e_jointstate_data_save[20:,0], 'k-', label = 'UR5e feedback')
        plt.xlabel('time (s)')
        plt.ylabel('joint label')
        plt.title('Measuring Robot Sim2Real Gap')
        plt.legend(loc='upper right')
        plt.show()

        self.rtde_c.servoStop()

        self.go_to_home()

    def map_work_space_movej(self):
        print('start mapping')

        for q1 in np.linspace(-110, -60, 5):
            for q2 in np.linspace(-140, -80, 5):
                for q3 in np.linspace(-30, -30 + 90, 5):

                    self.go_to_home()
                    self.reset_rope()
                    self.go_to_home()

                    q = [180, q1 ,q2, q3, 90, 0]
                    q = np.array(q) * np.pi / 180.0
                    self.rtde_c.moveJ(q)

        self.go_to_home()

    def zero_ati(self):
        ati_data = []
        for _ in range(100):
            rclpy.spin_once(self)
            if self.ati_data is None: continue 
            ati_data.append(self.ati_data)

        self.ati_data_zero = np.mean(np.array(ati_data), axis = 0)

    def reset_rope(self):
        q = np.copy(self.home_joint_pose)
        q[1] -= 10
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, speed = 0.5, acceleration = 0.5)

        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, speed = 0.05, acceleration = 0.1)
        time.sleep(2)
        self.zero_ati()
        # zero ati


def main(args=None):
    rclpy.init(args=args)

    ur5e = UR5E()

    # rclpy.spin(ur5e)
    ur5e.test_rope_swing()

    ur5e.destroy_node() # this line is optional 
    rclpy.shutdown()


if __name__ == '__main__':
    main()

