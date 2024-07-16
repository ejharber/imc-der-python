import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import rtde_control
import rtde_receive
import numpy as np

from geometry_msgs.msg import PoseStamped
from mocap4r2_msgs.msg import RigidBodies

import time
import os
import threading
import sys
from scipy.spatial import distance, ConvexHull
from deap import base, creator, tools, algorithms  # Import DEAP components

sys.path.append("../UR5e")
from CustomRobots import *

class UR5e_CollectData(Node):
    def __init__(self):
        super().__init__('collect_rope_data')

        self.UR5e = UR5eCustom()

        self.rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
        self.rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

        self.home_joint_pose = [180, -53.25, 134.66, -171.28, -90, 0]
        self.home_cart_pose = None

        # mocap cb
        self.mocap_subscription = self.create_subscription(RigidBodies, '/rigid_bodies', self.mocap_callback, 10)
        self.mocap_data = None

        # mocap pb
        self.mocap_data_goal = [-1.0, -1.0, -1.0]
        self.mocap_data_actual = [-1.0, -1.0, -1.0]

        timer_period = 0.002
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.mocap_history = []
        self.mocap_delay = 14

        self.calibration = np.load("calibration_data.npz")

        # Image subscriber
        self.subscription = self.create_subscription(
            Image,
            'camera/raw_image',
            self.image_callback,
            1)
        self.bridge = CvBridge()

        # Create scalable OpenCV window
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)

        time.sleep(1)
        self.go_to_home()

        # Load domain points and compute the n-sided polygon
        self.points, self.polygon = self.load_domain_and_polygon(n=20)

    def load_domain_and_polygon(self, n):
        folder_name = "raw_data"
        points = []

        for file in os.listdir(folder_name):
            if not file.endswith(".npz"): 
                continue 

            file_name = os.path.join(folder_name, file)
            data = np.load(file_name)

            mocap_data = data["mocap_data_save"][0, :3, 0]
            imgpoints, _ = cv2.projectPoints(mocap_data, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])

            points.append(imgpoints[0, 0, :])

        points = np.array(points)
        polygon = self.optimize_polygon(points, n)

        return points, polygon

    def optimize_polygon(self, points, n):
        def fitness(individual):
            polygon_points = points[np.array(individual, dtype=bool)]
            if len(polygon_points) < n:
                return float('inf'),  # Invalid polygon
            hull = ConvexHull(polygon_points)
            perimeter = np.sum([distance.euclidean(polygon_points[hull.vertices[i]],
                                                   polygon_points[hull.vertices[i-1]]) for i in range(len(hull.vertices))])
            return perimeter,

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        toolbox = base.Toolbox()
        toolbox.register("attr_bool", np.random.randint, 0, 2)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(points))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)  # Adjust mutation probability
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", fitness)

        population = toolbox.population(n=500)  # Increase population size
        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=100, verbose=False)  # Adjust parameters

        best_individual = tools.selBest(population, k=1)[0]
        best_polygon_points = points[np.array(best_individual, dtype=bool)]

        if len(best_polygon_points) < n:
            return points[:n]

        hull = ConvexHull(best_polygon_points)
        return best_polygon_points[hull.vertices]


    def timer_callback(self):
        pass

    def mocap_callback(self, msg):
        markers = []
        for rigidbody in msg.rigidbodies:
            for marker in rigidbody.markers:
                markers.append(marker)
        self.mocap_history.append(markers)
        if len(self.mocap_history) > self.mocap_delay:
            self.mocap_history = self.mocap_history[1:]

    def image_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if len(self.polygon) > 0:
                # Fill the polygon
                pts = self.polygon.reshape((-1, 1, 2)).astype(np.int32)
                cv2.fillPoly(img, [pts], (255, 0, 0))

                # Plot the original points
                for point in self.points:
                    img = cv2.circle(img, tuple(point.astype(int)), 3, (0, 255, 0), -1)

            if len(self.mocap_history) > 0:
                mocap_data = []
                for marker in self.mocap_history[0]:
                    mocap_data.append([marker.translation.x, marker.translation.y, marker.translation.z])
                mocap_data = np.array(mocap_data)

                imgpoints, _ = cv2.projectPoints(mocap_data, self.calibration["R"], self.calibration["t"], self.calibration["mtx"], self.calibration["dist"])
                for i in range(imgpoints.shape[0]):
                    if imgpoints[i, 0, 0] < 0 or imgpoints[i, 0, 1] < 0: continue 
                    if imgpoints[i, 0, 0] > img.shape[0] or imgpoints[i, 0, 1] > img.shape[1]: continue 
                    img = cv2.circle(img, (int(imgpoints[i, 0, 0]), int(imgpoints[i, 0, 1])), 3, (0, 0, 255), -1)

                cv2.imshow("Image", img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error("Error converting image: %s" % str(e))

    def go_to_home(self):
        print('go home')
        q = np.copy(self.home_joint_pose)
        q = np.array(q) * np.pi / 180.0
        self.rtde_c.moveJ(q, 0.2, 0.2)
        self.home_cart_pose = self.rtde_r.getActualTCPPose()

    def take_data(self):
        pass

    def repeat_data_routine(self):
        folder_name = "raw_data"

        for _ in range(100):
            count = np.random.randint(0, 26)

            print(count)
            
            file = str(count) + ".npz"

            if file not in os.listdir(folder_name): continue 

            file_name = folder_name + "/" + file
            data = np.load(file_name)

            self.mocap_data_goal = [-1.0, -1.0, -1.0]
            self.mocap_data_actual = [-1.0, -1.0, -1.0]

            for i in range(50):
                self.take_data()
                time.sleep(1.0/500)

            q0 = data["q0_save"]
            qf = data["qf_save"]
            self.mocap_data_goal = data["mocap_data_save"][0,:,0]

            for i in range(50):
                self.take_data()
                time.sleep(1.0/500)

            self.rope_swing(qf)

    def rope_swing(self, q):
        self.go_to_home()
        self.reset_rope()
        self.go_to_home()

        q0 = np.copy(self.home_joint_pose)
        qf = np.copy(q)

        traj = self.UR5e.create_trajectory(q0, qf, time=1)

        # Parameters
        velocity = 3
        acceleration = 5
        dt = 1.0/500  # 2ms
        lookahead_time = 0.1
        gain = 1000

        for i in range(traj.shape[1]):
            t_start = self.rtde_c.initPeriod()
            q = traj[:, i]

            self.rtde_c.servoJ(q, velocity, acceleration, dt, lookahead_time, gain)

            self.take_data()

            self.rtde_c.waitPeriod(t_start)

        self.mocap_data_actual = self.mocap_data

        self.take_data()

        self.rtde_c.servoStop()

        self.take_data()

        for i in range(50):
            self.take_data()
            time.sleep(dt)

        self.go_to_home()

    def reset_rope(self):
        p = np.copy(self.home_cart_pose)
        p[2] -= 0.03
        self.rtde_c.moveL(p, speed = 0.01, acceleration = 0.01)

        p = np.copy(self.home_cart_pose)
        self.rtde_c.moveL(p, speed = 0.01, acceleration = 0.01)

def main(args=None):
    rclpy.init(args=args)

    ur5e = UR5e_CollectData()

    # Use MultiThreadedExecutor to run the node with multiple threads
    executor = MultiThreadedExecutor()
    executor.add_node(ur5e)

    # Start the repeat_data_routine in a separate thread
    ur5e_thread = threading.Thread(target=ur5e.repeat_data_routine)
    ur5e_thread.start()

    try:
        # Spin the executor to process callbacks
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        ur5e.destroy_node()
        rclpy.shutdown()
        ur5e_thread.join()

if __name__ == '__main__':
    main()
