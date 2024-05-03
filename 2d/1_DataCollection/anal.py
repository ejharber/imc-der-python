import numpy as np
import matplotlib.pyplot as plt

plt.figure('Rope Position Trajectory')
ax_p = plt.axes(projection='3d')

plt.figure('Rope Force Trajectory')
ax_f = plt.axes(projection='3d')

plt.figure("Joint States")

for i in range(1):

    file = f'raw_data/{i}.npz'

    data = np.load(file)

    # mocap_data_save = data["mocap_data_save"]

    # zline = mocap_data_save[500+5:1000+5, 0, 2]
    # xline = mocap_data_save[500+5:1000+5, 1, 2]
    # yline = mocap_data_save[500+5:1000+5, 2, 2]
    # ax_p.plot3D(xline, yline, zline, 'b-')

    # mocap_data_save = data["ati_data_save"]

    # zline = mocap_data_save[500+5:1000+5, 0]
    # xline = mocap_data_save[500+5:1000+5, 1]
    # yline = mocap_data_save[500+5:1000+5, 2]
    # ax_f.plot3D(xline, yline, zline, 'b-')

    # # zline = mocap_data_save[1000, 0, 2]
    # # xline = mocap_data_save[1000, 1, 2]
    # # yline = mocap_data_save[1000, 2, 2]
    # # ax.plot3D(xline, yline, zline, 'b.')

    # plt.figure("Joint States")
    # Q = data["ur5e_tool_data_save"]
    # plt.plot(Q)

    print(data["q_save"])

    ur5e_tool_data_save = data["ur5e_tool_data_save"]

    # zline = mocap_data_save[1000, 0, 2]
    # xline = mocap_data_save[1000, 1, 2]
    # yline = mocap_data_save[1000, 2, 2]
    plt.figure("Joint States")

    plt.plot(ur5e_tool_data_save[:, :3])
    print(ur5e_tool_data_save.shape)

plt.show()