import numpy as np
import os

def load_data(folder_name):
	
	actions = []
	goals = []

	for file in os.listdir(folder_name):
		if not file[-4:] == ".npz": continue 

		data = np.load(folder_name + "/" + file)

		actions.append(data["qf_save"][:, [2, 3, 4]])
		goals.append(data["traj_pos_save"][:, -1, :])

	actions = np.array(actions)
	actions = np.reshape(actions, (actions.shape[0]*actions.shape[1], actions.shape[2]))

	goals = np.array(goals)
	goals = np.reshape(goals, (goals.shape[0]*goals.shape[1], goals.shape[2]))
	noise = np.random.normal(loc=0, scale=0.05, size=goals.shape)
	goals = goals + noise

	split = int((actions.shape[0])*2/3)
	return actions[:split, :], goals[:split, :], actions[split:, :], goals[split:, :] 

if __name__ == '__main__':
	load_data("../../3_ExpandDataSet/raw_data")