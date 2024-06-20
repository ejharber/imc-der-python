import numpy as np
import os

def load_data_zeroshot(folder_name):
	
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
	# goals = np.random.normal(loc=goals, scale=0.05, size=goals.shape)

	split = int((actions.shape[0])*2/3)
	return actions[:split, :], goals[:split, :], actions[split:, :], goals[split:, :] 

def load_data_iterative(folder_name):
	
	delta_actions = []
	delta_goals = []
	traj_pos = []

	for file in os.listdir(folder_name):
		if not file[-4:] == ".npz": continue 

		data = np.load(folder_name + "/" + file)

		actions = data["qf_save"][:, [2, 3, 4]]
		goals = data["traj_pos_save"][:, -1, :]

		split = int(actions.shape[0] / 2)

		delta_actions.append(actions[:split, :] - actions[split:, :])
		delta_goals.append(goals[:split, :] - goals[split:, :])

		traj_pos.append(data["traj_pos_save"][:split, :, :])
		
	delta_actions = np.array(delta_actions)
	delta_actions = np.reshape(delta_actions, (delta_actions.shape[0]*delta_actions.shape[1], delta_actions.shape[2]))
	delta_goals = np.array(delta_goals)
	delta_goals = np.reshape(delta_goals, (delta_goals.shape[0]*delta_goals.shape[1], delta_goals.shape[2]))
	traj_pos = np.array(traj_pos)
	traj_pos = np.reshape(traj_pos, (traj_pos.shape[0]*traj_pos.shape[1], traj_pos.shape[2], traj_pos.shape[3]))
	traj_pos = np.random.normal(loc=traj_pos, scale=0.05, size=traj_pos.shape)

	print(delta_actions.shape, delta_goals.shape, traj_pos.shape)

	split = int((actions.shape[0])*2/3)
	train_data = dict()
	train_data["time_series"] = traj_pos[:split, :, :]
	train_data["classic"] = delta_actions[:split, :]

	train_labels = delta_goals[:split, :]

	test_data = dict()
	test_data["time_series"] = traj_pos[split:, :, :]
	test_data["classic"] = delta_actions[split:, :]

	test_labels = delta_goals[split:, :]

	return train_data, train_labels, test_data, test_labels

if __name__ == '__main__':
	load_data_iterative("../3_ExpandDataSet/raw_data")