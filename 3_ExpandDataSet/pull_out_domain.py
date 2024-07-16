import numpy as np
import os

def main(folder_name):
	
	goals = []

	for file in os.listdir(folder_name):
		if not file[-4:] == ".npz": continue 

		data = np.load(folder_name + "/" + file)

		goals.append(data["traj_pos_save"][:, -1, :])

	goals = np.array(goals)
	goals = goals.reshape(goals.shape[0]*goals.shape[1], goals.shape[2])

	return goals

if __name__ == '__main__':
	main("../3_ExpandDataSet/raw_data")