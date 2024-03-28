import numpy as np
import matplotlib.pyplot as plt

names = []
results = []

for lstm_num_layers in [1, 2, 3, 4]:
    for lstm_hidden_size in [10, 20, 50, 100, 200]:
        for mlp_num_layers in [2, 3, 4, 5]:
            for mlp_hidden_size in [10, 50, 100, 500, 1000]:
                save_file_name = "LSTM_" + "alldata_" + str(lstm_num_layers) + "_" + str(lstm_hidden_size) + "_" + str(mlp_num_layers) + "_" + str(mlp_hidden_size)
                
                try:
                    data = np.load("models_daction_dgoal/" + save_file_name + ".npz")
                except: 
                    continue 

                names.append(save_file_name[13:])
                training_data = data['loss_values_test']
                results.append(training_data[-1])

                # plt.plot(training_data)
results = np.array(results)
names = np.array(names)
results_sorted_i = np.argsort(results)
print(results_sorted_i)
print(names)
results = results[results_sorted_i]
names = names[results_sorted_i]

plt.bar(names, results)
plt.xticks(rotation=90)
plt.show()
