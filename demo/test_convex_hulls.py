import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from scipy.spatial import distance

# Define the list of points (replace this with your own list)
points = np.array([
    [0, 0],
    [1, 1],
    [1, 0],
    [0, 1],
    [0.5, 0.5],
    [2, 2],
    [2, 3],
    [3, 3],
    [3, 2]
])

# Number of sides for the polygon
n_sides = 12

# Fitness function: calculate the perimeter of the polygon
def fitness(individual):
    poly_points = points[individual]
    perim = 0
    for i in range(len(poly_points)):
        perim += distance.euclidean(poly_points[i], poly_points[(i + 1) % len(poly_points)])
    return perim,

# Setup DEAP genetic algorithm
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", np.random.permutation, len(points))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    np.random.seed(42)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=200, 
                        stats=None, halloffame=hof, verbose=False)

    best_ind = hof[0]
    return best_ind

best_ind = main()

# Plot the points
plt.scatter(points[:, 0], points[:, 1])

# Plot the best polygon
poly_points = points[best_ind[:n_sides]]
poly_points = np.append(poly_points, [poly_points[0]], axis=0)
plt.plot(poly_points[:, 0], poly_points[:, 1], 'r-')

# Show the plot
plt.show()
