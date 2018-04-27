from MatNN.NN import *
from MatNN.GeneticAlgorithm import *


test_inputs = np.array([1, 2, 3, 4])
test_outputs = np.array([-1, 0.5])

NN_AMOUNT = 100

gen_alg = GeneticAlgorithm()
nns = [NN(4, 5, 5, 2) for i in range(NN_AMOUNT)]

while True:
    # calculate nns and store results
    results = [n.calculate(test_inputs) for n in nns]
    # calculate fitness
    results = np.array(results)
    dist = results[:] - test_outputs
    cost = np.sum(np.abs(dist), axis=1)
    fitness = np.max(cost) - cost

    # export chromosomes
    pop = np.array([n.export_genome() for n in nns])

    # do genetic algorithm
    changed_pop = gen_alg.recombine(pop, fitness)
    # import genomes
    for i in changed_pop:
        nns[i].import_genome(pop[i])



