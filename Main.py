from MatNN.NN import *
from MatNN.GeneticAlgorithm import *


def test_function(inputs):
    if np.sum(inputs) > 3:
        return 1
    else:
        return 0


max_cost = 2


NN_AMOUNT = 100

gen_alg = GeneticAlgorithm()
nns = [NN(6, 20, 2, 1) for i in range(NN_AMOUNT)]

while True:
    inputs = np.random.rand(6)
    test_outputs = test_function(inputs)

    # calculate nns and store results
    results = [n.calculate(inputs) for n in nns]
    # calculate fitness
    results = np.array(results)
    dist = np.abs(results[:] - test_outputs)
    cost = np.sum(dist, axis=1)
    fitness = max_cost - cost

    print(np.sum(fitness))

    # export chromosomes
    pop = np.array([n.export_genome() for n in nns])

    # do genetic algorithm
    changed_pop = gen_alg.recombine(pop, fitness)
    # import genomes
    for i in changed_pop:
        nns[i].import_genome(pop[i])



