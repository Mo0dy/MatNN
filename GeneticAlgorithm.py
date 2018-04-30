import numpy as np


class GeneticAlgorithm(object):
    def __init__(self, mutation_rate=0.1, mutation_changse=0.01, recombination_amount=0.1):
        # the maximum stepsize in any direction
        self.m_amount = mutation_rate
        self.m_changse = mutation_changse
        self.r_amount = recombination_amount

    def mutate(self, population):
        mutation_matrix = np.random.rand(population.shape[0], population.shape[1]) < self.m_changse
        ran_num = np.sum(mutation_matrix)
        population[mutation_matrix] += (np.random.rand(ran_num) - 0.5) * (self.m_amount * 2)

    def recombine(self, population, fitness):
        # stochastic universal sampling (select all parents at once with a change proportional to their fitness)
        # the probabilities to be parents
        p = (fitness - np.min(fitness))
        p = p / np.sum(p)
        children_amount = int(fitness.shape[0] * self.r_amount)
        parents_amount =  children_amount * 2
        # select the parents
        parents_indices = np.random.choice(population.shape[0], parents_amount, False, p)
        chromosome_length = population.shape[1]
        # the indices at which the genomes of the different parents will be taken from
        split_points = np.random.random_integers(0, chromosome_length, children_amount)
        # create children chromosomes
        children = np.zeros((children_amount, chromosome_length))
        for i in range(children_amount):
            children[i, :] = np.append(population[parents_indices[i * 2], :split_points[i]], population[parents_indices[i * 2 + 1], split_points[i]:])

        # mutate children
        self.mutate(children)

        inverse_fitness = np.max(fitness) - fitness
        inverse_p = inverse_fitness / np.sum(inverse_fitness)

        # select the chromosomes that will be replaced with the inverse fitness
        dead_chromosomes_indices = np.random.choice(population.shape[0], children_amount, False, inverse_p)

        # replace the chromosomes with the children
        population[dead_chromosomes_indices] = children[:]

        # returns the indices of the genome that changed
        return dead_chromosomes_indices
