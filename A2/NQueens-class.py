import random

class GeneticAlgorithmSolver:
    def __init__(self):
        self.BOARD_SIZE = 8
        self.POPULATION_SIZE = 100
        self.MUTATION_RATE = 0.3
        self.MAX_GENERATIONS = 404

    def generate_chromosome(self):
        chromosome = list(range(self.BOARD_SIZE))
        random.shuffle(chromosome)
        return chromosome

    def calculate_fitness(self, chromosome):
        threats = 0
        for i in range(self.BOARD_SIZE):
            for j in range(i + 1, self.BOARD_SIZE):
                if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == j - i:
                    threats += 1
        return threats

    def selection(self, population):
        parents = []
        for _ in range(self.POPULATION_SIZE):
            tournament = random.sample(population, 5)
            tournament.sort(key=lambda x: x['fitness'])
            parents.append(tournament[0]['chromosome'])
        return parents

    def crossover(self, parent1, parent2):
        crossover_point = random.randint(1, self.BOARD_SIZE - 2)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutate(self, chromosome):
        mutated_chromosome = chromosome[:]
        if random.random() < self.MUTATION_RATE:
            index = random.randint(0, self.BOARD_SIZE - 1)
            mutated_chromosome[index] = random.randint(0, self.BOARD_SIZE - 1)
        return mutated_chromosome

    def run(self):
        population = [{'chromosome': self.generate_chromosome(), 'fitness': 0} for _ in range(self.POPULATION_SIZE)]

        for generation in range(self.MAX_GENERATIONS):
            for individual in population:
                individual['fitness'] = self.calculate_fitness(individual['chromosome'])
            best_fitness = min([individual['fitness'] for individual in population])
            if best_fitness == 0:
                break

            parents = self.selection(population)
            offspring = []
            while len(offspring) < self.POPULATION_SIZE:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                mutated_child1 = self.mutate(child1)
                mutated_child2 = self.mutate(child2)
                offspring.extend([{'chromosome': mutated_child1, 'fitness': 0}, {'chromosome': mutated_child2, 'fitness': 0}])
            population = offspring

        if generation == self.MAX_GENERATIONS - 1:
            print("No solution found in", self.MAX_GENERATIONS, "iterations.")
            exit(0)

        solution = min(population, key=lambda x: x['fitness'])
        return solution['chromosome']

if __name__ == "__main__":
    ga = GeneticAlgorithmSolver()
    solution = ga.run()

    print("===================Solution:===================")
    print("Chromosome: ", solution)
    print("Board: ")
    for row in solution:
        print("".join([' Q ' if col == row else ' - ' for col in range(ga.BOARD_SIZE)]))
