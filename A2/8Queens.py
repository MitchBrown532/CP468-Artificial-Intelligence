import random

# Constants
BOARD_SIZE = 8
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
MAX_GENERATIONS = 1000

def generate_chromosome():
    # Generate a random chromosome
    chromosome = [random.randint(0, BOARD_SIZE-1) for _ in range(BOARD_SIZE)]
    return chromosome

def calculate_fitness(chromosome):
    # Calculate the fitness of a chromosome
    threats = 0
    for i in range(BOARD_SIZE):
        for j in range(i+1, BOARD_SIZE):
            if chromosome[i] == chromosome[j] or abs(chromosome[i] - chromosome[j]) == j - i:
                threats += 1
    return threats

def selection(population):
    # Perform tournament selection to select parents
    parents = []
    for _ in range(POPULATION_SIZE):
        tournament = random.sample(population, 5)
        tournament.sort(key=lambda x: x['fitness'])
        parents.append(tournament[0]['chromosome'])
    return parents

def crossover(parent1, parent2):
    # Perform one-point crossover
    crossover_point = random.randint(1, BOARD_SIZE-2)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(chromosome):
    # Perform mutation by randomly changing a position in the chromosome
    mutated_chromosome = chromosome[:]
    if random.random() < MUTATION_RATE:
        index = random.randint(0, BOARD_SIZE-1)
        mutated_chromosome[index] = random.randint(0, BOARD_SIZE-1)
    return mutated_chromosome

def genetic_algorithm():
    # Generate initial population
    population = [{'chromosome': generate_chromosome(), 'fitness': 0} for _ in range(POPULATION_SIZE)]

    # Main loop
    for generation in range(MAX_GENERATIONS):
        # Calculate fitness for each chromosome in the population
        for individual in population:
            individual['fitness'] = calculate_fitness(individual['chromosome'])

        # Check termination condition
        best_fitness = min([individual['fitness'] for individual in population])
        if best_fitness == 0:
            break

        # Perform selection
        parents = selection(population)

        # Create new generation
        offspring = []
        while len(offspring) < POPULATION_SIZE:
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            mutated_child1 = mutate(child1)
            mutated_child2 = mutate(child2)
            offspring.extend([{'chromosome': mutated_child1, 'fitness': 0}, {'chromosome': mutated_child2, 'fitness': 0}])

        # Replace old population with the new generation
        population = offspring

    # Extract the solution
    solution = min(population, key=lambda x: x['fitness'])
    return solution['chromosome']

# Run the genetic algorithm and print the solution
solution = genetic_algorithm()
print("Solution:")
for row in solution:
    print("".join(['Q' if col == row else '-' for col in range(BOARD_SIZE)]))
