from random import random
import matplotlib.pyplot as plt
import pandas as pd

class Individual:

  def __init__(self, chromosome_size):
    self.chromosome_size = chromosome_size
    self.fitness = 0
    self.chromosome = [1 if random() > 0.5 else 0 for i in range(chromosome_size)]
    
class Population:

  def __init__(self, population_size, chromosome_size=10):
    self.population_size = population_size
    self.chromosome_size = chromosome_size
    self.population_fitness = 0
    self.individuals = [Individual(chromosome_size) for i in range(population_size)]

  def get_fittest(self, offset=0):
    return sorted(self.individuals, key=lambda individual: individual.fitness, reverse=True)[offset]

class Item(object):
  
  def __init__(self, valor, peso):
        self.valor = valor;
        self.peso = peso;
        
        
dataset = pd.read_csv('P08-cwp.txt.txt')
items = [Item(dataset.iloc[i,2],dataset.iloc[i,1]) for i in range(len(dataset))]
capacity = 6404180

class GeneticAlgorithm:

  def __init__(self, population_size, mutation_rate, crossover_rate, elitism_count):
    self.population_size = population_size
    self.mutation_rate = mutation_rate
    self.crossover_rate = crossover_rate
    self.elitism_count = elitism_count

  def init_population(self, chromosome_size):
    return Population(self.population_size, chromosome_size)

  def calculate_item(self, individual):
    valortotal = 0
    pesototal = 0
    
    for i in range(len(individual.chromosome)):
        if individual.chromosome[i] == 1:
            valortotal += items[i].valor
            pesototal += items[i].peso
      
    print(f'Capacidade: {capacity}')
    print(f'Valor Total: {valortotal}')
    print(f'Peso Total: {pesototal}')

  def calculate_fitness(self, individual):
    valortotal = 0
    pesototal = 0
    
    for i in range(len(individual.chromosome)):
        if individual.chromosome[i] == 1:
            valortotal += items[i].valor
            pesototal += items[i].peso
    
    if pesototal > capacity:
        individual.fitness = 0
        return 0
    else:
        individual.fitness = valortotal
    return valortotal

  def evaluate_population(self, population):
    population_fitness = 0
    for i in range(population.population_size):
       population_fitness += self.calculate_fitness(population.individuals[i])

    population.population_fitness = population_fitness

  def select_parent(self, population):
    roulette_whell_position = random() * population.population_fitness
    spin_whell = 0
    for i in range(population.population_size):
      spin_whell += population.get_fittest(i).fitness
      if spin_whell >= roulette_whell_position:
        return population.get_fittest(i)

    return population.get_fittest(-1)

  def crossover_population(self, population):
    new_population = Population(population.population_size, population.chromosome_size)
    
    for i in range(population.population_size):
      parent1 = population.get_fittest(i)
      if self.crossover_rate > random() and i > self.elitism_count:
        parent2 = self.select_parent(population)
        offspring = Individual(population.chromosome_size)

        cut_index = int(random() * parent1.chromosome_size)
        for j in range(parent1.chromosome_size):
          if j <= cut_index:
            offspring.chromosome[j] = parent1.chromosome[j]
          else:
            offspring.chromosome[j] = parent2.chromosome[j]

        new_population.individuals[i] = offspring
      else:
        new_population.individuals[i] = parent1

    return new_population

  def mutate_population(self, population):
    new_population = Population(population.population_size, population.chromosome_size)
    for i in range(population.population_size):
      individual = population.individuals[i]

      for j in range(individual.chromosome_size):
        if self.mutation_rate > random():
          if individual.chromosome[j] == 1:
             individual.chromosome[j] = 0
          else:
            individual.chromosome[j] = 1

      new_population.individuals[i] = individual
      
    return new_population

generation = 0

ga = GeneticAlgorithm(100, 0.01, 0.97, 2)
population = ga.init_population(24)
ga.evaluate_population(population)



while generation < 100:
    
  print(f'treinando: {generation}')

  population = ga.crossover_population(population)

  population = ga.mutate_population(population)

  ga.evaluate_population(population)
  
#  for i in range(len(population.individuals)):
#      print(f'Individuo {i+1}: {population.individuals[i].chromosome}')
      
  
#  print(f'Melhor individuo: {population.get_fittest().chromosome}')
#  print(f'Valores: {ga.calculate_item(population.get_fittest())}')

  generation += 1 

print(f'Found solution in {generation} generations')
print(f'The best solution: {population.get_fittest().chromosome}')
ga.calculate_item(population.get_fittest())


saida = population.get_fittest().chromosome;

posicao = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24
nome = "1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24"

plt.figure()
#plt.show()

plt.figure(figsize = (10, 1))
plt.title('ESPAÇO MOCHILA')
plt.xlabel('Livros')
plt.xticks(posicao, nome)
plt.bar(posicao, saida, width = 1)
plt.show()
