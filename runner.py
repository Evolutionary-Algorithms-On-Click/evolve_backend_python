import random
from deap import base, creator, tools, algorithms
import numpy
import matplotlib.pyplot as plt
import os
import pickle

from evalFunctions import *

class Runner:

    def __init__(self, id):
        self.toolbox = base.Toolbox()
        self.id = id


    def create(
            self,
            individual = "binaryString",
            populationFunction = "initRepeat",
            weights=(1.0,),
            individualSize=10,
            indpb=0.10,
            randomRange = [0, 100],
            crossoverFunction="cxOnePoint",
            mutationFunction="mutFlipBit",
            selectionFunction="selTournament",
            tournamentSize=3
            ):
        
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMax)

        match individual:
            case "binaryString":
                self.toolbox.register("attr", random.randint, 0, 1)
            case "permutation":
                self.toolbox.register("attr", random.sample, range(individualSize), individualSize)
            case "float":
                self.toolbox.register("attr", random.uniform, randomRange[0], randomRange[1])
            case "int":
                self.toolbox.register("attr", random.randint, randomRange[0], randomRange[1])
            case _:
                raise ValueError("Invalid individual type")
            
        
        self.toolbox.register("individual", getattr(tools, populationFunction), creator.Individual, self.toolbox.attr, n=individualSize)

        
        self.toolbox.register("population", getattr(tools, populationFunction), list, self.toolbox.individual)

        self.toolbox.register("evaluate", evalOneMax)

        self.toolbox.register("mate", getattr(tools, crossoverFunction) )
        self.toolbox.register("mutate", getattr(tools, mutationFunction), indpb=indpb)

        self.toolbox.register("select", getattr(tools, selectionFunction), tournsize=tournamentSize)


    def run(
        self,
        poputlationSize=5000,
        generations=10,
        cxpb=0.5,
        mutpb=0.2
        ):

        pop = self.toolbox.population(n=poputlationSize)
        hof = tools.HallOfFame(1)
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean)
        stats.register("min", numpy.min)
        stats.register("max", numpy.max)
        
        pop, logbook = algorithms.eaSimple(
            pop, 
            self.toolbox, 
            cxpb=cxpb, 
            mutpb=mutpb, 
            ngen=generations, 
            stats=stats, 
            halloffame=hof, 
            verbose=True
        )

        self.storePopulation(pop)
        
        return logbook, hof
    

    def storePopulation(self,pop):

        os.makedirs(f"population/{self.id}/", exist_ok=True)
        
        with open(f"population/{self.id}/population.pkl", "wb") as f:
            pickle.dump(pop, f)

        # Loading (Unpickling) the list
        # with open("large_list.pkl", "rb") as f:
        #     loaded_list = pickle.load(f)

    def createPlots(self, logbook):
        gen = logbook.select("gen")
        avg = logbook.select("avg")
        min_ = logbook.select("min")
        max_ = logbook.select("max")

        os.makedirs(f"plots/{self.id}/", exist_ok=True)

        self.createFitnessPlot(gen, avg, min_, max_)
        self.createMutationCrossoverEffectPlot(gen, avg)

    def createMutationCrossoverEffectPlot(self, gen, avg_fitness):
        fitness_diff = [avg_fitness[i] - avg_fitness[i-1] for i in range(1, len(avg_fitness))]
        plt.plot(gen[1:], fitness_diff, label="Fitness Change", color="purple")
        plt.xlabel("Generation")
        plt.ylabel("Fitness Change")
        plt.title("Effect of Mutation and Crossover on Fitness")
        plt.legend()
        plt.savefig(f"plots/{self.id}/mutation_crossover_effect.png", dpi=300)
        plt.close()

    def createFitnessPlot(self, gen, avg, min_, max_):
        plt.plot(gen, avg, label="average")
        plt.plot(gen, min_, label="minimum")
        plt.plot(gen, max_, label="maximum")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc="lower right")
        plt.savefig(f"plots/{self.id}/fitness_plot.png", dpi=300)  
        plt.close() 
        


