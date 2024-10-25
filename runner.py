import random
from deap import base, creator, tools, algorithms
import numpy
import matplotlib.pyplot as plt
import os

from evalFunctions import *

class Runner:

    def __init__(self, id):
        self.toolbox = base.Toolbox()
        self.id = id


    def create(self, weights=(1.0,), individualSize=10, indpb=0.10):
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox.register("attr_bool", random.randint, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_bool, n=individualSize)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", evalOneMax)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)

        self.toolbox.register("select", tools.selTournament, tournsize=3)


    def run(self, poputlationSize=5000, generations=10, cxpb=0.5, mutpb=0.2):
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
        
        return pop, logbook, hof
    
    def createPlot(self, gen, avg, min_, max_):
        plt.plot(gen, avg, label="average")
        plt.plot(gen, min_, label="minimum")
        plt.plot(gen, max_, label="maximum")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.legend(loc="lower right")
        
        os.makedirs(f"plots/{self.id}/", exist_ok=True)
        
        plt.savefig(f"plots/{self.id}/fitness_plot.png", dpi=300)  
        plt.close() 
        


