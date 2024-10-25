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
            
        
        
        match populationFunction:
            case "initRepeat":
                self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr, n=individualSize)
            case "initIterate":
                self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attr, n=individualSize)
            case "initCycle":
                self.toolbox.register("individual", tools.initCycle, creator.Individual, self.toolbox.attr, n=individualSize)
            case _:
                raise ValueError("Invalid population function")

        
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", evalOneMax)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)

        self.toolbox.register("select", getattr(tools, "selTournament"), tournsize=tournamentSize)


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
        # Saving (Pickling) the list
        with open(f"population/{self.id}/population.pkl", "wb") as f:
            pickle.dump(pop, f)

        # Loading (Unpickling) the list
        # with open("large_list.pkl", "rb") as f:
        #     loaded_list = pickle.load(f)

    

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
        


