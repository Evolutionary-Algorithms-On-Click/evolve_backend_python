import random
from deap import base, creator, tools, algorithms, cma
import numpy
import matplotlib.pyplot as plt
import os
import pickle

from evalFunctions import EvalFunctions
# from scoop import futures

class Runner:

    def __init__(self, id):
        self.toolbox = base.Toolbox()
        self.id = id

        # Create a file to store the code.
        self.code = open(f"code/{self.id}.py", "w")
        self.code.write("import random\nfrom deap import base, creator, tools, algorithms\nimport numpy\nimport matplotlib.pyplot as plt\nfrom functools import reduce\n")
        self.code.write("from scoop import futures\n\n")

    def setEvalFunction(self, evalFunction, *args):
        if evalFunction in ["evalOneMax", "evalProduct", "evalDifference"]:
            self.toolbox.register("evaluate", getattr(EvalFunctions, evalFunction))
            self.code.write(f"\ntoolbox.register(\"evaluate\", {evalFunction})\n")
        else:
            # self.toolbox.register("evaluate", EvalFunctions.evalFunction, evalFunction, *args)
            raise ValueError("Invalid evaluation function")

    def setPopulationFunction(self, populationFunction, n):
        if populationFunction == "initRepeat":
            self.toolbox.register("individual", getattr(tools, populationFunction), creator.Individual, self.toolbox.attr, n=n)
            self.code.write(f"toolbox.register(\"individual\", tools.initRepeat, creator.Individual, toolbox.attr, n={n})\n")
        else:
            raise ValueError("Invalid population function")

    def create(
            self,
            individual = "binaryString",
            populationFunction = "initRepeat",
            evaluationFunction = "evalOneMax",
            weights=(1.0,),
            individualSize=10,
            indpb=0.10,
            randomRange = [0, 100],
            crossoverFunction="cxOnePoint",
            mutationFunction="mutFlipBit",
            selectionFunction="selTournament",
            tournamentSize=3
            ):
        
        self.code.write(EvalFunctions.getCodeString(evaluationFunction))
        self.code.write("\n\n")

        self.code.write("toolbox = base.Toolbox()\n\n")
        
        creator.create("FitnessMax", base.Fitness, weights=weights)
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.code.write(f"\ncreator.create(\"FitnessMax\", base.Fitness, weights={weights})\n")
        self.code.write("creator.create(\"Individual\", list, fitness=creator.FitnessMax)\n")

        match individual:
            case "binaryString":
                self.toolbox.register("attr", random.randint, 0, 1)
                self.code.write("toolbox.register(\"attr\", random.randint, 0, 1)\n")
            case "floatingPoint":
                self.toolbox.register("attr", random.uniform, randomRange[0], randomRange[1])
                self.code.write(f"toolbox.register(\"attr\", random.uniform, {randomRange[0]}, {randomRange[1]})\n")
            case "integer":
                self.toolbox.register("attr", random.randint, randomRange[0], randomRange[1])
                self.code.write(f"toolbox.register(\"attr\", random.randint, {randomRange[0]}, {randomRange[1]})\n")
            # case "permutation":
            #     self.toolbox.register("attr", random.sample, range(individualSize), individualSize)
            case _:
                raise ValueError("Invalid individual type")
            
        self.setPopulationFunction(populationFunction, n=individualSize)

        self.toolbox.register("population", getattr(tools, populationFunction), list, self.toolbox.individual)
        self.code.write(f"toolbox.register(\"population\", tools.{populationFunction}, list, toolbox.individual)\n")

        self.setEvalFunction(evaluationFunction)

        self.toolbox.register("mate", getattr(tools, crossoverFunction) )
        self.code.write(f"\ntoolbox.register(\"mate\", tools.{crossoverFunction})\n")

        self.toolbox.register("mutate", getattr(tools, mutationFunction), indpb=indpb)
        self.code.write(f"toolbox.register(\"mutate\", tools.{mutationFunction}, indpb={indpb})\n")

        if selectionFunction == "selTournament":
            self.toolbox.register("select", getattr(tools, selectionFunction), tournsize=tournamentSize)
            self.code.write(f"toolbox.register(\"select\", tools.{selectionFunction}, tournsize={tournamentSize})\n")
        else:
            self.toolbox.register("select", getattr(tools, selectionFunction))
            self.code.write(f"toolbox.register(\"select\", tools.{selectionFunction})\n")
        
        self.code.write("\ntoolbox.register(\"map\", futures.map)\n\n")


    def run(
        self,
        algorithm="eaSimple",
        populationSize=5000,
        generations=10,
        cxpb=0.5,
        mutpb=0.2,
        mu=1000,
        lambda_=4,
        N = 10,
        hofSize = 1
        ):

        self.code.write(f"\ndef main():\n")
        self.code.write(f"\tpopulationSize = {populationSize}\n")
        self.code.write(f"\tgenerations = {generations}\n")
        self.code.write(f"\tcxpb = {cxpb}\n")
        self.code.write(f"\tmutpb = {mutpb}\n")
        self.code.write(f"\tN = {N}\n")
        self.code.write(f"\thofSize = {hofSize}\n")

        pop = self.toolbox.population(n=populationSize)
        self.code.write(f"\n\tpop = toolbox.population(n=populationSize)\n")

        hof = tools.HallOfFame(hofSize)
        self.code.write(f"\thof = tools.HallOfFame(hofSize)\n")
        
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.code.write("\n\tstats = tools.Statistics(lambda ind: ind.fitness.values)\n")

        stats.register("avg", numpy.mean)
        self.code.write("\tstats.register(\"avg\", numpy.mean)\n")

        stats.register("min", numpy.min)
        self.code.write("\tstats.register(\"min\", numpy.min)\n")

        stats.register("max", numpy.max)
        self.code.write("\tstats.register(\"max\", numpy.max)\n")
        
        # Run the algorithm in parallel.
        # self.toolbox.register("map", futures.map)

        match algorithm:
            case "eaSimple":
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
                self.code.write("\tpop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=stats, halloffame=hof, verbose=True)\n")

                self.storePopulation(pop)
            
                return logbook, hof
            
            case "eaMuPlusLambda":
                pop, logbook = algorithms.eaMuPlusLambda(
                    pop, 
                    self.toolbox, 
                    mu=mu, 
                    lambda_=lambda_, 
                    cxpb=cxpb, 
                    mutpb=mutpb, 
                    ngen=generations, 
                    stats=stats, 
                    halloffame=hof, 
                    verbose=True
                )
                self.code.write(f"\tmu = {mu}\n")
                self.code.write(f"\tlambda_ = {lambda_}\n")
                self.code.write("\tpop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=mu, lambda_=lambda_, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=stats, halloffame=hof, verbose=True)\n")

                self.storePopulation(pop)
            
                return logbook, hof
        
            case "eaMuCommaLambda":
                pop, logbook = algorithms.eaMuCommaLambda(
                    pop, self.toolbox,
                    mu = mu,
                    lambda_= lambda_,
                    cxpb=cxpb,
                    mutpb=mutpb,
                    ngen=generations,
                    stats=stats,
                    halloffame=hof,
                    verbose=True
                )
                self.code.write(f"\tmu = {mu}\n")
                self.code.write(f"\tlambda_ = {lambda_}\n")
                self.code.write("\tpop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=mu, lambda_=lambda_, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=stats, halloffame=hof, verbose=True)\n")

                self.storePopulation(pop)

                return logbook, hof
            
            case "eaGenerateUpdate":
                numpy.random.seed(128)
                self.code.write("\tnumpy.random.seed(128)\n")

                strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=20*N)
                self.code.write(f"\tstrategy = cma.Strategy(centroid=[5.0]*{N}, sigma=5.0, lambda_=20*{N})\n")

                self.toolbox.register("generate", strategy.generate, creator.Individual)
                self.code.write("\ttoolbox.register(\"generate\", strategy.generate, creator.Individual)\n")

                self.toolbox.register("update", strategy.update)
                self.code.write("\ttoolbox.register(\"update\", strategy.update)\n")

                pop, logbook = algorithms.eaGenerateUpdate(
                    self.toolbox, 
                    ngen=generations, 
                    stats=stats, 
                    halloffame=hof, 
                    verbose=True
                )
                self.code.write("\tpop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=generations, stats=stats, halloffame=hof, verbose=True)\n")
    
                return logbook, hof
            
            case _:
                raise ValueError("Algorithm not available")
    

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
        
        self.code.write("\n\n")
        self.code.write(f"\tgen = logbook.select(\"gen\")\n")
        self.code.write(f"\tavg = logbook.select(\"avg\")\n")
        self.code.write(f"\tmin_ = logbook.select(\"min\")\n")
        self.code.write(f"\tmax_ = logbook.select(\"max\")\n\n")

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

        self.code.write("\tplt.plot(gen, avg, label=\"average\")\n")
        self.code.write("\tplt.plot(gen, min_, label=\"minimum\")\n")
        self.code.write("\tplt.plot(gen, max_, label=\"maximum\")\n")
        self.code.write("\tplt.xlabel(\"Generation\")\n")
        self.code.write("\tplt.ylabel(\"Fitness\")\n")
        self.code.write("\tplt.legend(loc=\"lower right\")\n")
        self.code.write("\tplt.savefig(f\"fitness_plot.png\", dpi=300)\n")
        self.code.write("\tplt.close()\n")
