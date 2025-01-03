import operator
import math
import random
import os
import pickle

import numpy
import matplotlib.pyplot as plt
import networkx as nx

from functools import partial

from deap import algorithms, base, creator, tools, gp, cma

from operators import Operators
from evalFunctions import EvalFunctions


class GpRunner:

    def __init__(self, id, arity=1):
        self.id = id
        self.toolbox = base.Toolbox()
        self.pset = gp.PrimitiveSet("MAIN", arity)
        self.arity = arity
        self.operators = {
            "add": {"operator": operator.add, "arity": 2},
            "sub": {"operator": operator.sub, "arity": 2},
            "mul": {"operator": operator.mul, "arity": 2},
            "div": {"operator": Operators.protectedDiv, "arity": 2},
            "neg": {"operator": operator.neg, "arity": 1},
            "cos": {"operator": math.cos, "arity": 1},
            "sin": {"operator": math.sin, "arity": 1},
            "lf": {"operator": Operators.lf, "arity": 1},
        }

    def addPrimitives(self, primitives):
        for primitive in primitives:
            if primitive in self.operators:
                operator_info = self.operators[primitive]
                self.pset.addPrimitive(
                    operator_info["operator"], operator_info["arity"]
                )
            else:
                print(f"Warning: Primitive '{primitive}' not found in operators.")

    def addEphemeralConstant(self):
        self.pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))

    def renameArguments(self, arg_names):
        arg_dict = {f"ARG{i}": name for i, name in enumerate(arg_names)}
        self.pset.renameArguments(**arg_dict)

    def evalSymbReg(self, individual, points, realFunction):
        # Transform the tree expression in a callable function
        func = self.toolbox.compile(expr=individual)
        # Evaluate the mean squared error between the expression
        # and the real function : x**4 + x**3 + x**2 + x
        sqerrors = ((func(x) - eval(realFunction)) ** 2 for x in points)
        return (math.fsum(sqerrors) / len(points),)

    def create(
        self,
        individualType="PrimitiveTree",
        expr="genHalfAndHalf",
        min_=1,
        max_=2,
        realFunction="x**4 + x**3 + x**2 + x",
        individualFunction="initIterate",
        populationFunction="initRepeat",
        selectionFunction="selTournament",
        tournamentSize=3,
        expr_mut="genFull",
        expr_mut_min=0,
        expr_mut_max=2,
        crossoverFunction="cxOnePoint",
        terminalProb=0.1,
        mutationFunction="mutUniform",
        mateHeight=17,
        mutHeight=17,
        weights=(-1.0,),
    ):

        creator.create("Fitness", base.Fitness, weights=weights)
        creator.create(
            "Individual", getattr(gp, individualType), fitness=creator.Fitness
        )

        self.toolbox.register(
            "expr", getattr(gp, expr), pset=self.pset, min_=min_, max_=max_
        )
        self.toolbox.register(
            "individual",
            getattr(tools, individualFunction),
            creator.Individual,
            self.toolbox.expr,
        )
        self.toolbox.register(
            "population",
            getattr(tools, populationFunction),
            list,
            self.toolbox.individual,
        )
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register(
            "evaluate",
            self.evalSymbReg,
            points=[x / 10.0 for x in range(-10, 10)],
            realFunction=realFunction,
        )

        if selectionFunction == "selTournament":
            self.toolbox.register(
                "select", getattr(tools, selectionFunction), tournsize=tournamentSize
            )
        else:
            self.toolbox.register("select", getattr(tools, selectionFunction))

        match crossoverFunction:
            case "cxOnePoint":
                self.toolbox.register("mate", getattr(gp, crossoverFunction))
            case "cxOnePointLeafBiased":
                self.toolbox.register(
                    "mate",
                    getattr(gp, crossoverFunction),
                    termpb=terminalProb,
                )
            case "cxSemantic":
                self.toolbox.register(
                    "mate",
                    getattr(gp, crossoverFunction),
                    gen_func=getattr(gp, expr_mut),
                    pset=self.pset,
                )
            case _:
                raise ValueError("Selected crossover function is not available")

        self.toolbox.register(
            "expr_mut", getattr(gp, expr_mut), min_=expr_mut_min, max_=expr_mut_max
        )
        self.toolbox.register(
            "mutate",
            getattr(gp, mutationFunction),
            expr=self.toolbox.expr_mut,
            pset=self.pset,
        )

        self.toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=mateHeight),
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=mutHeight),
        )

    def run(
        self,
        algorithm="eaSimple",
        populationSize=300,
        generations=40,
        cxpb=0.5,
        mutpb=0.1,
        mu=1000,
        lambda_=4,
        N=10,
        hofSize=1,
    ):

        random.seed(318)

        pop = self.toolbox.population(n=populationSize)
        hof = tools.HallOfFame(hofSize)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        match algorithm:
            case "eaSimple":
                pop, logbook = algorithms.eaSimple(
                    pop,
                    self.toolbox,
                    cxpb=cxpb,
                    mutpb=mutpb,
                    ngen=generations,
                    stats=mstats,
                    halloffame=hof,
                    verbose=True,
                )

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
                    stats=mstats,
                    halloffame=hof,
                    verbose=True,
                )

                self.storePopulation(pop)

                return logbook, hof

            case "eaMuCommaLambda":
                pop, logbook = algorithms.eaMuCommaLambda(
                    pop,
                    self.toolbox,
                    mu=mu,
                    lambda_=lambda_,
                    cxpb=cxpb,
                    mutpb=mutpb,
                    ngen=generations,
                    stats=mstats,
                    halloffame=hof,
                    verbose=True,
                )

                self.storePopulation(pop)

                return logbook, hof

            case "eaGenerateUpdate":
                numpy.random.seed(128)
                strategy = cma.Strategy(centroid=[5.0] * N, sigma=5.0, lambda_=20 * N)
                self.toolbox.register("generate", strategy.generate, creator.Individual)
                self.toolbox.register("update", strategy.update)

                pop, logbook = algorithms.eaGenerateUpdate(
                    self.toolbox,
                    ngen=generations,
                    stats=mstats,
                    halloffame=hof,
                    verbose=True,
                )

                return logbook, hof

            case _:
                raise ValueError("Algorithm not available")

    def storePopulation(self, pop):

        os.makedirs(f"population/{self.id}/", exist_ok=True)

        with open(f"population/{self.id}/population.pkl", "wb") as f:
            pickle.dump(pop, f)

        # Loading (Unpickling) the list
        # with open("large_list.pkl", "rb") as f:
        #     loaded_list = pickle.load(f)

    def createPlots(self, logbook):
        # gen = logbook.select("gen")
        # avg = logbook.select("avg")
        # min_ = logbook.select("min")
        # max_ = logbook.select("max")

        os.makedirs(f"plots/{self.id}/", exist_ok=True)

        self.plotGraph()

    def plotGraph(self):
        expr = self.toolbox.individual()
        nodes, edges, labels = gp.graph(expr)

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        # pos = nx.graphviz_layout(g, prog="dot")

        # Use a built-in layout
        pos = nx.spring_layout(g)  # Replace with
        # pos = nx.circular_layout(g)
        # pos = nx.shell_layout(g)

        nx.draw_networkx_nodes(g, pos)
        nx.draw_networkx_edges(g, pos)
        nx.draw_networkx_labels(g, pos, labels)
        plt.savefig(f"plots/{self.id}/graph.png", dpi=300)
        plt.close()
