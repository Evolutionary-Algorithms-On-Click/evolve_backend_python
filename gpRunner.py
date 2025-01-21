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
from scoop import futures


class GpRunner:

    def __init__(self, id, arity=1):
        self.id = id

        # Create code file.
        self.code = open(f"code/{id}.py", "w")
        self.code.write("import operator\nimport math\nimport random\n")
        self.code.write("import numpy\nimport matplotlib.pyplot as plt\n")
        self.code.write("import networkx as nx\n")
        self.code.write("from functools import partial\n")
        self.code.write("from deap import algorithms, base, creator, tools, gp, cma\n")
        self.code.write("from scoop import futures\n\n")

        self.code.write("def evalSymbReg(individual, points, realFunction):\n")
        self.code.write("\t# Transform the tree expression in a callable function\n")
        self.code.write("\tfunc = toolbox.compile(expr=individual)\n")
        self.code.write("\t# Evaluate the mean squared error between the expression\n")
        self.code.write("\t# and the real function : x**4 + x**3 + x**2 + x\n")
        self.code.write("\tsqerrors= ((func(x) - eval(realFunction))**2 for x in points)\n")
        self.code.write("\treturn (math.fsum(sqerrors) / len(points),)\n\n")

        self.toolbox = base.Toolbox()
        self.pset = gp.PrimitiveSet("MAIN", arity)
        
        self.code.write("toolbox = base.Toolbox()\n")
        self.code.write("pset = gp.PrimitiveSet('MAIN', 1)\n")

        self.code.write("\n\n")

        self.code.write("def protectedDiv(left, right):\n")
        self.code.write("\ttry:\n")
        self.code.write("\t\treturn left / right\n")
        self.code.write("\texcept ZeroDivisionError:\n")
        self.code.write("\t\treturn 1\n\n")

        self.code.write("def lf(x): \n")
        self.code.write("\treturn 1 / (1 + numpy.exp(-x))\n\n")
    
        self.arity = arity
        self.operators = {
            "add": {"operator": operator.add, "arity": 2, 'code': 'operator.add'},
            "sub": {"operator": operator.sub, "arity": 2, 'code': 'operator.sub'},
            "mul": {"operator": operator.mul, "arity": 2, 'code': 'operator.mul'},
            "div": {"operator": Operators.protectedDiv, "arity": 2, 'code': 'protectedDiv'},
            "neg": {"operator": operator.neg, "arity": 1, 'code': 'operator.neg'},
            "cos": {"operator": math.cos, "arity": 1, 'code': 'math.cos'},
            "sin": {"operator": math.sin, "arity": 1, 'code': 'math.sin'},
            "lf": {"operator": Operators.lf, "arity": 1, 'code': 'lf'},
        }

    def addPrimitives(self, primitives):
        for primitive in primitives:
            if primitive in self.operators:
                operator_info = self.operators[primitive]
                self.pset.addPrimitive(
                    operator_info["operator"], operator_info["arity"]
                )
                self.code.write(f"pset.addPrimitive({operator_info['code']}, {operator_info['arity']})\n")
            else:
                print(f"Warning: Primitive '{primitive}' not found in operators.")

    def addEphemeralConstant(self):
        self.pset.addEphemeralConstant("rand101", partial(random.randint, -1, 1))
        self.code.write("pset.addEphemeralConstant('rand101', partial(random.randint, -1, 1))\n")

    def renameArguments(self, arg_names):
        arg_dict = {f"ARG{i}": name for i, name in enumerate(arg_names)}
        self.code.write(f"arg_dict = {arg_dict}\n")
        
        self.pset.renameArguments(**arg_dict)
        self.code.write(f"pset.renameArguments(**arg_dict)\n\n")

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
        mutationMode="one",
        mateHeight=17,
        mutHeight=17,
        weights=(-1.0,),
    ):

        creator.create("Fitness", base.Fitness, weights=weights)
        creator.create(
            "Individual", getattr(gp, individualType), fitness=creator.Fitness
        )

        self.code.write(f"creator.create('Fitness', base.Fitness, weights={weights})\n")
        self.code.write("creator.create('Individual', gp.PrimitiveTree, fitness=creator.Fitness)\n")

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

        self.code.write(f"toolbox.register('expr', gp.{expr}, pset=pset, min_={min_}, max_={max_})\n")
        self.code.write(f"toolbox.register('individual', tools.{individualFunction}, creator.Individual, toolbox.expr)\n")
        self.code.write(f"toolbox.register('population', tools.{populationFunction}, list, toolbox.individual)\n")
        self.code.write("toolbox.register('compile', gp.compile, pset=pset)\n\n")

        self.toolbox.register(
            "evaluate",
            self.evalSymbReg,
            points=[x / 10.0 for x in range(-10, 10)],
            realFunction=realFunction,
        )
        self.code.write("toolbox.register('evaluate', evalSymbReg, points=[x / 10.0 for x in range(-10, 10)], realFunction='x**4 + x**3 + x**2 + x')\n\n")

        if selectionFunction == "selTournament":
            self.toolbox.register(
                "select", getattr(tools, selectionFunction), tournsize=tournamentSize
            )
            self.code.write(f"toolbox.register('select', tools.{selectionFunction}, tournsize={tournamentSize})\n")
        else:
            self.toolbox.register("select", getattr(tools, selectionFunction))
            self.code.write(f"toolbox.register('select', tools.{selectionFunction})\n")

        match crossoverFunction:
            case "cxOnePoint":
                self.toolbox.register("mate", getattr(gp, crossoverFunction))
                self.code.write("toolbox.register('mate', gp.cxOnePoint)\n")
            case "cxOnePointLeafBiased":
                self.toolbox.register(
                    "mate",
                    getattr(gp, crossoverFunction),
                    termpb=terminalProb,
                )
                self.code.write(f"toolbox.register('mate', gp.cxOnePointLeafBiased, termpb={terminalProb})\n")
            case "cxSemantic":
                self.toolbox.register(
                    "mate",
                    getattr(gp, crossoverFunction),
                    gen_func=getattr(gp, expr_mut),
                    pset=self.pset,
                )
                self.code.write("toolbox.register('mate', gp.cxSemantic, gen_func=gp.genFull, pset=pset)\n")
            case _:
                raise ValueError("Selected crossover function is not available")

        self.toolbox.register(
            "expr_mut", getattr(gp, expr_mut), min_=expr_mut_min, max_=expr_mut_max
        )
        self.code.write(f"toolbox.register('expr_mut', gp.{expr_mut}, min_={expr_mut_min}, max_={expr_mut_max})\n")

        match mutationFunction:
            case "mutUniform":
                self.toolbox.register(
                    "mutate",
                    getattr(gp, mutationFunction),
                    expr=self.toolbox.expr_mut,
                    pset=self.pset,
                )
                self.code.write("toolbox.register('mutate', gp.mutUniform, expr=toolbox.expr_mut, pset=pset)\n")
            case "mutShrink":
                self.toolbox.register(
                    "mutate",
                    getattr(gp, mutationFunction),
                )
                self.code.write("toolbox.register('mutate', gp.mutShrink)\n")
            case "mutNodeReplacement":
                self.toolbox.register(
                    "mutate",
                    getattr(gp, mutationFunction),
                    pset=self.pset,
                )
                self.code.write("toolbox.register('mutate', gp.mutNodeReplacement, pset=pset)\n")
            case "mutInsert":
                self.toolbox.register(
                    "mutate",
                    getattr(gp, mutationFunction),
                    pset=self.pset,
                )
                self.code.write("toolbox.register('mutate', gp.mutInsert, pset=pset)\n")
            case "mutEphemeral":
                self.toolbox.register(
                    "mutate",
                    getattr(gp, mutationFunction),
                    mode=mutationMode,
                )
                self.code.write(f"toolbox.register('mutate', gp.mutEphemeral, mode='{mutationMode}')\n")
            case "mutSemantic":
                self.toolbox.register(
                    "mutate",
                    getattr(gp, mutationFunction),
                    gen_func=getattr(gp, expr_mut),
                    pset=self.pset,
                )
                self.code.write("toolbox.register('mutate', gp.mutSemantic, gen_func=gp.genFull, pset=pset)\n")
            case _:
                raise ValueError("Selected mutation function is not available")

        self.toolbox.decorate(
            "mate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=mateHeight),
        )
        self.toolbox.decorate(
            "mutate",
            gp.staticLimit(key=operator.attrgetter("height"), max_value=mutHeight),
        )

        self.code.write(f"toolbox.decorate('mate', gp.staticLimit(key=operator.attrgetter('height'), max_value={mateHeight}))\n")
        self.code.write(f"toolbox.decorate('mutate', gp.staticLimit(key=operator.attrgetter('height'), max_value={mutHeight}))\n")

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
        self.code.write("\ntoolbox.register('map', futures.map)\n")
        self.code.write("\ndef main():\n")

        random.seed(318)
        self.code.write("\trandom.seed(318)\n")

        pop = self.toolbox.population(n=populationSize)
        hof = tools.HallOfFame(hofSize)

        self.code.write(f"\tpop = toolbox.population(n={populationSize})\n")
        self.code.write(f"\thof = tools.HallOfFame({hofSize})\n")

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", numpy.mean)
        mstats.register("std", numpy.std)
        mstats.register("min", numpy.min)
        mstats.register("max", numpy.max)

        self.code.write("\tstats_fit = tools.Statistics(lambda ind: ind.fitness.values)\n")
        self.code.write("\tstats_size = tools.Statistics(len)\n")
        self.code.write("\tmstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)\n")
        self.code.write("\tmstats.register('avg', numpy.mean)\n")
        self.code.write("\tmstats.register('std', numpy.std)\n")
        self.code.write("\tmstats.register('min', numpy.min)\n")
        self.code.write("\tmstats.register('max', numpy.max)\n")

        # Run the algorithm in parallel.
        self.toolbox.register("map", futures.map)

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
                self.code.write(f"\tpop, logbook = algorithms.eaSimple(pop, toolbox, cxpb={cxpb}, mutpb={mutpb}, ngen={generations}, stats=mstats, halloffame=hof, verbose=True)\n")

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
                self.code.write(f"\tpop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu={mu}, lambda_={lambda_}, cxpb={cxpb}, mutpb={mutpb}, ngen={generations}, stats=mstats, halloffame=hof, verbose=True)\n")

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
                self.code.write(f"\tpop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu={mu}, lambda_={lambda_}, cxpb={cxpb}, mutpb={mutpb}, ngen={generations}, stats=mstats, halloffame=hof, verbose=True)\n")

                self.storePopulation(pop)

                return logbook, hof

            case "eaGenerateUpdate":
                numpy.random.seed(128)
                strategy = cma.Strategy(centroid=[5.0] * N, sigma=5.0, lambda_=20 * N)
                self.toolbox.register("generate", strategy.generate, creator.Individual)
                self.toolbox.register("update", strategy.update)

                self.code.write("\tnumpy.random.seed(128)\n")
                self.code.write(f"\tstrategy = cma.Strategy(centroid=[5.0] * {N}, sigma=5.0, lambda_=20 * {N})\n")
                self.code.write("\ttoolbox.register('generate', strategy.generate, creator.Individual)\n")
                self.code.write("\ttoolbox.register('update', strategy.update)\n")

                pop, logbook = algorithms.eaGenerateUpdate(
                    self.toolbox,
                    ngen=generations,
                    stats=mstats,
                    halloffame=hof,
                    verbose=True,
                )
                self.code.write(f"\tpop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen={generations}, stats=mstats, halloffame=hof, verbose=True)\n")

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

    def createPlots(self, logbook, hof):
        # gen = logbook.select("gen")
        # avg = logbook.select("avg")
        # min_ = logbook.select("min")
        # max_ = logbook.select("max")

        os.makedirs(f"plots/{self.id}/", exist_ok=True)

        self.plotGraph(hof[0])

    def plotGraph(self, ind):
        expr = ind
        nodes, edges, labels = gp.graph(expr)

        self.code.write(f"\n\texpr = hof[0]\n")
        self.code.write(f"\tnodes, edges, labels = gp.graph(expr)\n")

        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")

        plt.figure(figsize=(7,7))
        nx.draw_networkx_nodes(g, pos, node_size=900, node_color="skyblue")
        nx.draw_networkx_edges(g, pos, edge_color="gray")
        nx.draw_networkx_labels(g, pos, labels, font_color="black")
        plt.axis("off")
        plt.savefig(f"plots/{self.id}/graph.png", dpi=300)
        plt.close()

        self.code.write("\tg = nx.Graph()\n")
        self.code.write("\tg.add_nodes_from(nodes)\n")
        self.code.write("\tg.add_edges_from(edges)\n")
        self.code.write("\tpos = nx.nx_agraph.graphviz_layout(g, prog='dot')\n")
        self.code.write("\n\tplt.figure(figsize=(7,7))\n")
        self.code.write("\tnx.draw_networkx_nodes(g, pos, node_size=900, node_color='skyblue')\n")
        self.code.write("\tnx.draw_networkx_edges(g, pos, edge_color='gray')\n")
        self.code.write("\tnx.draw_networkx_labels(g, pos, labels, font_color='black')\n")
        self.code.write("\tplt.axis('off')\n")
        self.code.write("\tplt.savefig('graph.png', dpi=300)\n")
        self.code.write("\tplt.close()\n")

        # g = nx.Graph()
        # g.add_nodes_from(nodes)
        # g.add_edges_from(edges)
        # # pos = nx.graphviz_layout(g, prog="dot")

        # # Use a built-in layout
        # pos = nx.spring_layout(g)  # Replace with
        # # pos = nx.circular_layout(g)
        # # pos = nx.shell_layout(g)

        # nx.draw_networkx_nodes(g, pos)
        # nx.draw_networkx_edges(g, pos)
        # nx.draw_networkx_labels(g, pos, labels)
        # plt.savefig(f"plots/{self.id}/graph.png", dpi=300)
        # plt.close()

        # self.code.write("\tg = nx.Graph()\n")
        # self.code.write("\tg.add_nodes_from(nodes)\n")
        # self.code.write("\tg.add_edges_from(edges)\n")
        # self.code.write("\tpos = nx.spring_layout(g)\n")
        # self.code.write("\tnx.draw_networkx_nodes(g, pos)\n")
        # self.code.write("\tnx.draw_networkx_edges(g, pos)\n")
        # self.code.write("\tnx.draw_networkx_labels(g, pos, labels)\n")
        # self.code.write("\tplt.savefig('graph.png', dpi=300)\n")
        # self.code.write("\tplt.close()\n")
        
