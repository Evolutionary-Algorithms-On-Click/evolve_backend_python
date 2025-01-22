import os

class MLRunner:
    def __init__(self, id, mlImportCodeString, evalFunctionCodeString, sep):
        # self.toolbox = base.Toolbox()
        self.id = id

        # Create a file to store the code.
        os.makedirs(f"ml/{self.id}", exist_ok=True)
        self.code = open(f"ml/{self.id}/code.py", "w")
        self.code.write("# DEAP Imports\n\n")
        self.code.write("import random, os\n")
        self.code.write("from deap import base, creator, tools, algorithms\n")
        self.code.write("import numpy\n")
        self.code.write("import matplotlib.pyplot as plt\n")
        self.code.write("from functools import reduce\n")
        self.code.write("from scoop import futures\n\n")
        self.code.write("import pandas as pd\n")
        self.code.write("import warnings\n")
        self.code.write("warnings.filterwarnings(\"ignore\")\n\n")
        self.code.write("# ML Imports\n\n")
        self.code.write(mlImportCodeString)
        self.code.write("\n\n")

        self.code.write("def download_csv_from_google_drive_share_link(url):\n")
        self.code.write("\tfile_id = url.split(\"/\")[-2]\n")
        self.code.write("\tdwn_url = \"https://drive.google.com/uc?export=download&id=\" + file_id\n")
        self.code.write(f"\treturn pd.read_csv(dwn_url, sep=\"{sep}\")\n\n")

        self.code.write(evalFunctionCodeString)
        self.code.write("\n\n")

    def setEvalFunction(self,):
        self.code.write("\ttoolbox.register(\"evaluate\", mlEvalFunction, X=X, y=y)\n")

    def setPopulationFunction(self):
        self.code.write(f"\ttoolbox.register(\"individual\", tools.initRepeat, creator.Individual, toolbox.attr, n=len(X.columns))\n")
        self.code.write(f"\ttoolbox.register(\"population\", tools.initRepeat, list, toolbox.individual)\n")


    def create(
            self,
            # evaluationFunction = "evalOneMax",
            # weights=(1.0,),
            # individualSize=10,
            indpb=0.10,
            crossoverFunction="cxOnePoint",
            mutationFunction="mutFlipBit",
            selectionFunction="selTournament",
            tournamentSize=3
            ):

        self.code.write("toolbox = base.Toolbox()\n\n")
        self.code.write(f"\ntoolbox.register(\"mate\", tools.{crossoverFunction})\n")
        self.code.write(f"toolbox.register(\"mutate\", tools.{mutationFunction}, indpb={indpb})\n")

        if selectionFunction == "selTournament":
            self.code.write(f"toolbox.register(\"select\", tools.{selectionFunction}, tournsize={tournamentSize})\n")
        else:
            self.code.write(f"toolbox.register(\"select\", tools.{selectionFunction})\n")
        
        self.code.write("\ntoolbox.register(\"map\", futures.map)\n\n")


    def run(
        self,
        algorithm="eaSimple",
        googleDriveUrl="https://drive.google.com/file/d/15Xi9UkwuBCJPpj_--reO2Wz9nOScA0Wd/view?usp=share_link",
        targetColumnName="target",
        weights=(1.0,),
        populationSize=24,
        generations=100,
        cxpb=0.5,
        mutpb=0.2,
        mu=1000,
        lambda_=4,
        hofSize = 1
        ):

        self.code.write(f"\ndef main():\n")
        self.code.write(f"\trootPath = os.path.dirname(os.path.abspath(__file__))\n")
        self.code.write(f"\turl = \"{googleDriveUrl}\"\n")
        self.code.write(f"\tdf = download_csv_from_google_drive_share_link(url)\n")
        self.code.write(f"\n\ttarget = \"{targetColumnName}\"\n")
        self.code.write(f"\n\tX = df.drop(target, axis=1)\n")
        self.code.write(f"\ty = df[target]\n")

        self.code.write(f"\n\taccuracy = mlEvalFunction([1 for _ in range(len(X.columns))], X, y)\n")
        # self.code.write(f"\tprint(\"No Feature Selection Accuracy: \", accuracy)\n")

        self.code.write(f"\n\tcreator.create(\"FitnessMax\", base.Fitness, weights={weights})\n")
        self.code.write("\tcreator.create(\"Individual\", list, fitness=creator.FitnessMax)\n")
        self.code.write("\ttoolbox.register(\"attr\", random.randint, 0, 1)\n")
            
        self.setPopulationFunction()
        self.setEvalFunction()

        self.code.write(f"\tpopulationSize = {populationSize}\n")
        self.code.write(f"\tgenerations = {generations}\n")
        self.code.write(f"\tcxpb = {cxpb}\n")
        self.code.write(f"\tmutpb = {mutpb}\n")
        self.code.write(f"\tN = len(X.columns)\n")
        self.code.write(f"\thofSize = {hofSize}\n")
        self.code.write(f"\n\tpop = toolbox.population(n=populationSize)\n")
        self.code.write(f"\thof = tools.HallOfFame(hofSize)\n")
        self.code.write("\n\tstats = tools.Statistics(lambda ind: ind.fitness.values)\n")
        self.code.write("\tstats.register(\"avg\", numpy.mean)\n")
        self.code.write("\tstats.register(\"min\", numpy.min)\n")
        self.code.write("\tstats.register(\"max\", numpy.max)\n")

        match algorithm:
            case "eaSimple":
                self.code.write("\tpop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=stats, halloffame=hof, verbose=True)\n")
            
            case "eaMuPlusLambda":
                self.code.write(f"\tmu = {mu}\n")
                self.code.write(f"\tlambda_ = {lambda_}\n")
                self.code.write("\tpop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=mu, lambda_=lambda_, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=stats, halloffame=hof, verbose=True)\n")
        
            case "eaMuCommaLambda":
                self.code.write(f"\tmu = {mu}\n")
                self.code.write(f"\tlambda_ = {lambda_}\n")
                self.code.write("\tpop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=mu, lambda_=lambda_, cxpb=cxpb, mutpb=mutpb, ngen=generations, stats=stats, halloffame=hof, verbose=True)\n")
            
            case "eaGenerateUpdate":
                self.code.write("\tnumpy.random.seed(128)\n")
                self.code.write(f"\tstrategy = cma.Strategy(centroid=[5.0]*len(X.columns), sigma=5.0, lambda_=20*len(X.columns))\n")
                self.code.write("\ttoolbox.register(\"generate\", strategy.generate, creator.Individual)\n")
                self.code.write("\ttoolbox.register(\"update\", strategy.update)\n")
                self.code.write("\tpop, logbook = algorithms.eaGenerateUpdate(toolbox, ngen=generations, stats=stats, halloffame=hof, verbose=True)\n")
            
            case _:
                raise ValueError("Algorithm not available")
        
        self.code.write("\tout_file = open(f\"{rootPath}/best.txt\", \"w\")\n")
        self.code.write("\tout_file.write(f\"Original Accuracy before applying EA: {accuracy}\\n\")\n")
        self.code.write("\tout_file.write(f\"Best individual is:\\n{hof[0]}\\nwith fitness: {hof[0].fitness}\\n\")\n")
        self.code.write("\tbest_columns = [i for i in range(len(hof[0])) if hof[0][i] == 1]\n")
        self.code.write("\tbest_column_names = X.columns[best_columns]\n")
        self.code.write("\tout_file.write(f\"\\nBest individual columns:\\n{best_column_names.values}\")\n")
        self.code.write("\tout_file.close()\n")

        self.createPlots()
        self.code.write("\n\n")
        self.code.write("if __name__ == '__main__':\n")
        self.code.write("\tmain()\n")

        self.code.close()

        # Execute the code: python -m scoop code.py
        exitCode = os.system(f"python -m scoop ml/{self.id}/code.py")
        return exitCode


    def createPlots(self,):
        self.code.write("\n\n")
        self.code.write(f"\tgen = logbook.select(\"gen\")\n")
        self.code.write(f"\tavg = logbook.select(\"avg\")\n")
        self.code.write(f"\tmin_ = logbook.select(\"min\")\n")
        self.code.write(f"\tmax_ = logbook.select(\"max\")\n\n")

        # Save LogBook as .log.
        self.code.write("\twith open(f\"{rootPath}/logbook.txt\", \"w\") as f:\n")
        self.code.write(f"\t\tf.write(str(logbook))\n")
        self.code.write("\n")

        self.createFitnessPlot()

    # def createMutationCrossoverEffectPlot(self, gen, avg_fitness):
    #     fitness_diff = [avg_fitness[i] - avg_fitness[i-1] for i in range(1, len(avg_fitness))]
    #     plt.plot(gen[1:], fitness_diff, label="Fitness Change", color="purple")
    #     plt.xlabel("Generation")
    #     plt.ylabel("Fitness Change")
    #     plt.title("Effect of Mutation and Crossover on Fitness")
    #     plt.legend()
    #     plt.savefig(f"plots/{self.id}/mutation_crossover_effect.png", dpi=300)
    #     plt.close()

    def createFitnessPlot(self,):
        self.code.write("\tplt.plot(gen, avg, label=\"average\")\n")
        self.code.write("\tplt.plot(gen, min_, label=\"minimum\")\n")
        self.code.write("\tplt.plot(gen, max_, label=\"maximum\")\n")
        self.code.write("\tplt.xlabel(\"Generation\")\n")
        self.code.write("\tplt.ylabel(\"Fitness\")\n")
        self.code.write("\tplt.legend(loc=\"lower right\")\n")
        self.code.write("\tplt.savefig(f\"{rootPath}/fitness_plot.png\", dpi=300)\n")
        self.code.write("\tplt.close()\n")
