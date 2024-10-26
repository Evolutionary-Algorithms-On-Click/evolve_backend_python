class ParamsList:
    def __init__(self):
        self.algorithm = ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda", "eaGenerateUpdate"]
        self.individual = ["binaryString", "floatingPoint", "integer"]
        self.populationFunction = ["initRepeat"]
        self.evaluationFunction = ["evalOneMax", "evalProduct", "evalDifference"]
        self.crossoverFunction = ["cxOnePoint", "cxTwoPoint", "cxPartialyMatched", "cxOrdered", "cxMessyOnePoint"]
        self.mutationFunction = ["mutShuffleIndexes", "mutFlipBit"]
        self.selectionFunction = [
                                    "selTournament", "selRoulette","selNSGA2","selSPEA2",
                                    "selRandom","selBest","selWorst","selStochasticUniversalSampling",
                                    "selLexicase","selAutomaticEpsilonLexicase",
                                ]