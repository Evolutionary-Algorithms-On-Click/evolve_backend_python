from functools import reduce

class EvalFunctions:
    def evalOneMax(individual):
        return sum(individual),

    def evalProduct(individual):
        return reduce(lambda x, y: x*y, individual),

    def evalDifference(individual):
        return reduce(lambda x, y: x-y, individual),
    
    def evalKnapsack(individual, values, weights, maxWeight):
        value = 0
        weight = 0
        for i in range(len(individual)):
            value += individual[i] * values[i]
            weight += individual[i] * weights[i]
        if weight > maxWeight:
            return 0,
        return value, 

    def evalTSP(individual, distanceMatrix):
        distance = 0
        for i in range(len(individual)):
            distance += distanceMatrix[individual[i-1]][individual[i]]
        return distance,

    def evalNQueens(individual):
        size = len(individual)
        diagonal1 = [0] * (2*size)
        diagonal2 = [0] * (2*size)
        for i in range(size):
            diagonal1[i+individual[i]] += 1
            diagonal2[size-i+individual[i]-1] += 1
        conflicts = 0
        for i in range(2*size):
            if diagonal1[i] > 1:
                conflicts += diagonal1[i] - 1
            if diagonal2[i] > 1:
                conflicts += diagonal2[i] - 1
        return conflicts,

    def evalFunction(individual, evalFunction, *args):
        return eval(evalFunction)(individual, *args)
    
    
