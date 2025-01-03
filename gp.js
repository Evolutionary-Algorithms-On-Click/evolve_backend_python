const gpConfig = {
  "algorithm": "eaSimple",
  "arity": 1,
  "operators": [
    "add", "sub", "mul", "div", "neg", "cos", "sin"
  ],
  "argNames": [
    "x"
  ],
  "individualType": "PrimitiveTree",
  "expr": "genHalfAndHalf",
  "min_": 1,
  "max_": 2,
  "realFunction": "x**4 + x**3 + x**2 + x",
  "individualFunction": "initIterate",
  "populationFunction": "initRepeat",
  "selectionFunction": "selTournament",
  "tournamentSize": 3,
  "expr_mut": "genFull",
  "expr_mut_min": 0,
  "expr_mut_max": 2,
  "crossoverFunction": "cxOnePoint", // "cxSemantic" needs ['lf', 'mul', 'add', 'sub'] operators exactly.
  "terminalProb": 0.1, // Only when crossoverFunction is "cxOnePointLeafBiased". Max value is 0.2 that works well.
  "mutationFunction": "mutUniform",
  "mateHeight": 17,
  "mutHeight": 17,
  "weights": [
    1.0
  ],
  "populationSize": 300,
  "generations": 40,
  "cxpb": 0.5,
  "mutpb": 0.1,
  "mu": 1000,
  "lambda_": 4,
  "individualSize": 10,
  "hofSize": 1
}
