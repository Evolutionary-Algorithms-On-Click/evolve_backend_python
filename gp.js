const gpConfig = {
  "algorithm": "eaSimple", // DONE
  "arity": 1, // DONE
  "operators": [
    "add", "sub", "mul", "div", "neg", "cos", "sin"
  ], // DONE
  "argNames": [
    "x"
  ], // DONE
  "individualType": "PrimitiveTree", // DONE
  "expr": "genHalfAndHalf", // DONE
  "min_": 1, // DONE
  "max_": 2, // DONE
  "realFunction": "x**4 + x**3 + x**2 + x",
  "individualFunction": "initIterate", // DONE
  "populationFunction": "initRepeat", // DONE
  "selectionFunction": "selTournament", // DONE
  "tournamentSize": 3, // DONE
  "expr_mut": "genFull", // DONE
  "expr_mut_min": 0, // DONE
  "expr_mut_max": 2, // DONE
  "crossoverFunction": "cxOnePoint", // "cxSemantic" needs ['lf', 'mul', 'add', 'sub'] operators exactly. DONE
  "terminalProb": 0.1, // Only when crossoverFunction is "cxOnePointLeafBiased". Max value is 0.2 that works well. DONE
  "mutationFunction": "mutUniform", // DONE
  "mutationMode": "one", // One of "one" or "all". Only when mutationFunction is "mutEphemeral". // DONE
  "mateHeight": 17, // DONE
  "mutHeight": 17, // DONE
  "weights": [
    1.0
  ], // DONE
  
  "populationSize": 300,
  "generations": 40,
  "cxpb": 0.5,
  "mutpb": 0.1,
  "mu": 1000,
  "lambda_": 4,
  "hofSize": 1
}
