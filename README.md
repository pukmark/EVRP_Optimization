TO run this code, download the entire repository;
git clone git@github.com:pukmark/EVRP_Optimization.git

to run the code:

choose the instances to run by commenting out/in the relevent instances names.
to run a randomized scenario, comment out all named instances.

The parameters that control the solution type:
    SolutionProbabilityTimeReliability = 0.9
    SolutionProbabilityEnergyReliability = 0.999
    DeterministicProblem = 0 # 0 - stochastic problem, 1- deterministic problem
    CostFunctionType = 1 # 1: Min Sum of Time Travelled, 2: ,Min Max Time Travelled by any car
    SolverType = 'Recursive' # 'Gurobi' or 'Recursive' or 'Gurobi_NoClustering' or 'GRASP'
    ClusteringMethod = "Max_EigenvalueN" # "Max_EigenvalueN" or "Frobenius" or "Sum_AbsEigenvalue" or "SumSqr_AbsEigenvalue" or "Mean_MaxRow" or "PartialMax_Eigenvalue" or "Greedy_Method"
    SolveAlsoWithGurobi = 1 # Solve the problem as MIP using the heuristics solution as initial guess
