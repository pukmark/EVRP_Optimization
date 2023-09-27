import numpy as np
import SimDataTypes
import cvxpy as cp
import gurobipy as gp
import itertools

def SolveCVX(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N
    DecisionVar = cp.Variable((n,n), boolean=True)

    # Construct Cost Vector for every decision variable
    CostMatrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                CostMatrix[i,j] = -999.0
                continue
            else:
                CostMatrix[i,j] = NominalPlan.PriorityCoefInCost*NominalPlan.NodesPriorities[j]**2 - NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]


    # Defining the objective function
    objective = cp.Maximize(cp.sum(cp.multiply(CostMatrix, DecisionVar)))

    # Defining the constraints
    ones = np.ones((n,1))
    ones_m1 = np.ones((n-1,1))
    constraints = []
    if NominalPlan.ReturnToBase == True:
        constraints += [DecisionVar[0,:] @ ones == NominalPlan.NumberOfCars]
        constraints += [DecisionVar[1:,:] @ ones == ones_m1]
        constraints += [DecisionVar.T[0,:] @ ones == NominalPlan.NumberOfCars]
        constraints += [DecisionVar.T[1:,:] @ ones == ones_m1]
    else:
        constraints += [DecisionVar[0,:] @ ones == NominalPlan.NumberOfCars] # Node 0 has exit edge
        constraints += [DecisionVar.T[0,:] @ ones == 0] # Node 0 has no entry edges
        constraints += [cp.sum(DecisionVar[1:,:] @ ones) == n-2] # Number of total Exit edges is n-2 (for nodes 1-n)
        constraints += [DecisionVar[1:,:] @ ones <= 1] # Every node can have only 1 Exit edge (for nodes 1-n)
        constraints += [DecisionVar.T[1:,:] @ ones == ones_m1] # every node 1-n has entry edge

    u = cp.Variable(n)
    constraints += [u[1:] >= 2]
    constraints += [u[1:] <= n]
    constraints += [u[0] == 1]
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                constraints += [ u[i] + NominalPlan.MaxNumberOfNodesPerCar*DecisionVar[i, j] <= u[j] + NominalPlan.MaxNumberOfNodesPerCar-1 ]


    # Dantzig–Fulkerson–Johnson formulation for subtour elimination
    subtourlen = (2,3,4,n-3,n-2,n-1)
    for m in subtourlen:
        premut = list(itertools.combinations(range(n),m))
        if len(premut)>10000:
            continue
        for i in range(len(premut)):
            ones = np.zeros((n,n))
            for j in range(m-1):
                ones[premut[i][j],premut[i][j+1]] = 1
            ones[premut[i][m-1],premut[i][0]] = 1
            constraints += [cp.sum(cp.multiply(ones, DecisionVar)) <= m-1]

    

    # Solving the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.GUROBI)

    # Transforming the solution to a path
    X_sol = np.argwhere(np.round(DecisionVar.value)==1)
    NodesTrajectory = np.zeros((n+1, NominalPlan.NumberOfCars), dtype=int)

    for m in range(NominalPlan.NumberOfCars):
        NodesTrajectory[0,m] = 0
        NodesTrajectory[1,m] = X_sol[m,1]
        i = 2
        while True:
            NodesTrajectory[i,m] = X_sol[np.argwhere(X_sol[:,0]==NodesTrajectory[i-1,m]),1]
            if NodesTrajectory[i,m] == 0:
                break
            i += 1

    return NodesTrajectory, prob.value, 0.
    # Construct the problem.













