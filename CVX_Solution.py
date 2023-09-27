import numpy as np
import SimDataTypes
import cvxpy as cp
import gurobipy as gp


def SolveCVX(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N
    DecisionVar = cp.Variable((n,n), boolean=True)
    
    # Construct Cost Vector for every decision variable
    CostVector = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            else:
                CostVector[i,j] = NominalPlan.PriorityCoefInCost*NominalPlan.NodesPriorities[j]**2 - NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]

    u = cp.Variable(n, integer=True)
    

    # Defining the objective function
    objective = cp.Maximize(cp.sum(cp.multiply(CostVector, DecisionVar)))

    # Defining the constraints
    ones = np.ones((n,1))
    constraints = []
    if NominalPlan.ReturnToBase == True:
        constraints += [DecisionVar @ ones == ones]
        constraints += [DecisionVar.T @ ones == ones]
    else:
        ones_m1 = np.ones((n-1,1))
        constraints += [DecisionVar[0,:] @ ones == 1] # Node 0 has exit edge
        constraints += [cp.sum(DecisionVar[:,0]) == 0] # Node 0 has no entry edges
        constraints += [cp.sum(DecisionVar[1:,:] @ ones) == n-2] # Number of total Exit edges is n-2 (for nodes 1-n)
        constraints += [DecisionVar[1:,:] @ ones <= 1] # Every node can have only 1 Exit edge (for nodes 1-n)
        constraints += [DecisionVar.T[1:,:] @ ones == ones_m1] # every node 1-n has entry edge
    constraints += [cp.diag(DecisionVar) == 0]
    constraints += [u[1:] >= 2]
    constraints += [u[1:] <= n]
    constraints += [u[0] == 1]

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                constraints += [ u[i] + (n - 1)*DecisionVar[i, j]  <= u[j] + (n - 2) ]

    # Solving the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False, solver=cp.GUROBI)

    # Transforming the solution to a path
    X_sol = np.argwhere(DecisionVar.value==1)
    orden = X_sol[0].tolist()

    for i in range(1, X_sol.shape[0]):
        indx = np.argwhere(X_sol[:,0]==orden[-1])
        orden.append(X_sol[indx[0][0],1])

    return orden, prob.value, 0.
    # Construct the problem.













