import numpy as np
import SimDataTypes
import cvxpy as cp
import gurobipy as gp


def SolveCVX_ChargingStations(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N
    DecisionVar = cp.Variable((n,n), boolean=True)
    
    # Construct Cost Vector for every decision variable
    CostMatrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                continue
            else:
                CostMatrix[i,j] = NominalPlan.PriorityCoefInCost*NominalPlan.NodesPriorities[j]**2 - NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]

    u = cp.Variable(n, integer=True) # u[i] is the position of node i in the path
    e = cp.Variable(n) # the energy when arriving at each node
    y = cp.Variable(n) # the charging time spent at each node
    
    ones = np.ones((n,1))
    # Defining the objective function
    objective = cp.Maximize(cp.sum(cp.multiply(CostMatrix, DecisionVar)) - NominalPlan.TimeCoefInCost*y@ones)

    # Defining the constraints
    constraints = []
    if NominalPlan.ReturnToBase == True:
        constraints += [DecisionVar @ ones == ones] # Every node has an entry edge
        constraints += [DecisionVar.T @ ones == ones] # Every node has an exit edge
    else:
        ones_m1 = np.ones((n-1,1))
        constraints += [DecisionVar[0,:] @ ones == 1] # Node 0 has exit edge
        constraints += [cp.sum(DecisionVar[:,0]) == 0] # Node 0 has no entry edges
        constraints += [cp.sum(DecisionVar[1:,:] @ ones) == n-2] # Number of total Exit edges is n-2 (for nodes 1-n)
        constraints += [DecisionVar[1:,:] @ ones <= 1] # Every node can have only 1 Exit edge (for nodes 1-n)
        constraints += [DecisionVar.T[1:,:] @ ones == ones_m1] # every node 1-n has entry edge
    constraints += [cp.diag(DecisionVar) == 0] # No self loops

    constraints += [u[1:] >= 2] # No node can be the first node
    constraints += [u[1:] <= n] # No node can be the last node
    constraints += [u[0] == 1] # Node 0 is the first node
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                constraints += [ u[i] + (n - 1)*DecisionVar[i, j]  <= u[j] + (n - 2) ] # No cycles

    # energy constraints:
    for i in range(0, n):
        if np.any(i == NominalPlan.ChargingStations):
            constraints += [y[i] >= 0] # Charging time is always positive
            # constraints += [y[i] <= PltParams.BatteryCapacity/NominalPlan.StationRechargePower] # Charging time is always positive
        else:
            constraints += [y[i] == 0] # Not a charging station, no charging time
    constraints += [e[0] == PowerLeft] # Initial energy
    constraints += [e[1:] >= 0] # Energy is always positive
    constraints += [e[1:] <= PltParams.BatteryCapacity] # Energy is always less than battery capacity

    # Energy constraints
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                constraints += [ e[j] + (DecisionVar[i, j]-1)*PltParams.BatteryCapacity <= e[i] - NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j] + y[i]*NominalPlan.StationRechargePower ] # Energy is always positive
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            constraints += [(DecisionVar[i, 0]-1)*PltParams.BatteryCapacity <= e[i] - NominalPlan.NodesEnergyTravel[i,0]*DecisionVar[i, 0] + y[i]*NominalPlan.StationRechargePower ]
    # Limit the Charging capacity
    for i in range(1, n):
        if np.any(i == NominalPlan.ChargingStations):
            for j in range(1, n):
                if i != j:
                    constraints += [ e[j] <= PltParams.BatteryCapacity - NominalPlan.NodesEnergyTravel[i,j]*DecisionVar[i, j]] # Can't entry a node fully charged

    # Solving the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=True, solver=cp.GUROBI)

    # Transforming the solution to a path
    X_sol = np.argwhere(DecisionVar.value==1)
    NodesTrajectory = X_sol[0].tolist()

    for i in range(1, X_sol.shape[0]):
        indx = np.argwhere(X_sol[:,0]==NodesTrajectory[-1])
        NodesTrajectory.append(X_sol[indx[0][0],1])

    return NodesTrajectory, prob.value, e.value, y.value
    # Construct the problem.













