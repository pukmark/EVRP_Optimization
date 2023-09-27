import numpy as np
import SimDataTypes
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import gurobipy as gp
import time


def SolvePyomo_ChargingStations(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N
    model = pyo.ConcreteModel()
    model.n = n
    model.nIDX = pyo.Set( initialize= range(model.n), ordered=True )
    model.DecisionVar = pyo.Var(model.nIDX, model.nIDX, domain=pyo.Binary)
    model.u = pyo.Var(model.nIDX) # u[i] is the position of node i in the path
    model.e = pyo.Var(model.nIDX) # the energy when arriving at each node
    model.y = pyo.Var(model.nIDX) # the charging time spent at each node
    
    # Construct Cost Vector for every decision variable
    CostMatrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                CostMatrix[i,j] = 9999.0
                continue
            else:
                CostMatrix[i,j] = NominalPlan.TimeCoefInCost*NominalPlan.NodesTimeOfTravel[i,j]
    model.CostMatrix = CostMatrix
    ones = np.ones((n,1))
    ones_m1 = np.ones((n-1,1))
    
    # Defining the objective function
    def objective_rule(model):
        cost = 0.0
        for i in model.nIDX:
            for j in model.nIDX:
                cost += model.CostMatrix[i, j]*model.DecisionVar[i, j]
            cost += model.y[i]*NominalPlan.TimeCoefInCost
        return cost
    
    model.cost = pyo.Objective(rule = objective_rule, sense=pyo.minimize)
    model.Constraint = pyo.ConstraintList()
    # Defining the constraints
    if NominalPlan.ReturnToBase == True:
        model.Constraint.add(expr = sum(model.DecisionVar[0,i] for i in model.nIDX) == NominalPlan.NumberOfCars)
        for i in range(1, n):
            model.Constraint.add(expr = sum(model.DecisionVar[i,j] for j in model.nIDX) == 1)
        model.Constraint.add(expr = sum(model.DecisionVar[i,0] for i in model.nIDX) == NominalPlan.NumberOfCars)
        for i in range(1, n):
            model.Constraint.add(expr = sum(model.DecisionVar[j,i] for j in model.nIDX) == 1)
    else:
        model.Constraint.add(expr = model.DecisionVar[0,:] @ ones == 1) # Node 0 has exit edge
        model.Constraint.add(expr = sum(model.DecisionVar[:,0]) == 0) # Node 0 has no entry edges
        model.Constraint.add(expr = sum(model.DecisionVar[1:,:] @ ones) == n-2) # Number of total Exit edges is n-2 (for nodes 1-n)
        model.Constraint.add(expr = model.DecisionVar[1:,:] @ ones <= 1) # Every node can have only 1 Exit edge (for nodes 1-n)
        model.Constraint.add(expr = model.DecisionVar.T[1:,:] @ ones == ones_m1) # every node 1-n has entry edge

    for i in range(1, n):
        model.Constraint.add(expr = model.u[i] >= 2)
    # model.Constraint(expr = model.u[1:] <= n)
    for i in range(1, n):
        model.Constraint.add(model.u[i] <= model.n)
    model.Constraint.add(expr = model.u[0] == 1) # Node 0 is the first node
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.Constraint.add(expr =  model.u[i] + NominalPlan.MaxNumberOfNodesPerCar*model.DecisionVar[i, j]  <= model.u[j] + NominalPlan.MaxNumberOfNodesPerCar - 1 )

    # energy constraints:
    for i in range(0, n):
        if np.any(i == NominalPlan.ChargingStations):
            model.Constraint.add(expr = model.y[i] >= 0.0) # Charging time is always positive
        else:
            model.Constraint.add(expr = model.y[i] == 0.0) # Not a charging station, no charging time
    model.Constraint.add(expr = model.e[0] == PowerLeft) # Initial energy
    for i in range(1, n):
        model.Constraint.add(expr = model.e[i] >= PltParams.MinimalSOC) # Energy is always positive
        # model.Constraint.add(expr = model.e[i] <= PltParams.BatteryCapacity) # Energy is always positive

    # Energy constraints
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                model.Constraint.add(expr =  model.e[j] + (model.DecisionVar[i, j]-1)*PltParams.BatteryCapacity <= model.e[i] + NominalPlan.NodesEnergyTravel[i,j]*model.DecisionVar[i, j] + model.y[i]*NominalPlan.StationRechargePower ) # Energy is always positive
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            model.Constraint.add(expr = PltParams.MinimalSOC + (model.DecisionVar[i, 0]-1)*PltParams.BatteryCapacity <= model.e[i] + NominalPlan.NodesEnergyTravel[i,0] + model.y[i]*NominalPlan.StationRechargePower )

    # Limit the Charging capacity
    for i in range(1, n):
        if np.any(i == NominalPlan.ChargingStations):
            for j in range(1, n):
                if i != j:
                    model.Constraint.add(expr =  model.e[j] <= PltParams.BatteryCapacity + NominalPlan.NodesEnergyTravel[i,j]*model.DecisionVar[i, j]) # Can't entry a node fully charged

    # Solving the problem

    # optimizer = pyo.SolverFactory('mindtpy')
    # optimizer.options["threads"] = 14
    # optimizer.options["TimeLimit"] = 100
    # optimizer.options["MIPGap"] = 0.02
    # results = optimizer.solve(model,
    #                           mip_solver='gurobi',
    #                           nlp_solver='ipopt',
    #                           solver_tee=True,
    #                           time_limit=10,
    #                           tee=True)


    optimizer = pyo.SolverFactory('scip', executable="/home/optimization/scip/build/bin/scip")
    # optimizer.options["threads"] = 16
    # optimizer.options["TimeLimit"] = 60
    # optimizer.options["MIPGap"] = 0.08
    results = optimizer.solve(model, tee=True)


    # Transforming the solution to a path
    DecisionVar = np.asarray([[model.DecisionVar[t,i]() for i in model.nIDX] for t in model.nIDX])
    e = np.asarray([model.e[i]() for i in model.nIDX])
    y = np.asarray([model.y[i]() for i in model.nIDX])
    X_sol = np.argwhere(DecisionVar==1)
    NodesTrajectory = np.zeros((n, NominalPlan.NumberOfCars), dtype=int)

    for m in range(NominalPlan.NumberOfCars):
        NodesTrajectory[0,m] = 0
        NodesTrajectory[1,m] = X_sol[m,1]
        i = 2
        while True:
            NodesTrajectory[i,m] = X_sol[np.argwhere(X_sol[:,0]==NodesTrajectory[i-1,m]),1]
            if NodesTrajectory[i,m] == 0:
                break
            i += 1

    return NodesTrajectory, model.cost(), e, y
    # Construct the problem.




def SolvePyomo_ChargingStations_MinMax(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost):
    n = NominalPlan.N
    model = pyo.ConcreteModel()
    model.n = n
    model.nIDX = pyo.Set( initialize= range(model.n), ordered=True)
    model.IDX = pyo.Set( initialize= range(0), ordered=True )
    model.DecisionVar = pyo.Var(model.nIDX, model.nIDX, domain=pyo.Binary)
    model.u = pyo.Var(model.nIDX) # u[i] is the position of node i in the path
    model.e = pyo.Var(model.nIDX) # the energy when arriving at each node
    model.y = pyo.Var(model.nIDX) # the charging time spent at each node
    model.T = pyo.Var(model.nIDX) # Time elapsed when arriving at each node
    model.Tf = pyo.Var() # Final time
        
    ones = np.ones((n,1))
    ones_m1 = np.ones((n-1,1))
    # Defining the objective function
    if NominalPlan.ReturnToBase == True:
        model.cost = pyo.Objective(rule = model.Tf, sense=pyo.minimize)
    else:
        model.cost = pyo.Objective(rule = max(model.T), sense=pyo.minimize)

    # Defining the objective function  
    model.Constraint = pyo.ConstraintList()
    for i in range(n):
        model.Constraint.add(expr = (model.DecisionVar[i,i] == 0)) # No self loops
    # Defining the constraints
    if NominalPlan.ReturnToBase == True:
        model.Constraint.add(expr = sum(model.DecisionVar[0,i] for i in model.nIDX) == NominalPlan.NumberOfCars)
        for i in range(1, n):
            model.Constraint.add(expr = sum(model.DecisionVar[i,j] for j in model.nIDX) == 1)
        model.Constraint.add(expr = sum(model.DecisionVar[i,0] for i in model.nIDX) == NominalPlan.NumberOfCars)
        for i in range(1, n):
            model.Constraint.add(expr = sum(model.DecisionVar[j,i] for j in model.nIDX) == 1)
    else:
        model.Constraint.add(expr = model.DecisionVar[0,:] @ ones == 1) # Node 0 has exit edge
        model.Constraint.add(expr = sum(model.DecisionVar[:,0]) == 0) # Node 0 has no entry edges
        model.Constraint.add(expr = sum(model.DecisionVar[1:,:] @ ones) == n-2) # Number of total Exit edges is n-2 (for nodes 1-n)
        model.Constraint.add(expr = model.DecisionVar[1:,:] @ ones <= 1) # Every node can have only 1 Exit edge (for nodes 1-n)
        model.Constraint.add(expr = model.DecisionVar.T[1:,:] @ ones == ones_m1) # every node 1-n has entry edge

    for i in range(1, n):
        model.Constraint.add(expr = model.u[i] >= 2)
    # model.Constraint(expr = model.u[1:] <= n)
    for i in range(1, n):
        model.Constraint.add(model.u[i] <= model.n)
    model.Constraint.add(expr = model.u[0] == 1) # Node 0 is the first node
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                model.Constraint.add(expr =  model.u[i] + NominalPlan.MaxNumberOfNodesPerCar*model.DecisionVar[i, j]  <= model.u[j] + NominalPlan.MaxNumberOfNodesPerCar - 1 )

    # energy constraints:
    for i in range(0, n):
        if np.any(i == NominalPlan.ChargingStations):
            model.Constraint.add(expr = model.y[i] >= 0.0) # Charging time is always positive
        else:
            model.Constraint.add(expr = model.y[i] == 0.0) # Not a charging station, no charging time
    model.Constraint.add(expr = model.e[0] == PowerLeft) # Initial energy
    for i in range(1, n):
        model.Constraint.add(expr = model.e[i] >= PltParams.MinimalSOC) # Energy is always positive
        # model.Constraint.add(expr = model.e[i] <= PltParams.BatteryCapacity) # Max Energy is BatteryCapacity

    # Energy constraints
    # This implemets the energy constraints for the case of charging stations by the following dynamic model:
    # the charging rate is charging state dependent, and the charging rate is a function of the current energy level
    # the higher the energy level, the lower the charging rate (Chargeing rate at 100% is 0.5*NominalRechargeRate):
    # e_dot = StationRechargeRate * (1 - 0.5*e/BatteryCapacity)
    # by integrating the above equation, we get the following equation:
    # e(j) = StationRechargeRate/FullRechargeRateFactor + (e(i) - StationRechargeRate/FullRechargeRateFactor)*exp(-FullRechargeRateFactor/BatteryCapacity*NominalRechargeRate*y(i))
    a1 = PltParams.BatteryCapacity/PltParams.FullRechargeRateFactor
    a2 = PltParams.FullRechargeRateFactor*NominalPlan.StationRechargePower/PltParams.BatteryCapacity
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                if PltParams.RechargeModel == 'ConstantRate':
                    model.Constraint.add(expr =  model.e[j] + (model.DecisionVar[i,j]-1)*PltParams.BatteryCapacity <= model.e[i] + NominalPlan.NodesEnergyTravel[i,j]*model.DecisionVar[i,j] + model.y[i]*NominalPlan.StationRechargePower ) # Energy is always positive
                else:
                    model.Constraint.add(expr =  model.e[j] + (model.DecisionVar[i, j]-1)*PltParams.BatteryCapacity <= NominalPlan.NodesEnergyTravel[i,j]*model.DecisionVar[i, j] + (a1 + (model.e[i]-a1)*pyo.exp(-a2*model.y[i])) ) # Energy is always positive
    if NominalPlan.ReturnToBase == True:
        for i in range(1, n):
            if PltParams.RechargeModel == 'ConstantRate':
                model.Constraint.add(expr = PltParams.MinimalSOC + (model.DecisionVar[i,0]-1)*PltParams.BatteryCapacity <= model.e[i] + NominalPlan.NodesEnergyTravel[i,0] + model.y[i]*NominalPlan.StationRechargePower )
            else:
                model.Constraint.add(expr = PltParams.MinimalSOC + (model.DecisionVar[i, 0]-1)*PltParams.BatteryCapacity <= NominalPlan.NodesEnergyTravel[i,0] + (a1 + (model.e[i]-a1)*pyo.exp(-a2*model.y[i])) )

    # Limit the charging capacity
    for i in NominalPlan.ChargingStations[:,0]:
        if PltParams.RechargeModel == 'ConstantRate':
            model.Constraint.add(expr = model.e[i] + model.y[i]*NominalPlan.StationRechargePower <= PltParams.BatteryCapacity) # Can't entry a node fully charged
        else:
            model.Constraint.add(expr =  a1 + (model.e[i]-a1)*pyo.exp(-a2*model.y[i]) <= PltParams.BatteryCapacity) # Can't entry a node fully charged


    # for i in range(1, n):
    #     if np.any(i == NominalPlan.ChargingStations):
    #         for j in range(1, n):
    #             if i != j:
    #                 model.Constraint.add(expr =  model.e[j] <= PltParams.BatteryCapacity + NominalPlan.NodesEnergyTravel[i,j]*model.DecisionVar[i, j]) # Can't entry a node fully charged

    # Time Dynamics (constraints):
    MaxT = np.max(NominalPlan.NodesTimeOfTravel)*n
    model.Constraint.add(expr = model.T[0] == 0.0)
    for i in range(0, n):
        for j in range(1, n):
            if i != j:
                model.Constraint.add(expr = model.T[j] + (1-model.DecisionVar[i, j])*MaxT >= model.T[i] + model.DecisionVar[i, j]*NominalPlan.NodesTimeOfTravel[i,j] + model.y[i] )
    if NominalPlan.ReturnToBase == True:
        # model.Constraint.add(expr = model.Tf >= 40.0)
        L = MaxT*n
        for i in range(1, n):
                model.Constraint.add(expr = model.Tf + (1-model.DecisionVar[i, 0])*L >= model.T[i] + NominalPlan.NodesTimeOfTravel[i,0] + model.y[i] )

    # model.display()
    # Solving the problem
    optimizer = pyo.SolverFactory('mindtpy')
    results = optimizer.solve(model,
                              mip_solver='gurobi',
                              nlp_solver='ipopt',
                              mip_solver_tee=True,
                              time_limit=120,
                            #   threads=14,
                              mip_solver_mipgap=0.01,
                            #   nlp_solver_args={'timelimit': 60},
                              mip_solver_args={'timelimit': 30},
                            #   iteration_limit=5,
                              tee=True)
    
    # Transforming the solution to a path
    DecisionVar = np.asarray([[model.DecisionVar[t,i]() for i in model.nIDX] for t in model.nIDX])
    e = np.asarray([model.e[i]() for i in model.nIDX])
    y = np.asarray([model.y[i]() for i in model.nIDX])
    X_sol = np.argwhere(np.round(DecisionVar)==1)
    NodesTrajectory = np.zeros((n, NominalPlan.NumberOfCars), dtype=int)

    for m in range(NominalPlan.NumberOfCars):
        NodesTrajectory[0,m] = 0
        NodesTrajectory[1,m] = X_sol[m,1]
        i = 2
        while True:
            NodesTrajectory[i,m] = X_sol[np.argwhere(X_sol[:,0]==NodesTrajectory[i-1,m]),1]
            if NodesTrajectory[i,m] == 0:
                break
            i += 1

    return NodesTrajectory, model.cost(), e, y
    # Construct the problem.

