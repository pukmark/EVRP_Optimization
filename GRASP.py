import numpy as np
import SimDataTypes



def greedyRandomConstruction(NominalPlan: SimDataTypes.NominalPlanning):
    
    Traj = []
    CustomersNodes = set(range(NominalPlan.N)) - set(NominalPlan.CarsInDepots) - set(NominalPlan.ChargingStations)
    NodesTimeOfTravel = NominalPlan.NodesTimeOfTravel + NominalPlan.TimeAlpha * NominalPlan.TravelSigma
    # Randomize First Customer
    for iDepot in NominalPlan.CarsInDepots:
        Tour = [iDepot]
        iCustomer = np.random.choice(list(CustomersNodes))
        CustomersNodes.remove(iCustomer)
        Tour.append(iCustomer)
        Traj.append(Tour)
    # Greedy Construction
    while len(CustomersNodes) > 0:
        for iCar in range(len(Traj)):
            if len(CustomersNodes) == 0:
                break
            iCustomer = np.argmin(NodesTimeOfTravel[Traj[iCar][-1], list(CustomersNodes)])
            Traj[iCar].append(list(CustomersNodes)[iCustomer])
            CustomersNodes.remove(list(CustomersNodes)[iCustomer])
    # Add Return to Depot to End of Traj
    for iCar in range(len(Traj)):
        Traj[iCar].append(Traj[iCar][0])
    return Traj

def repairCapacityTraj(Traj, NominalPlan):
    CurLoad = NominalPlan.LoadCapacity
    for iCar in range(len(Traj)):
        Traj_CVRP = [Traj[iCar][0]]
        for iNode in range(1,len(Traj[iCar])):
            CurLoad -= NominalPlan.LoadDemand[Traj[iCar][iNode]]
            if CurLoad >= 0:
                Traj_CVRP.append(Traj[iCar][iNode])
            else: # If Load is Negative, Return to Depot
                Traj_CVRP.append(Traj[iCar][0])
                Traj_CVRP.append(Traj[iCar][iNode])
                CurLoad = NominalPlan.LoadCapacity - NominalPlan.LoadDemand[Traj[iCar][iNode]]
        Traj[iCar] = Traj_CVRP
    return Traj


def repairEnergyTraj(Traj, NominalPlan):

    for iCar in range(len(Traj)):
        Traj_EVRP = [Traj[iCar][0]]
        CurEnergy = NominalPlan.BatteryCapacity
        CurEnergySigma = 0
        for i in range(1,len(Traj[iCar])):
            CurNode = Traj_EVRP[-1]
            NextNode = Traj[iCar][i]
            if NextNode in NominalPlan.ChargingStations or NextNode in NominalPlan.CarsInDepots:
                NextEnergy = NominalPlan.BatteryCapacity
            else:
                NextEnergy = CurEnergy + NominalPlan.NodesEnergyTravel[CurNode,NextNode]
            
            Next_iCS = np.argmin(NominalPlan.NodesTimeOfTravel[NextNode, NominalPlan.ChargingStations])
            Next_CS = NominalPlan.ChargingStations[Next_iCS]
            Reachable = (CurEnergy+NominalPlan.NodesEnergyTravel[CurNode,NextNode] >=  NominalPlan.EnergyAlpha * np.sqrt(CurEnergySigma**2 + NominalPlan.NodesEnergyTravelSigma2[CurNode,NextNode]))
            Stuck = (NextEnergy+NominalPlan.NodesEnergyTravel[NextNode,Next_CS] < NominalPlan.EnergyAlpha * np.sqrt(CurEnergySigma**2 + NominalPlan.NodesEnergyTravelSigma2[CurNode,NextNode]+ + NominalPlan.NodesEnergyTravelSigma2[NextNode,Next_CS]))
            if (Reachable) and (not Stuck):
                Traj_EVRP.append(NextNode)
                CurEnergy = NextEnergy
                CurEnergySigma = np.sqrt(CurEnergySigma**2 + NominalPlan.NodesEnergyTravelSigma2[CurNode,NextNode])
            else:
                Cur_iCS = np.argmin(NominalPlan.NodesTimeOfTravel[CurNode, NominalPlan.ChargingStations])
                Cur_CS = NominalPlan.ChargingStations[Cur_iCS]
                Traj_EVRP.append(Cur_CS)
                CurEnergy = NominalPlan.BatteryCapacity
                CurEnergySigma = np.sqrt(CurEnergySigma**2 + NominalPlan.NodesEnergyTravelSigma2[CurNode,Cur_CS])
                Traj_EVRP.append(NextNode)
                CurEnergy += NominalPlan.NodesEnergyTravel[Cur_CS,NextNode]
                CurEnergySigma = np.sqrt(CurEnergySigma**2 + NominalPlan.NodesEnergyTravelSigma2[Cur_CS,NextNode])


        Traj[iCar] = Traj_EVRP
    return Traj

def localSearch(Traj, NominalPlan):

    # for iCar in range(len(Traj)):

    return Traj

def computeCost(Traj, NominalPlan):
    Cost = 0
    for iCar in range(len(Traj)):
        for i in range(1,len(Traj[iCar])):
            Cost += NominalPlan.NodesTimeOfTravel[Traj[iCar][i-1],Traj[iCar][i]]

    return Cost


def verifyEnergyTraj(Traj, NominalPlan):
    for iCar in range(len(Traj)):
        CurEnergy = NominalPlan.BatteryCapacity
        for i in range(1,len(Traj[iCar])):
            CurNode = Traj[iCar][i-1]
            NextNode = Traj[iCar][i]
            CurEnergy += NominalPlan.NodesEnergyTravel[CurNode,NextNode]
            if CurEnergy < 0:
                return False
            if NextNode in NominalPlan.ChargingStations or NextNode in NominalPlan.CarsInDepots:
                CurEnergy = NominalPlan.BatteryCapacity
    return True



def GRASP(NominalPlan: SimDataTypes.NominalPlanning):
    Traj_Best = []
    Cost_Best = np.inf
    for i in range(0, 10000):
        Traj = greedyRandomConstruction(NominalPlan)
        Traj = repairCapacityTraj(Traj, NominalPlan)
        Traj = repairEnergyTraj(Traj, NominalPlan)
        Feas = verifyEnergyTraj(Traj, NominalPlan)
        Traj = localSearch(Traj, NominalPlan)
        Cost = computeCost(Traj, NominalPlan)
        
        if Cost < Cost_Best:
            Cost_Best = Cost
            Traj_Best = Traj
    
    ChargingTime = np.zeros((NominalPlan.N,NominalPlan.NumberOfCars))

    Traj_out = np.zeros((NominalPlan.N,NominalPlan.NumberOfCars), dtype=int)
    for iCar in range(len(Traj_Best)):
        for i in range(1,len(Traj_Best[iCar])):
            Traj_out[i,iCar] = Traj_Best[iCar][i]
            if Traj_Best[iCar][i] in NominalPlan.ChargingStations:
                ChargingTime[Traj_Best[iCar][i],iCar] = 1.9e-3

    return Traj_out, Cost_Best, ChargingTime