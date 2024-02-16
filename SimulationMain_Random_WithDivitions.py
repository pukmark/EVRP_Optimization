import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as DataTypes
from Gurobi_Convex_ChargingStations import *
from DivideToGroups import *
from GRASP import *
from RecursiveOptimalSolution_ChargingStations import *
import os
import time
import seaborn as sns
import matplotlib as mpl

os.system('cls' if os.name == 'nt' else 'clear')


def main(iSeed=10, iplot = 0):
    np.random.seed(iSeed)


    # Define Platform Parameters:
    PltParams = DataTypes.PlatformParams()
    # Define Simulation Parameters:
    #############################$
    ScenarioFileNames = []

    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/E-n22-k4.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/E-n23-k3.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/E-n30-k3.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/E-n33-k4.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/E-n51-k5.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/E-n76-k7.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/E-n101-k8.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n143-k7.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n214-k11.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n351-k40.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n459-k26.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n573-k30.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n685-k75.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n749-k98.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n819-k171.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n916-k207.evrp')
    # ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/X-n1001-k43.evrp')
    # ScenarioFileNames.append('./VRP_Instances/Uchoa/X-n101-k25.evrp')
    # ScenarioFileNames.append('./VRP_Instances/Uchoa/X-n251-k28.evrp')
    # ScenarioFileNames.append('./VRP_Instances/Uchoa/X-n336-k84.evrp')
    # ScenarioFileNames.append('./VRP_Instances/Uchoa/X-n502-k39.evrp')
    # ScenarioFileNames.append('./VRP_Instances/Uchoa/X-n1001-k43.evrp')
    # ScenarioFileNames.append('./VRP_Instances/Leuven1.evrp')
    # Number of Nodes (Rnadomized Scenario):
    try:
        os.remove("./Groups.txt")
    except:
        pass

    N = 22
    CarsInDepots = [0,0,1] # Number of Cars per depot
    NumberOfCars = len(CarsInDepots)
    NumberOfDepots = len(np.unique(CarsInDepots))
    MaxNumberOfNodesPerCar = int(3.0*(N-NumberOfDepots)/(NumberOfCars)) if NumberOfCars > 1 else N
    MaxNodesToSolver = 100
    PltParams.BatteryCapacity = 100.0
    PltParams.LoadCapacity = 100
    SolutionProbabilityTimeReliability = 0.9
    SolutionProbabilityEnergyReliability = 0.999
    DeterministicProblem = 0
    MinLoadPerNode, MaxLoadPerNode = 10, 20
    MaxMissionTime = 120
    ReturnToBase = True
    MustVisitAllNodes = True
    CostFunctionType = 1 # 1: Min Sum of Time Travelled, 2: ,Min Max Time Travelled by any car
    MaxTotalTimePerVehicle  = 200.0
    PltParams.RechargeModel = 'ConstantRate' # 'ExponentialRate' or 'ConstantRate'
    SolverType = 'Recursive' # 'Gurobi' or 'Recursive' or 'Gurobi_NoClustering' or 'GRASP'
    ClusteringMethod = "Max_EigenvalueN" # "Max_EigenvalueN" or "Frobenius" or "Sum_AbsEigenvalue" or "SumSqr_AbsEigenvalue" or "Mean_MaxRow" or "PartialMax_Eigenvalue" or "Greedy_Method"
    MaxCalcTimeFromUpdate = 10.0 # Max time to calculate the solution from the last update [sec]
    SolveAlsoWithGurobi = 1
    ##############################$
    # If MustVisitAllNodes is True, then the mission time is set to a large number
    if MustVisitAllNodes == True:
        MaxMissionTime = 10e5
    if SolverType == 'Gurobi_NoClustering':
        MaxCalcTimeFromUpdate = 3000.0

    # Map Size
    Xmin, Xmax = -100, 100
    Ymin, Ymax = -100, 100

    if len(ScenarioFileNames) == 0:
        ScenarioFileNames = ['RandomScenario']

    for ScenarioFileName in ScenarioFileNames:
        M = np.sum(NumberOfCars)
        # Platform Parameters:
        PltParams.Vmax = 10 # Platform Max Speed
        PltParams.MinVelReductionCoef, PltParams.MaxVelReductionCoef = 0.0, 0.75 # min/max speed reduction factor for node2node travel
        PltParams.VelEnergyConsumptionCoef = 0.047 # Power consumption due to velocity = VelEnergyConsumptionCoef* Vel^2
        PltParams.VelConstPowerConsumption = 0.05
        ## Total Power to travel From Node i to Node J = (ConstPowerConsumption + VelEnergyConsumptionCoef* Vel^2)*Time_i2j
        PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask = 2, 10
        PltParams.MinTimePerTask, PltParams.MaxTimePerTask = 1, 5
        # PltParams.RechargePowerPerDay = 5
        PltParams.MinimalSOC = 0.0*PltParams.BatteryCapacity
        PltParams.FullRechargeRateFactor = 0.25

        ## Randomize The Nodes Locations:
        NodesPosition = np.block([np.random.uniform(Xmin,Xmax, size=(N,1)), np.random.uniform(Ymin,Ymax, size=(N,1))])
        # NodesPosition[0,0] = 0.0; NodesPosition[0,1] = 0.0

        # Set the Nomianl Time of Travel between any 2 nodes as the distance between
        # the nodes divided by the estimated travel velocity:
        NominalPlan = DataTypes.NominalPlanning(N)
        NominalPlan.NodesPosition = NodesPosition
        NominalPlan.NumberOfDepots = NumberOfDepots
        NominalPlan.NumberOfCars = NumberOfCars
        NominalPlan.CarsInDepots = CarsInDepots
        for i in range(N):
            NominalPlan.NodesVelocity[i,i+1:] = 6 #np.random.uniform(PltParams.Vmax*(1.0-PltParams.MaxVelReductionCoef), PltParams.Vmax*(1.0-PltParams.MinVelReductionCoef), size=(1,N-i-1))
            NominalPlan.NodesVelocity[i+1:,i] = NominalPlan.NodesVelocity[i,i+1:].T
            NominalPlan.LoadDemand[i] = 0 
            for j in range(i,N):
                if i==j: continue
                NominalPlan.NodesDistance[i,j] = np.linalg.norm(np.array([NodesPosition[i,0]-NodesPosition[j,0], NodesPosition[i,1]-NodesPosition[j,1]]))
                NominalPlan.NodesDistance[j,i] = NominalPlan.NodesDistance[i,j]
                NominalPlan.NodesTimeOfTravel[i,j] = NominalPlan.NodesDistance[i,j] / NominalPlan.NodesVelocity[i,j]
                NominalPlan.NodesTimeOfTravel[j,i] = NominalPlan.NodesTimeOfTravel[i,j]
                NominalPlan.TravelSigma[i,j] = np.random.uniform(0.05*NominalPlan.NodesTimeOfTravel[i,j], 0.3*NominalPlan.NodesTimeOfTravel[i,j],1)[0]
                NominalPlan.TravelSigma[j,i] = NominalPlan.TravelSigma[i,j]
                NominalPlan.NodesEnergyTravel[i,j] = -NominalPlan.NodesTimeOfTravel[i,j] * (PltParams.VelConstPowerConsumption + PltParams.VelEnergyConsumptionCoef*NominalPlan.NodesVelocity[i,j]**2)
                NominalPlan.NodesEnergyTravel[j,i] = NominalPlan.NodesEnergyTravel[i,j]
                NominalPlan.NodesEnergyTravelSigma[i,j] = np.abs(np.random.uniform(0.05*NominalPlan.NodesEnergyTravel[i,j], 0.3*NominalPlan.NodesEnergyTravel[i,j],1))[0]
                NominalPlan.NodesEnergyTravelSigma[j,i] = NominalPlan.NodesEnergyTravelSigma[i,j]

        NominalPlan.NodesEnergyTravelSigma2 = NominalPlan.NodesEnergyTravelSigma**2
        NominalPlan.TravelSigma2 = NominalPlan.TravelSigma**2

        NominalPlan.LoadCapacity = PltParams.LoadCapacity
        NominalPlan.BatteryCapacity = PltParams.BatteryCapacity   

        ## Calculate Nominal Time to spend and Energy Consumption for task in Node i
        NominalPlan.NodesTaskTime = np.random.uniform(PltParams.MinTimePerTask, PltParams.MaxTimePerTask, size=(N,1))
        NominalPlan.NodesTaskPower = np.random.uniform(PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask, size=(N,1))

        if DeterministicProblem == True:
            NominalPlan.TravelSigma = np.zeros((N,N))
            NominalPlan.TravelSigma2 = np.zeros((N,N))
            NominalPlan.NodesEnergyTravelSigma = np.zeros((N,N))
            NominalPlan.NodesEnergyTravelSigma2 = np.zeros((N,N))

        ## Nodes Task Prioroties:
        NominalPlan.N = N
        InitialChargeStage = 1.0 * PltParams.BatteryCapacity
        NominalPlan.TimeCoefInCost = 1.0
        NominalPlan.PriorityCoefInCost = 100.0 if MustVisitAllNodes == False else 0.0
        NominalPlan.ReturnToBase = ReturnToBase
        NominalPlan.MustVisitAllNodes = MustVisitAllNodes
        NominalPlan.MaxNumberOfNodesPerCar = MaxNumberOfNodesPerCar
        NominalPlan.SolutionProbabilityTimeReliability = SolutionProbabilityTimeReliability
        NominalPlan.SolutionProbabilityEnergyReliability = SolutionProbabilityEnergyReliability
        NominalPlan.CostFunctionType = CostFunctionType
        NominalPlan.MaxTotalTimePerVehicle = MaxTotalTimePerVehicle
        NominalPlan.EnergyAlpha = norm.ppf(SolutionProbabilityEnergyReliability)
        NominalPlan.TimeAlpha = norm.ppf(SolutionProbabilityTimeReliability)
        NominalPlan.InitialChargeStage = InitialChargeStage
        NominalPlan.ChargingStations = list()
        NominalPlan.StationRechargePower = 3

        # Divide the nodes to groups (Clustering) - for charging stations position:
        if ScenarioFileName == 'RandomScenario':
            while True:
                NodesGroups = DivideNodesToGroups(NominalPlan, 'Sum_AbsEigenvalue', ClusterSubGroups=False, isplot=iplot>=3)
                if len(NodesGroups) == 0:
                    print('Can"t find initial solution for this problem. Adding more trucks... ', NumberOfCars+1)
                    # add more trucks:
                    NumberOfCars += 1
                    M = NumberOfCars
                    NominalPlan.NumberOfCars = NumberOfCars
                    CarsInDepots = np.append(CarsInDepots, 0)
                    NominalPlan.CarsInDepots = CarsInDepots
                    continue
                break
            
            ## Charging Stations and Load Demand:
            for i in range (M):
                NumGroupChargers = min(2,int(np.ceil(len(NodesGroups[i])/10)))
                GroupCharging = []
                if len(NodesGroups[i]) < NumGroupChargers+2:
                    continue
                rand_CS = np.random.randint(1,len(NodesGroups[i]),size=(NumGroupChargers,)).tolist()
                for j in rand_CS:
                    GroupCharging.append(NodesGroups[i][j])
                while len(np.unique(GroupCharging)) < NumGroupChargers:
                    GroupCharging = []
                    rand_CS = np.random.randint(NumberOfDepots,len(NodesGroups[i]),size=(NumGroupChargers,)).tolist()
                    for j in rand_CS:
                        GroupCharging.append(NodesGroups[i][j])        
                for j in range(NumGroupChargers):
                    NominalPlan.ChargingStations.append(GroupCharging[j])
            NominalPlan.ChargingStations = np.array(NominalPlan.ChargingStations).reshape(-1,)
            NominalPlan.NumberOfChargeStations = len(NominalPlan.ChargingStations)
            for i in range(NominalPlan.NumberOfChargeStations):
                NominalPlan.ChargingProfile.append(np.array([[0.0, 0.0],[PltParams.BatteryCapacity,PltParams.BatteryCapacity/NominalPlan.StationRechargePower]]))
                NominalPlan.ChargingRateProfile.append(np.array([[0.0, NominalPlan.StationRechargePower],[PltParams.BatteryCapacity, 0.0]]))

            for i in range(NumberOfDepots,N):
                NominalPlan.LoadDemand[i] = 0 if i in NominalPlan.ChargingStations else np.random.randint(MinLoadPerNode, MaxLoadPerNode)
            OptVal = -1

        else:
            ScenarioName = ScenarioFileName.split('/')[-1].split('.')
            if ScenarioName[-1] == 'evrp':
                NominalPlan, PltParams, OptVal = LoadScenarioFileEvrp(ScenarioFileName, NominalPlan, PltParams)
            elif ScenarioName[-1] == 'xml':
                NominalPlan, PltParams, OptVal = LoadScenarioFileXml(ScenarioFileName, NominalPlan, PltParams)
                NominalPlan.NumberOfCars = NumberOfCars
            Xmin, Xmax = np.min(NominalPlan.NodesPosition[:,0])-10, np.max(NominalPlan.NodesPosition[:,0])+10
            Ymin, Ymax = np.min(NominalPlan.NodesPosition[:,1])-10, np.max(NominalPlan.NodesPosition[:,1])+10
            N = NominalPlan.N
            M = NominalPlan.NumberOfCars
            NumberOfCars = M
            NominalPlan.InitialChargeStage = PltParams.BatteryCapacity
            MaxNumberOfNodesPerCar = N
            NominalPlan.MaxNumberOfNodesPerCar = MaxNumberOfNodesPerCar
            NominalPlan.MaxTotalTimePerVehicle = 1.0e6
            if DeterministicProblem == True:
                NominalPlan.TravelSigma = np.zeros((N,N))
                NominalPlan.TravelSigma2 = np.zeros((N,N))
                NominalPlan.NodesEnergyTravelSigma = np.zeros((N,N))
                NominalPlan.NodesEnergyTravelSigma2 = np.zeros((N,N))
            if np.max(NominalPlan.TravelSigma) == 0.0 and np.max(NominalPlan.NodesEnergyTravelSigma) == 0.0:
                DeterministicProblem = True
        NominalPlan.NodesRealNames = np.arange(N)


        if np.sum(NominalPlan.LoadDemand)> PltParams.LoadCapacity*NominalPlan.NumberOfCars:
            print('Load Demand is too high - Lower Demand, Increase Load Capacity or Increase Number of Cars')
            exit()

        t = time.time()
        NodesTrajectory = np.zeros((N+1,M), dtype=int)
        EnergyEnteringNodes = np.zeros((N,))
        ChargingTime = np.zeros((N+1,M))
        EnergyExitingNodes = np.zeros((N,))
        Cost = 0.0
        if SolverType == 'Gurobi_NoClustering':
            NodesTrajectory, Cost, EnergyEnteringNodes, ChargingTime, EnergyExitingNodes = SolveGurobi_Convex_MinMax(PltParams=PltParams,
                                                                NominalPlan= NominalPlan,
                                                                MaxCalcTimeFromUpdate= MaxCalcTimeFromUpdate)
            ChargingTime = ChargingTime.reshape(-1,1)@np.ones((1,M))
            NodesGroups = []
            for i in range(M):
                NodesGroups.append(np.unique(NodesTrajectory[:,i]).tolist())
        elif SolverType == 'GRASP':
            NominalPlan.BatteryCapacity = PltParams.BatteryCapacity
            NodesTrajectory, Cost, ChargingTime = GRASP(NominalPlan= NominalPlan)
            # ChargingTime = ChargingTime.reshape(-1,1)@np.ones((1,M))
            NodesGroups = []
            for i in range(M):
                NodesGroups.append(np.unique(NodesTrajectory[:,i]).tolist())
        else:
            # Divide the nodes to groups (Clustering)
            while True:
                NodesGroups = DivideNodesToGroups(deepcopy(NominalPlan), ClusteringMethod, MaxGroupSize=MaxNumberOfNodesPerCar, ClusterSubGroups=False, isplot=iplot>=2)
                if len(NodesGroups) == 0:
                    print('Can"t find initial solution for this problem. Adding more trucks... ', NumberOfCars+1)
                    # add more trucks:
                    NumberOfCars += 1
                    M = NumberOfCars
                    NominalPlan.NumberOfCars = NumberOfCars
                    NominalPlan.CarsInDepots = np.append(NominalPlan.CarsInDepots, 0)
                    continue
                break

            NodesTrajectory = np.zeros((N+1,M), dtype=int)
            EnergyEnteringNodes = np.zeros((N,))
            ChargingTime = np.zeros((N+1,M))
            EnergyExitingNodes = np.zeros((N,))

            for Ncar in range(M):
                NominalPlanGroup = CreateSubPlanFromPlan(NominalPlan, NodesGroups[Ncar])
                if iplot>=3:
                    PlotCluster(NominalPlan, [NodesGroups[Ncar]], LoadCapacity=PltParams.LoadCapacity)
                NodesTrajectoryGroup = []

                if NominalPlanGroup.N > MaxNodesToSolver:
                    NodesSubGroups = []
                    NumSubGroups = int(np.ceil(NominalPlanGroup.N/MaxNodesToSolver))
                    NodesSubGroups_unindexed = DivideNodesToGroups(NominalPlanGroup,
                                                                ClusteringMethod,
                                                                ClusterSubGroups=True,
                                                                MaxGroupSize = MaxNodesToSolver,
                                                                isplot=iplot>=3)
                    for iSubGroup in range(NumSubGroups):
                        NodesSubGroups.append([NodesGroups[Ncar][i] for i in NodesSubGroups_unindexed[iSubGroup]])
                else:
                    NodesSubGroups = []
                    NodesSubGroups.append(NodesGroups[Ncar])
                    NumSubGroups = 1
                    NominalSubPlanGroup = NominalPlanGroup
                
                MaxChargingPotential = 0.0
                for iSubGroup in range(NumSubGroups):
                    if NumSubGroups > 1:
                        NominalSubPlanGroup = CreateSubPlanFromPlan(NominalPlan, NodesSubGroups[iSubGroup])
                    else:
                        NominalSubPlanGroup = NominalPlanGroup
                    
                    print("Starting Calculation for Car {}, With {} Custemers and {} CS:".format(Ncar, NominalSubPlanGroup.N- NominalSubPlanGroup.NumberOfChargeStations, NominalSubPlanGroup.NumberOfChargeStations))
                    t_i = time.time()
                    if SolverType == 'Gurobi':
                        iAddCS = 0
                        while True:
                            NodesTrajectorySubGroup, CostSubGroup, EnergyEnteringNodesGroup, ChargingTimeGroup, EnergyExitingNodesGroup = SolveGurobi_Convex_MinMax(PltParams=PltParams,
                                                                                    NominalPlan= NominalSubPlanGroup,
                                                                                    MaxCalcTimeFromUpdate= MaxCalcTimeFromUpdate)
                            if CostSubGroup == np.inf:
                                print('No Solution Found. Adding More Charging Station Stops...')
                                NominalSubPlanGroup = AddChargingStations(NominalSubPlanGroup, iAddCS)
                                iAddCS += 1
                            else:
                                break
                        
                        NodesTrajectorySubGroup = NodesTrajectorySubGroup.T[0].tolist()
                        ChargingStationsDataSubGroup = DataTypes.ChargingStations(NominalSubPlanGroup.ChargingStations)
                        for i in range(NominalSubPlanGroup.ChargingStations.shape[0]):
                            ChargingStationsDataSubGroup.MaxChargingPotential += PltParams.BatteryCapacity - EnergyExitingNodesGroup[NominalSubPlanGroup.ChargingStations[i]]
                            ChargingStationsDataSubGroup.EnergyEntered[i] = EnergyEnteringNodesGroup[NominalSubPlanGroup.ChargingStations[i]]
                            ChargingStationsDataSubGroup.EnergyExited[i] = EnergyExitingNodesGroup[NominalSubPlanGroup.ChargingStations[i]]
                            ChargingStationsDataSubGroup.ChargingTime[i] = ChargingTimeGroup[NominalSubPlanGroup.ChargingStations[i]]
                        MaxChargingPotential = ChargingStationsDataSubGroup.MaxChargingPotential
                    
                    else: # SolverType == 'Recursive'
                        # Divide the nodes to sub-groups (Clustering)
                        NominalPlanGroup.NumberOfCars = min(11-NominalPlanGroup.NumberOfChargeStations, NominalPlanGroup.N-NominalPlanGroup.NumberOfChargeStations-1)
                        NominalPlanGroup.CarsInDepots = np.zeros((NominalPlanGroup.NumberOfCars,), dtype=int)
                        NodesGroupsG = DivideNodesToGroups(deepcopy(NominalPlanGroup), 
                                                        ClusteringMethod, 
                                                        MaxGroupSize=MaxNumberOfNodesPerCar, 
                                                        ClusterSubGroups=True, 
                                                        isplot=iplot>=2)
                        for iCS in NominalPlanGroup.ChargingStations:
                            NodesGroupsG.append([iCS])

                        InitTraj = []
                        NextNode = 0
                        TourTime = 0.0
                        TourTimeUncertainty = 0.0
                        EnergyLeft = PltParams.BatteryCapacity
                        EnergyLeftUncertainty = 0.0

                        for i in range(NominalSubPlanGroup.N):
                            NominalSubPlanGroup.NodesTimeOfTravel[i,i] = 999.0
                            NominalSubPlanGroup.TravelSigma[i,i] = 999.0
                            NominalSubPlanGroup.TravelSigma2[i,i] = 9999.0
                            NominalSubPlanGroup.NodesEnergyTravel[i,i] = -999.0
                            NominalSubPlanGroup.NodesEnergyTravelSigma[i,i] = -999.0
                            NominalSubPlanGroup.NodesEnergyTravelSigma2[i,i] = 9999.0
                        
                        if np.sum(NominalSubPlanGroup.LoadDemand)> PltParams.LoadCapacity:
                            print('Load Demand is too high')
                            exit()
                        
                        NominalSubPlanGroup.SubGroups = NodesGroupsG
                        NominalSubPlanGroup.SubGroupsTime = np.zeros((len(NodesGroupsG),len(NodesGroupsG)))
                        for i in range(len(NodesGroupsG)):
                            for j in range(len(NodesGroupsG)):
                                if i==j: 
                                    NominalSubPlanGroup.SubGroupsTime[i,j] = 999.0
                                    continue
                                NominalSubPlanGroup.SubGroupsTime[i,j] = np.mean(NominalSubPlanGroup.NodesTimeOfTravel[np.ix_(NodesGroupsG[i],NodesGroupsG[j])])
                        
                        iAddCS = 0
                        while True:
                            BestPlan, NodesTrajectorySubGroup, CostSubGroup, ChargingStationsDataSubGroup = SolveParallelRecursive_ChargingStations(PltParams=PltParams,
                                                                                    NominalPlan= NominalSubPlanGroup,
                                                                                    i_CurrentNode = NextNode, 
                                                                                    TourTime = TourTime,
                                                                                    TourTimeUncertainty = TourTimeUncertainty,
                                                                                    EnergyLeft = EnergyLeft + MaxChargingPotential,
                                                                                    EnergyLeftUncertainty = EnergyLeftUncertainty,
                                                                                    ChargingStationsData = DataTypes.ChargingStations(NominalSubPlanGroup.ChargingStations),
                                                                                    NodesTrajectory = InitTraj, 
                                                                                    BestPlan = DataTypes.BestPlan(NominalSubPlanGroup.N, NominalSubPlanGroup.ChargingStations, time.time()),
                                                                                    MaxCalcTimeFromUpdate = MaxCalcTimeFromUpdate)
                        
                            if BestPlan.Cost == np.inf:
                                # .0, 0.0, 0.0, 0.0
                                if NextNode in NominalSubPlanGroup.ChargingStations:
                                    InitTraj = []
                                    NextNode = 0
                                    TourTime = 0.0
                                    TourTimeUncertainty = 0.0
                                    EnergyLeft = PltParams.BatteryCapacity
                                    EnergyLeftUncertainty = 0.0
                                else:
                                    print('No Solution Found. Adding More Charging Station Stops...')
                                    NominalSubPlanGroup = AddChargingStations(NominalSubPlanGroup, NominalSubPlanGroup.ChargingStations[iAddCS])
                                    NominalPlan = AddChargingStations(NominalPlan, NominalSubPlanGroup.NodesRealNames[NominalSubPlanGroup.ChargingStations[iAddCS]])
                                    iAddCS += 1
                                    NodesSubGroups[iSubGroup].append(NominalPlan.N-1)
                                    NodesGroups[Ncar].append(NominalPlan.N-1)
                            else:
                                break
                            
                    print("Calculation Time for Car {} with {} Nodes = ".format(Ncar, NominalSubPlanGroup.N), time.time()-t_i)
                    MaxChargingPotential = ChargingStationsDataSubGroup.MaxChargingPotential
                    if NumSubGroups <= 1 or iSubGroup == 0:
                        NodesTrajectoryGroup = []
                        for i in range(len(NodesTrajectorySubGroup)):
                            NodesTrajectoryGroup.append(NodesSubGroups[iSubGroup][NodesTrajectorySubGroup[i]])
                        CostGroup = CostSubGroup
                        ChargingStationsDataGroup = ChargingStationsDataSubGroup
                        # if NumSubGroups <= 1 and ChargingStationsDataSubGroup.ChargingStationsNodes.shape[0] > 0:
                        #     ChargingStationsDataGroup.ChargingStationsNodes = np.array([NodesSubGroups[iSubGroup][ChargingStationsDataSubGroup.ChargingStationsNodes]]).reshape(-1,)
                    else:
                        NodesTrajectoryGroup_toAdd = []
                        for i in range(len(NodesTrajectorySubGroup)):
                            NodesTrajectoryGroup_toAdd.append(NodesSubGroups[iSubGroup][NodesTrajectorySubGroup[i]])
                        NodesTrajectoryGroup, CostGroup, ChargingStationsDataGroup = ConnectSubGroups(PltParams=PltParams, 
                                                                                                        NominalPlan=NominalPlan, 
                                                                                                        NodesTrajectoryGroup=NodesTrajectoryGroup.copy(), 
                                                                                                        NodesTrajectorySubGroup=NodesTrajectoryGroup_toAdd.copy(),
                                                                                                        isplot = iplot>=3)

                NodesTrajectoryGroup = np.array(NodesTrajectoryGroup).reshape(-1,1)
                for i in range(ChargingStationsDataGroup.ChargingStationsNodes.shape[0]):
                    if ChargingStationsDataGroup.ChargingTime[i] > 0:
                        ChargingTime[NodesSubGroups[iSubGroup][ChargingStationsDataGroup.ChargingStationsNodes[i]],Ncar] = ChargingStationsDataGroup.ChargingTime[i]

                print('Car ', Ncar, ' Nodes: ', NodesTrajectoryGroup.T[0].tolist(), 'Cost: '+"{:.2f}".format(CostGroup))
                for i in range(len(NodesTrajectoryGroup)):
                    NodesTrajectory[i,Ncar] = NodesTrajectoryGroup[i][0]
                Cost += CostGroup
                if iplot>=2:
                    PlotCluster(NominalPlan, [NodesTrajectoryGroup[i]], LoadCapacity=PltParams.LoadCapacity, TrajSol=NodesTrajectoryGroup)
        if type(Cost) == np.ndarray:
            Cost = Cost[0]

        print(SolverType+' Calculation Time = ', time.time()-t)
        CalcTime = time.time()-t
        NominalPlan.OptVal = OptVal
        NominalPlan.ScenarioFileName = ScenarioFileName
        NominalPlan.Xmin, NominalPlan.Xmax = Xmin, Xmax
        NominalPlan.Ymin, NominalPlan.Ymax = Ymin, Ymax
        NominalPlan.ReturnToBase = ReturnToBase
        NominalPlan.MustVisitAllNodes = MustVisitAllNodes
        NominalPlan.ClusteringMethod = ClusteringMethod
        NominalPlan.MaxNodesToSolver = MaxNodesToSolver
        NominalPlan.t = t
        FinalCost = CheckAndPlotSolution(NominalPlan, PltParams, NodesTrajectory, ChargingTime, SolverType, iplot=iplot>=0)
            
        CalcTime_Gurobi = 0.0
        FinalCost_Grobi = 0.0
        if SolveAlsoWithGurobi == True:

            for ii in range(NodesTrajectory.shape[1]):
                indx_ii = np.argwhere(NodesTrajectory[:,ii] >= NominalPlan.NumberOfDepots).reshape(-1,)
                for jj in range(ii,NodesTrajectory.shape[1]):
                    if ii==jj: continue
                    indx_jj = np.argwhere(NodesTrajectory[:,jj] >= NominalPlan.NumberOfDepots).reshape(-1,)
                    if np.any(np.isin(NodesTrajectory[indx_ii,ii], NodesTrajectory[indx_jj,jj])):
                        print('Error in Clustering')
                        return 0.0, 0.0, 0.0, 0.0


            t1 = time.time()

            # InitGuess = np.array([[0, 6, 1, 2, 29, 5, 7, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 13, 11, 4, 3, 25, 8, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 16, 19, 21, 22, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 17, 26, 20, 18, 15, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
            # InitGuess = np.array([[0, 9, 7, 5, 2, 1, 29, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 12, 27, 15, 18, 20, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 13, 11, 4, 3, 25, 6, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 16, 19, 21, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).T
            NodesTrajectory_Gurobi, Cost_Gurobi, EnergyEnteringNodes_Gurobi, ChargingTime_Gurobi, EnergyExitingNodes_Gurobi = SolveGurobi_Convex_MinMax(PltParams=PltParams,
                                                                NominalPlan= NominalPlan,
                                                                MaxCalcTimeFromUpdate= 200,
                                                                InitialGuess=NodesTrajectory)
            ChargingTime_Gurobi = ChargingTime_Gurobi.reshape(-1,1)@np.ones((1,M))

            print('Gurobi Trajectory Time = ', Cost_Gurobi)
            print('Gurobi Trajectory = ', NodesTrajectory_Gurobi.T.tolist())
            FinalCost_Grobi = CheckAndPlotSolution(NominalPlan, PltParams, NodesTrajectory_Gurobi, ChargingTime_Gurobi, 'Gurobi_NoClustering', iplot=iplot>=0)
            CalcTime_Gurobi = time.time()-t1

    if DeterministicProblem == True:
        plt.show()
        exit()
    # return FinalCost, FinalCost_Grobi, CalcTime, CalcTime_Gurobi

    ############################################################################################################
    # Run Monte-Carlo Simulation for solution:
    Nmc = 10000 # 200000
    FinalTime = np.zeros((Nmc,NumberOfCars))
    MinEnergyLevel = np.zeros((Nmc,NumberOfCars))
    for n in range(Nmc):
        TimeVec = np.zeros((NumberOfCars,1))
        for m in range(NumberOfCars):
            EnergyEntered = np.zeros((N+1,1)) + InitialChargeStage
            EnergyExited = np.zeros((N+1,1)) + InitialChargeStage
            for i in range(NodesTrajectory.shape[0]):
                TimeVec[m] += NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m],m] + np.random.normal(0,1)*NominalPlan.TravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
                EnergyEntered[i+1] = EnergyExited[i] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + np.random.normal(0,1)*NominalPlan.NodesEnergyTravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
                EnergyExited[i+1] = EnergyEntered[i+1] + ChargingTime[NodesTrajectory[i+1,m],m]*NominalPlan.StationRechargePower
                EnergyExited[i+1] = min(EnergyExited[i+1],PltParams.BatteryCapacity)
                if i>0 and NodesTrajectory[i+1,m] < NominalPlan.NumberOfDepots:
                    break
            MinEnergyLevel[n,m] = np.min(EnergyEntered)
        FinalTime[n,:] = TimeVec[:,0]

    leg_str1 = []
    leg_str2 = []
    for m in range(NumberOfCars):
        sorted = np.sort(FinalTime[:,m])
        indx = int(SolutionProbabilityTimeReliability*Nmc)
        print('Car '+str(m+1)+' '+str(SolutionProbabilityTimeReliability*100)+'% Lap Time: '+"{:.2f}".format(sorted[indx])+', '+str(95)+'% Lap Time: '+"{:.2f}".format(sorted[int(0.95*Nmc)]))
        print('Car '+str(m+1)+' Has a '+"{:.2f}".format(100*np.sum(MinEnergyLevel[:,m]<0)/Nmc)+'% chance of running out of energy')
        leg_str1.append(str(SolutionProbabilityTimeReliability*100)+'% Lap Time: '+"{:.2f}".format(sorted[indx]))
        leg_str2.append("{:.1f}".format(100*np.sum(MinEnergyLevel[:,m]<0)/Nmc)+'% running out of energy')

    col_vec = ['m','k','b','r','y','g','c']
    markers = ['o','s','^','v','<','>','*']
    plt.figure()
    plt.subplot(2,1,1)
    for m in range(NumberOfCars):
        colr = col_vec[m%len(col_vec)]
        sns.kdeplot(FinalTime[:,m],color=colr, cumulative=True, linewidth=4)
    plt.legend(leg_str1, fontsize=20)
    plt.grid('on')
    plt.ylabel('CDF', fontsize=20)
    plt.xlabel('Tour Time', fontsize=20)
    plt.yticks(fontsize=20)
    for m in range(NumberOfCars):
        colr = col_vec[m%len(col_vec)]
        sorted = np.sort(FinalTime[:,m])
        indx = int(SolutionProbabilityTimeReliability*Nmc)
        plt.plot([sorted[indx],sorted[indx]],[0,1],'--',color=colr)



    plt.subplot(2,1,2)
    for m in range(NumberOfCars):
        colr = col_vec[m%len(col_vec)]
        sns.kdeplot(MinEnergyLevel[:,m],color=colr, cumulative=True, linewidth=4 )
    plt.grid('on')
    plt.ylabel('CDF', fontsize=20)
    plt.xlabel('Min SoC Along the Tour', fontsize=20)
    plt.yscale('log')
    plt.ylim([1e-4,1])
    # plt.title(str(SolutionProbabilityEnergyReliability*100)+'% Energy Reliability')
    plt.legend(leg_str2, fontsize=20)
    FileName = SolverType+"_N"+str(NominalPlan.N)+"_Cars"+str(NominalPlan.NumberOfCars)+"_MaxNodesPerCar"+str(NominalPlan.MaxNumberOfNodesPerCar)+"_MaxNodesToSolver"+str(NominalPlan.MaxNodesToSolver)+"Method"+str(NominalPlan.ClusteringMethod)+"_RechargeModel"+str(PltParams.RechargeModel)
    ScenarioName = ScenarioFileName.split('/')[-1].split('.')[0]
    

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.savefig("Results//"+ScenarioName+"//CDF_"+FileName, dpi=200, bbox_inches='tight')
    # plt.show()

    print("Finished")

    return FinalCost, FinalCost_Grobi, CalcTime, CalcTime_Gurobi



if __name__ == "__main__":
    SolGap = 0.0
    AvgCost = 0.0
    AvgCost_Grobi = 0.0
    AvgCalcTime = 0.0
    AvgCalcTime_Gurobi = 0.0
    Nmc = 1
    i = 30
    n=0
    while n< Nmc:
        i += 1
        FinalCost, FinalCost_Grobi, CalcTime, CalcTime_Gurobi = main(i, iplot = 0)
        if FinalCost == 0.0:
            continue
        n += 1
        SolGap += (FinalCost_Grobi-FinalCost)/FinalCost_Grobi
        AvgCost += FinalCost
        AvgCost_Grobi += FinalCost_Grobi
        AvgCalcTime += CalcTime
        AvgCalcTime_Gurobi += CalcTime_Gurobi

    print('Average Solution Gap [%]= ', SolGap/Nmc*100)
    print('Average Cost = ', AvgCost/Nmc)
    print('Average Cost_Grobi = ', AvgCost_Grobi/Nmc)
    print('Average CalcTime = ', AvgCalcTime/Nmc)
    print('Average CalcTime_Gurobi = ', AvgCalcTime_Gurobi/Nmc)
    plt.show()
    


