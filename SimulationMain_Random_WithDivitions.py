import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as DataTypes
from Gurobi_Convex_ChargingStations import *
from DivideToGroups import *
from GRASP import *
from RecursiveOptimalSolution_ChargingStations import *
import os
import time

os.system('cls' if os.name == 'nt' else 'clear')

np.random.seed(20)


# Define Platform Parameters:
PltParams = DataTypes.PlatformParams()
# Define Simulation Parameters:
#############################$
ScenarioFileNames = []
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/E-n37-k4-s4.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n1006-k43-s5.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/E-n60-k5-s9.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/E-n89-k7-s13.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/E-n112-k8-s11.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/F-n80-k4-s8.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/F-n140-k5-s5.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/M-n110-k10-s9.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/M-n126-k7-s5.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/M-n163-k12-s12.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/M-n212-k16-s12.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n147-k7-s4.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n221-k11-s7.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n360-k40-s9.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n469-k26-s10.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n577-k30-s4.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n698-k75-s13.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n830-k171-s11.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n920-k207-s4.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/X-n759-k98-s10.evrp')

# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/E-n29-k4-s7.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/E-n30-k3-s7.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/E-n35-k3-s5.evrp')
# ScenarioFileNames.append('/home/pmark/Desktop/ResearchMPC/Energy/Trajectory Simulation/VRP_Instances/ecvrp_benchmark_instances/F-n49-k4-s4.evrp')

ScenarioFileNames.append('./VRP_Instances/evrp-benchmark-set/E-n22-k4.evrp')
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
# ScenarioFileNames.append('./VRP_Instances/montoya-et-al-2017/tc1c10s3ct4.xml')
# ScenarioFileNames.append('./VRP_Instances/montoya-et-al-2017/tc1c80s12cf2.xml')
# Number of Nodes (Rnadomized Scenario):
iplot = 0 # 0: No Plot, 1: Plot Main Cluster, 2: Plot All Clusters, 3: Plot Connected Tours
# try:
#     os.remove("./Groups.txt")
# except:
#     pass

N = 30
CarsInDepots = [0,0,1] # Number of Cars per depot
NumberOfCars = len(CarsInDepots)
NumberOfDepots = len(np.unique(CarsInDepots))
MaxNumberOfNodesPerCar = int(3.0*(N-NumberOfDepots)/(NumberOfCars)) if NumberOfCars > 1 else N
MaxNodesToSolver = 100
PltParams.BatteryCapacity = 100.0
PltParams.LoadCapacity = 100
SolutionProbabilityTimeReliability = 0.9
SolutionProbabilityEnergyReliability = 0.999
DeterministicProblem = False
MinLoadPerNode, MaxLoadPerNode = 3, 15
MaxMissionTime = 120
ReturnToBase = True
MustVisitAllNodes = True
CostFunctionType = 1 # 1: Min Sum of Time Travelled, 2: ,Min Max Time Travelled by any car
MaxTotalTimePerVehicle  = 200.0
PltParams.RechargeModel = 'ConstantRate' # 'ExponentialRate' or 'ConstantRate'
SolverType = 'Gurobi_NoClustering' # 'Gurobi' or 'Recursive' or 'Gurobi_NoClustering' or 'GRASP'
ClusteringMethod = "Sum_AbsEigenvalue" # "Max_EigenvalueN" or "Frobenius" or "Sum_AbsEigenvalue" or "Mean_MaxRow" or "PartialMax_Eigenvalue" or "Greedy_Method"
MaxCalcTimeFromUpdate = 30.0 # Max time to calculate the solution from the last update [sec]
##############################$
# If MustVisitAllNodes is True, then the mission time is set to a large number
if MustVisitAllNodes == True:
    MaxMissionTime = 10e5
if SolverType == 'Gurobi_NoClustering':
    MaxCalcTimeFromUpdate = 300.0

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
    PltParams.VelEnergyConsumptionCoef = 0.05 # Power consumption due to velocity = VelEnergyConsumptionCoef* Vel^2
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
            NominalPlan.NodesEnergyTravelSigma[i,j] = np.abs(np.random.uniform(0.05*NominalPlan.NodesEnergyTravel[i,j], 0.1*NominalPlan.NodesEnergyTravel[i,j],1))[0]
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
            NodesGroups = DivideNodesToGroups(NominalPlan, 'Max_EigenvalueN', ClusterSubGroups=False, isplot=iplot>=1)
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
            rand_CS = np.random.randint(NumberOfDepots,len(NodesGroups[i]),size=(NumGroupChargers,)).tolist()
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
            NodesGroups = DivideNodesToGroups(deepcopy(NominalPlan), ClusteringMethod, MaxGroupSize=MaxNumberOfNodesPerCar, ClusterSubGroups=False, isplot=iplot>=1)
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
                                                            isplot=iplot>=2)
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

                    minDistToDepot = np.min(NominalSubPlanGroup.NodesTimeOfTravel[0,1:])
                    maxDsidtBetweenNodes = np.max(NominalSubPlanGroup.NodesTimeOfTravel[1:,1:])
                    InitTraj = []
                    NextNode = 0
                    TourTime = 0.0
                    TourTimeUncertainty = 0.0
                    EnergyLeft = PltParams.BatteryCapacity
                    EnergyLeftUncertainty = 0.0
                    # if minDistToDepot > maxDsidtBetweenNodes:
                    #     InitTraj = [0]
                    #     NextNode = np.argmin(NominalSubPlanGroup.NodesTimeOfTravel[0,1:])+1
                    #     TourTime = NominalSubPlanGroup.NodesTimeOfTravel[0,NextNode]
                    #     TourTimeUncertainty = NominalSubPlanGroup.TravelSigma[0,NextNode]
                    #     EnergyLeft = PltParams.BatteryCapacity + NominalSubPlanGroup.NodesEnergyTravel[0,NextNode]
                    #     EnergyLeftUncertainty = NominalSubPlanGroup.NodesEnergyTravelSigma[0,NextNode]

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
                            if NextNode in NominalSubPlanGroup.ChargingStations:
                                InitTraj = []
                                NextNode = 0
                                TourTime = 0.0
                                TourTimeUncertainty = 0.0
                                EnergyLeft = PltParams.BatteryCapacity
                                EnergyLeftUncertainty = 0.0
                            else:
                                print('No Solution Found. Adding More Charging Station Stops...')
                                NominalSubPlanGroup = AddChargingStations(NominalSubPlanGroup, iAddCS)
                                iAddCS += 1
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
        
    # Calculate the Trajectory Time and Energy:
    NumberOfCars = NodesTrajectory.shape[1]
    print(SolverType+' Calculation Time = ', time.time()-t)
    TimeVec = np.zeros((N+1,NumberOfCars))
    Energy =np.zeros((N+1,NumberOfCars)); Energy[0,:] = PltParams.BatteryCapacity
    EnergySigma2 =np.zeros((N+1,NumberOfCars)); EnergySigma2[0,:] = 0
    TimeSigma2 =np.zeros((N+1,NumberOfCars)); EnergySigma2[0,:] = 0
    RemainingLoad = np.zeros((N+1,NumberOfCars))
    a1 = PltParams.BatteryCapacity/PltParams.FullRechargeRateFactor
    a2 = PltParams.FullRechargeRateFactor*NominalPlan.StationRechargePower/PltParams.BatteryCapacity
    for m in range(NumberOfCars):
        i = 0
        RemainingLoad[i,:] = PltParams.LoadCapacity
        while NodesTrajectory[i,m] >= NominalPlan.NumberOfDepots or i==0:
            TimeVec[i+1,m] = TimeVec[i,m] + NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m],m]
            RemainingLoad[i+1,m] = RemainingLoad[i,m] - NominalPlan.LoadDemand[NodesTrajectory[i,m]]
            if PltParams.RechargeModel == 'ConstantRate':
                Energy[i+1,m] = Energy[i,m] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[i+1,m],m]
                EnergySigma2[i+1,m] = EnergySigma2[i,m] + NominalPlan.NodesEnergyTravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]**2
                TimeSigma2[i+1,m] = TimeSigma2[i,m] + NominalPlan.TravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]**2
            else:
                Energy[i+1,m] = NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + (a1 + (Energy[i,m]-a1)*np.exp(-a2*ChargingTime[NodesTrajectory[i,m],m]))
            i += 1
    EnergySigma = np.sqrt(EnergySigma2)
    TimeSigma = np.sqrt(TimeSigma2)
    UncertainEnergy = Energy - NominalPlan.EnergyAlpha*EnergySigma
    UncertainTime = TimeVec + NominalPlan.TimeAlpha*TimeSigma

    while np.sum(NodesTrajectory[-2,:]) < NominalPlan.NumberOfDepots:
        NodesTrajectory = NodesTrajectory[:-1,:]
        TimeVec = TimeVec[:-1,:]
        RemainingLoad = RemainingLoad[:-1,:]
        Energy = Energy[:-1,:]
        EnergySigma = EnergySigma[:-1,:]
        TimeSigma = TimeSigma[:-1,:]
        UncertainEnergy = UncertainEnergy[:-1,:]
        UncertainTime = UncertainTime[:-1,:]
    
    Customers = set(range(NominalPlan.NumberOfDepots,N)) - set(NominalPlan.ChargingStations)
    for m in range(NumberOfCars):
        Customers = Customers - set(NodesTrajectory[:,m].tolist())

    if len(Customers) > 0:
        print('Not all customers were visited: ', Customers)
        exit()


    print(SolverType+' Trajectory Time = ', np.sum(np.max(TimeVec+NominalPlan.TimeAlpha*TimeSigma, axis=0)))
    for i in range(NumberOfCars):
        iDepot = np.argwhere(NodesTrajectory[:,i] < NominalPlan.NumberOfDepots)[1]
        print('Car ', i, ' Trajectory: ', NodesTrajectory[:,i].T,' Cost: {:4.2f}'.format(TimeVec[iDepot[0],i]), ', Remaining Load: ', RemainingLoad[iDepot[0],i], ', Min Energy Along Traj: ',"{:.2f}".format(np.min(UncertainEnergy[:,i])))

    SolGap = 0
    if OptVal > 0:
        SolGap = (Cost-OptVal)/OptVal*100
        print('Solution Gap [%]= {:4.2f}'.format(SolGap))


    if len(ScenarioFileName) >= 0:
        ScenarioName = ScenarioFileName.split('/')[-1].split('.')[0]
        path = os.path.join("Results", ScenarioName) 
        os.makedirs(path, exist_ok=True)
    #     for i in range(NominalPlan.NumberOfCars):
    #         PlotGraph([i], NodesGroups, NodesTrajectory, TimeVec, UncertainTime, Energy, UncertainEnergy, NominalPlan, PltParams)
    #         FileName = SolverType+"_N"+str(NominalPlan.N)+"_Cars"+str(NominalPlan.NumberOfCars)+"_MaxNodesPerCar"+str(NominalPlan.MaxNumberOfNodesPerCar)+"_MaxNodesToSolver"+str(MaxNodesToSolver)+"Method"+str(ClusteringMethod)+"_RechargeModel"+str(PltParams.RechargeModel)+"CarNum_"+str(i)+'.png'
            # plt.savefig("Results//"+ScenarioName+"//"+FileName, dpi=300)
    #         plt.close()

    col_vec = ['m','y','b','r','g','c','k']
    markers = ['o','s','^','v','<','>','*']
    imarkers = 0
    plt.figure()
    plt.subplot(3,1,(1,2))
    leg_str = []
    legi = np.zeros((NumberOfCars,), dtype=object)
    plt.plot(NominalPlan.NodesPosition[0,0].T,NominalPlan.NodesPosition[0,1].T,'o',linewidth=20, color='k')
    for m in range(NumberOfCars):
        if 0 == i%len(col_vec) and i>0:
            imarkers += 1
        NodesToPlot = list(set(NodesGroups[m][1:])-set(NominalPlan.ChargingStations))
        plt.plot(NominalPlan.NodesPosition[NodesToPlot,0].T,NominalPlan.NodesPosition[NodesToPlot,1].T,markers[imarkers%len(markers)],linewidth=10, color=col_vec[m%len(col_vec)])
    plt.plot(NominalPlan.NodesPosition[NominalPlan.ChargingStations,0].T,NominalPlan.NodesPosition[NominalPlan.ChargingStations,1].T,'x',linewidth=50, color='k')
    plt.grid('on')
    plt.xlim((Xmin,Xmax))
    plt.ylim((Ymin,Ymax))
    # if N<=10:
    #     for i in range(N):
    #         for j in range(i+1,N):
    #             plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),0.5*(NodesPosition[j,0]-NodesPosition[i,0]),0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
    #             plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),-0.5*(NodesPosition[j,0]-NodesPosition[i,0]),-0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
    for i in range(NominalPlan.N):
        colr = 'r' if i<NominalPlan.NumberOfDepots else 'c'
        if i in NominalPlan.ChargingStations: continue
        plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=20)
    for m in range(NominalPlan.NumberOfCars):
        colr = col_vec[m%len(col_vec)]
        for i in range(0,len(NodesTrajectory)-1):
            j1 = NodesTrajectory[i,m]
            j2 = NodesTrajectory[i+1,m]
            if (ReturnToBase==True and j1 >= 0) or (ReturnToBase==False and j2>=0) or i==0:
                legi[m] = plt.arrow(NominalPlan.NodesPosition[j1,0],NominalPlan.NodesPosition[j1,1],NominalPlan.NodesPosition[j2,0]-NominalPlan.NodesPosition[j1,0],NominalPlan.NodesPosition[j2,1]-NominalPlan.NodesPosition[j1,1], width= 0.5, color=colr)
                # plt.text(0.5*NodesPosition[j1,0]+0.5*NodesPosition[j2,0], 1+0.5*NodesPosition[j1,1]+0.5*NodesPosition[j2,1]+4,"({:2.3},{:2.2})".format(NominalPlan.NodesTimeOfTravel[j1,j2], NominalPlan.NodesEnergyTravelSigma[j1,j2]), color='r', fontsize=10)
                # plt.text(0.5*NodesPosition[j1,0]+0.5*NodesPosition[j2,0], 1+0.5*NodesPosition[j1,1]+0.5*NodesPosition[j2,1]-4,"({:2.3},{:2.2})".format(NominalPlan.NodesEnergyTravel[j1,j2], NominalPlan.NodesEnergyTravelSigma[j1,j2]), color='b', fontsize=10)
            if j2 < NominalPlan.NumberOfDepots:
                break
        if np.max(NodesTrajectory[:,m])>0:
            indx = np.argwhere(NodesTrajectory[:,m] > NominalPlan.NumberOfDepots)
            indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
        else:
            indx = 2
        leg_str.append('Car '+str(m+1)+" Number of Nodes: {}".format(indx-2))

    plt.legend(legi,leg_str)
    plt.title("Cost is "+"{:.2f}".format(Cost)+", Solution Gap: {:4.2f}".format(SolGap)+"%")
    plt.subplot(3,1,3)
    leg_str = []
    for m in range(NominalPlan.NumberOfCars):
        colr = col_vec[m%len(col_vec)]
        if np.max(NodesTrajectory[:,m])>0:
            indx = np.argwhere(NodesTrajectory[:,m] > NominalPlan.NumberOfDepots)
            indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
        else:
            indx = 0
        plt.plot(Energy[0:indx,m],'o-',color=colr)
        leg_str.append('Car '+str(m+1)+' Nominal Lap Time: '+"{:.2f}".format(np.max(TimeVec[:,m])))
        for i in range(indx):
            plt.text(i,Energy[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
        for i in range(0,indx-1):
            plt.text(i+0.5,0.5*Energy[i,m]+0.5*Energy[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
    plt.grid('on')
    plt.ylim((0,PltParams.BatteryCapacity))
    plt.legend(leg_str)
    plt.ylabel('Energy')
    for m in range(NominalPlan.NumberOfCars):
        colr = col_vec[m%len(col_vec)]
        if np.max(NodesTrajectory[:,m])>0:
            indx = np.argwhere(NodesTrajectory[:,m] > NominalPlan.NumberOfDepots)
            indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
        else:
            indx = 0
        plt.plot(UncertainEnergy[0:indx,m],'-.',color=colr)
        # plt.plot(EnergyExitingNodes[NodesTrajectory[:,m]],'x:',color='k')
        

        for i in range(NominalPlan.NumberOfChargeStations):
            j = NominalPlan.ChargingStations[i]
            indx = np.where(NodesTrajectory[:,m] == j)
            if indx[0].shape[0]>0 and ChargingTime[j,m] > 0:
                if PltParams.RechargeModel == 'ConstantRate':
                    Engery_i = ChargingTime[j,m]*NominalPlan.StationRechargePower
                else:
                    Engery_i = a1 + (Energy[indx[0][0],indx[1][0]]-a1)*np.exp(-a2*ChargingTime[j,m])
                plt.arrow(indx[0][0],0,0,max(Engery_i,1.0), color=colr, width= 0.1)
                plt.text(indx[0][0]+0.2,5+i*5,"{:.2f}".format(Engery_i), color=colr,fontsize=20)
    plt.xlabel('Nodes')
    FileName = SolverType+"_N"+str(NominalPlan.N)+"_Cars"+str(NominalPlan.NumberOfCars)+"_MaxNodesPerCar"+str(NominalPlan.MaxNumberOfNodesPerCar)+"_MaxNodesToSolver"+str(MaxNodesToSolver)+"Method"+str(ClusteringMethod)+"_RechargeModel"+str(PltParams.RechargeModel)
    if ScenarioFileName is not None:
        ScenarioName = ScenarioFileName.split('/')[-1].split('.')[0]
        plt.savefig("Results//"+ScenarioName+"//"+FileName+'.png', dpi=300)
    else:
        plt.savefig("Results//Nodes_"+FileName+'.png', dpi=300)


    with open("Results//"+ScenarioName+"//"+FileName+'.txt', 'w') as f:
        f.write('Solver Type: '+SolverType+'\n')
        f.write('Solving Time: '+str(time.time()-t)+'\n')
        f.write('N: '+str(NominalPlan.N)+'\n')
        f.write('Cars: '+str(NominalPlan.NumberOfCars)+'\n')
        f.write('Clustering Method: '+str(ClusteringMethod)+'\n')
        f.write('Recharge Model: '+str(PltParams.RechargeModel)+'\n')
        f.write('Cost: '+str(Cost)+'\n')
        f.write('Solution Gap: '+str(SolGap)+'\n')
        f.write('Charging Stations: '+str(NominalPlan.ChargingStations)+'\n')
        for m in range(NominalPlan.NumberOfCars):
            f.write('Car '+str(m+1)+', Cost = {:4.2f}'.format(np.max(TimeVec[:,m]))+' Trajectory Node: '+str(NodesTrajectory[:,m].T))
            f.write('\n')
        for m in range(NominalPlan.NumberOfCars):
            f.write('Car '+str(m+1)+' Time: '+str(TimeVec[:,m].T))
            f.write('\n')
        for m in range(NominalPlan.NumberOfCars):
            f.write('Car '+str(m+1)+' Energy: '+str(Energy[:,m].T))
            f.write('\n')
        for m in range(NominalPlan.NumberOfCars):
            f.write('Car '+str(m+1)+' Uncertain Time: '+str(UncertainTime[:,m].T))
            f.write('\n')
        for m in range(NominalPlan.NumberOfCars):
            f.write('Car '+str(m+1)+' UncertainEnergy: '+str(UncertainEnergy[:,m].T))
            f.write('\n')

        f.close()

if DeterministicProblem == True:
    plt.show()
    exit()

############################################################################################################
# Run Monte-Carlo Simulation for solution:
Nmc = 10000
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
    leg_str1.append('Car '+' '+str(SolutionProbabilityTimeReliability*100)+'% Lap Time: '+"{:.2f}".format(sorted[indx])+', '+str(95)+'% Lap Time: '+"{:.2f}".format(sorted[int(0.95*Nmc)]))
    leg_str2.append('Car '+str(m+1)+' Has a '+"{:.2f}".format(100*np.sum(MinEnergyLevel[:,m]<0)/Nmc)+'% chance of running out of energy')
plt.figure()
plt.subplot(2,1,1)
for m in range(NumberOfCars):
    colr = col_vec[m%len(col_vec)]
    plt.plot(FinalTime[:,m],'s',color=colr)
plt.legend(leg_str1)
plt.grid('on')
plt.ylabel('Time')
plt.title(str(SolutionProbabilityTimeReliability*100)+'% Cost is '+"{:.2f}".format(Cost))
plt.subplot(2,1,2)
for m in range(NumberOfCars):
    colr = col_vec[m%len(col_vec)]
    plt.plot(MinEnergyLevel[:,m],'s',color=colr)
plt.grid('on')
plt.ylabel('Min Energy')
plt.xlabel('Monte-Carlo Simulation')
plt.title(str(SolutionProbabilityEnergyReliability*100)+'% Energy Reliability')
plt.legend(leg_str2)
plt.savefig("Results//MC_"+FileName, dpi=200)

plt.show()

print("Finished")




