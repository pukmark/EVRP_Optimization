import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as DataTypes
from Gurobi_Solution_ChargingStations import *
from Gurobi_Convex_ChargingStations import *
from Pyomo_Solution_ChargingStations import *
import os
import time
os.system('cls' if os.name == 'nt' else 'clear')

np.random.seed(20)

# Define Simulation Parameters:
##############################$
# Number of Nodes
N = 15
NumberOfCars = 0
MaxNumberOfNodesPerCar = int(N*0.75) if NumberOfCars > 1 else N
NumberOfChargeStations = int(np.ceil(N/10))
SolutionProbabilityTimeReliability = 0.9
SolutionProbabilityEnergyReliability = 0.999
MaxMissionTime = 120
ReturnToBase = True
MustVisitAllNodes = True
MaxPriority = 1 # Max Priority of a node Priority = 1,...,MaxPriority
CostFunctionType = 1 # 1: Min Sum of Time Travelled, 2: ,Min Max Time Travelled by any car
MaxTotalTimePerVehicle = 100.0
##############################$
# If MustVisitAllNodes is True, then the mission time is set to a large number
if MustVisitAllNodes == True:
    MaxMissionTime = 10e5

# Map Size
Xmin, Xmax = -100, 100
Ymin, Ymax = -100, 100

# Platform Parameters:
PltParams = DataTypes.PlatformParams()
PltParams.Vmax = 10 # Platform Max Speed
PltParams.MinVelReductionCoef, PltParams.MaxVelReductionCoef = 0.0, 0.75 # min/max speed reduction factor for node2node travel
PltParams.VelEnergyConsumptionCoef = 0.03 # Power consumption due to velocity = VelEnergyConsumptionCoef* Vel^2
PltParams.VelConstPowerConsumption = 0.04
## Total Power to travel From Node i to Node J = (ConstPowerConsumption + VelEnergyConsumptionCoef* Vel^2)*Time_i2j
PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask = 2, 10
PltParams.MinTimePerTask, PltParams.MaxTimePerTask = 1, 5
# PltParams.RechargePowerPerDay = 5
PltParams.BatteryCapacity = 100.0
PltParams.MinimalSOC = 0.0*PltParams.BatteryCapacity
PltParams.RechargeModel = 'ExponentialRate' # 'ExponentialRate' or 'ConstantRate'
PltParams.FullRechargeRateFactor = 0.5

## Randomize The Nodes Locations:
NodesPosition = np.block([np.random.uniform(Xmin,Xmax, size=(N,1)), np.random.uniform(Ymin,Ymax, size=(N,1))])
# NodesPosition[0,0] = 0.0; NodesPosition[0,1] = 0.0


# Set the Nomianl Time of Travel between any 2 nodes as the distance between 
# the nodes divided by the estimated travel velocity:
NominalPlan = DataTypes.NominalPlanning(N)

for i in range(N):
    NominalPlan.NodesVelocity[i,i+1:] = 8 #np.random.uniform(PltParams.Vmax*(1.0-PltParams.MaxVelReductionCoef), PltParams.Vmax*(1.0-PltParams.MinVelReductionCoef), size=(1,N-i-1))
    NominalPlan.NodesVelocity[i+1:,i] = NominalPlan.NodesVelocity[i,i+1:].T
    for j in range(i,N):
        if i==j: continue
        NominalPlan.NodesDistance[i,j] = np.linalg.norm(np.array([NodesPosition[i,0]-NodesPosition[j,0], NodesPosition[i,1]-NodesPosition[j,1]]))
        NominalPlan.NodesDistance[j,i] = NominalPlan.NodesDistance[i,j]
        NominalPlan.NodesTimeOfTravel[i,j] = NominalPlan.NodesDistance[i,j] / NominalPlan.NodesVelocity[i,j]
        NominalPlan.NodesTimeOfTravel[j,i] = NominalPlan.NodesTimeOfTravel[i,j]
        NominalPlan.TravelSigma[i,j] = np.random.uniform(0.05*NominalPlan.NodesTimeOfTravel[i,j], 0.3*NominalPlan.NodesTimeOfTravel[i,j],1)
        NominalPlan.TravelSigma[j,i] = NominalPlan.TravelSigma[i,j]
        NominalPlan.NodesEnergyTravel[i,j] = -NominalPlan.NodesTimeOfTravel[i,j] * (PltParams.VelConstPowerConsumption + PltParams.VelEnergyConsumptionCoef*NominalPlan.NodesVelocity[i,j]**2)
        NominalPlan.NodesEnergyTravel[j,i] = NominalPlan.NodesEnergyTravel[i,j]
        NominalPlan.NodesEnergyTravelSigma[i,j] = np.abs(np.random.uniform(0.05*NominalPlan.NodesEnergyTravel[i,j], 0.1*NominalPlan.NodesEnergyTravel[i,j],1))
        NominalPlan.NodesEnergyTravelSigma[j,i] = NominalPlan.NodesEnergyTravelSigma[i,j]

## Calculate Nominal Time to spend and Energy Consumption for task in Node i
NominalPlan.NodesTaskTime = np.random.uniform(PltParams.MinTimePerTask, PltParams.MaxTimePerTask, size=(N,1))
NominalPlan.NodesTaskPower = np.random.uniform(PltParams.MinPowerConsumptionPerTask, PltParams.MaxPowerConsumptionPerTask, size=(N,1))

## Nodes Task Prioroties:
NominalPlan.NodesPriorities = np.ceil(np.random.uniform(0,MaxPriority,size=(N,1)))
NominalPlan.N = N

## Charging Stations:
NominalPlan.ChargingStations = np.random.randint(1,N,size=(NumberOfChargeStations,1))
while len(np.unique(NominalPlan.ChargingStations)) < NumberOfChargeStations:
    NominalPlan.ChargingStations = np.random.randint(1,N,size=(NumberOfChargeStations,1))
NominalPlan.NRechargeLevels = 5
NominalPlan.StationRechargePower = 3
## Nominal Solution to the problem:
#
#       max sum(A*Priority_i^2 * Task_i - B*Time_ij)
#
InitialChargeStage = 1.0* PltParams.BatteryCapacity
NominalPlan.TimeCoefInCost = 1.0
NominalPlan.PriorityCoefInCost = 100.0 if MustVisitAllNodes == False else 0.0
NominalPlan.ReturnToBase = ReturnToBase
NominalPlan.MustVisitAllNodes = MustVisitAllNodes
NominalPlan.NumberOfCars = NumberOfCars
NominalPlan.MaxNumberOfNodesPerCar = MaxNumberOfNodesPerCar
NominalPlan.SolutionProbabilityTimeReliability = SolutionProbabilityTimeReliability
NominalPlan.SolutionProbabilityEnergyReliability = SolutionProbabilityEnergyReliability
NominalPlan.CostFunctionType = CostFunctionType
NominalPlan.MaxTotalTimePerVehicle = MaxTotalTimePerVehicle
NominalPlan.NumberOfChargeStations = NumberOfChargeStations

BestPlan = DataTypes.BestPlan(N)
t = time.time()
NodesTrajectory, Cost, EnergyEnteringNodes, ChargingTime, EnergyExitingNodes = SolveGurobi_Convex_MinMax(PltParams=PltParams,
                                                        NominalPlan= NominalPlan,
                                                        NodesWorkDone = np.zeros((N,1), dtype=int),
                                                        TimeLeft= MaxMissionTime,
                                                        PowerLeft= InitialChargeStage,
                                                        i_CurrentNode= 0,
                                                        NodesTrajectory= [],
                                                        NodesWorkSequence= [],
                                                        Cost= 0.0)
NumberOfCars = NodesTrajectory.shape[1]
print('Gurobi Calculation Time = ', time.time()-t)
BestPlan.NodesTrajectory = NodesTrajectory
BestPlan.PowerLeft = np.append(EnergyEnteringNodes,BestPlan.PowerLeft[-1] + NominalPlan.NodesEnergyTravel[NodesTrajectory[-2],0] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[-2]])
TimeVec = np.zeros((N+1,NumberOfCars))
Energy =np.zeros((N+1,NumberOfCars)); Energy[0,:] = InitialChargeStage
EnergySigma2 =np.zeros((N+1,NumberOfCars)); EnergySigma2[0,:] = 0
a1 = PltParams.BatteryCapacity/PltParams.FullRechargeRateFactor
a2 = PltParams.FullRechargeRateFactor*NominalPlan.StationRechargePower/PltParams.BatteryCapacity
for m in range(NumberOfCars):
    i = 0
    while NodesTrajectory[i,m] > 0 or i==0:
        TimeVec[i+1,m] = TimeVec[i,m] + NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m]]
        if PltParams.RechargeModel == 'ConstantRate':
            Energy[i+1,m] = Energy[i,m] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[i+1,m]]
            EnergySigma2[i+1,m] = EnergySigma2[i,m] + NominalPlan.NodesEnergyTravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]**2
        else:
            Energy[i+1,m] = NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + (a1 + (Energy[i,m]-a1)*np.exp(-a2*ChargingTime[NodesTrajectory[i,m]]))
        i += 1
EnergySigma = np.sqrt(EnergySigma2)
print('Gurobi Time = ', np.sum(np.max(TimeVec, axis=0)))
print('Best Traj = ',NodesTrajectory.T)


EnergyAlpha = norm.ppf(SolutionProbabilityEnergyReliability)
col_vec = ['m','y','b','r','g','c','k']
plt.figure()
plt.subplot(3,1,(1,2))
leg_str = []
for i in range(MaxPriority):
    PriorIndx_i = np.where(NominalPlan.NodesPriorities[:,0] == i+1)
    plt.plot(NodesPosition[PriorIndx_i,0].T,NodesPosition[PriorIndx_i,1].T,'o',linewidth=10, color=col_vec[i])
plt.grid('on')
plt.xlim((Xmin,Xmax))
plt.ylim((Ymin,Ymax))
if N<=10:
    for i in range(N):
        for j in range(i+1,N):
            plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),0.5*(NodesPosition[j,0]-NodesPosition[i,0]),0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
            plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),-0.5*(NodesPosition[j,0]-NodesPosition[i,0]),-0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
for i in range(N):
    colr = 'r' if i==0 else 'c'
    colr = 'k' if i in NominalPlan.ChargingStations else colr
    plt.text(NodesPosition[i,0]+1,NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    for i in range(len(NodesTrajectory)-1):
        j1 = NodesTrajectory[i,m]
        j2 = NodesTrajectory[i+1,m]
        if (ReturnToBase==True and j1 > 0) or (ReturnToBase==False and j2>0) or i==0:
            plt.arrow(NodesPosition[j1,0],NodesPosition[j1,1],NodesPosition[j2,0]-NodesPosition[j1,0],NodesPosition[j2,1]-NodesPosition[j1,1], width= 1, color=colr)
            plt.text(0.5*NodesPosition[j1,0]+0.5*NodesPosition[j2,0], 1+0.5*NodesPosition[j1,1]+0.5*NodesPosition[j2,1],"{:.3}".format(NominalPlan.NodesTimeOfTravel[j1,j2]), color='r', fontsize=10)
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    leg_str.append('Car '+str(m+1)+" Number of Nodes: {}".format(indx-2))
plt.legend(leg_str)
# plt.title(SolverType)
plt.subplot(3,1,3)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(Energy[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+' Lap Time: '+"{:.2f}".format(np.max(TimeVec[:,m])))
    for i in range(indx):
        plt.text(i,Energy[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*Energy[i,m]+0.5*Energy[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
plt.grid('on')
plt.ylim((0,PltParams.BatteryCapacity))
plt.legend(leg_str)
plt.ylabel('Energy')
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(Energy[0:indx,m]-EnergyAlpha*EnergySigma[0:indx,m],'-.',color=colr)
    # plt.plot(EnergyExitingNodes[NodesTrajectory[:,m]],'x:',color='k')
    

for i in range(NumberOfChargeStations):
    j = NominalPlan.ChargingStations[i]
    indx = np.where(BestPlan.NodesTrajectory == j)
    if ChargingTime[int(j)] > 0:
        colr = col_vec[indx[1][0]] if m < len(col_vec) else 'k'
        if PltParams.RechargeModel == 'ConstantRate':
            Engery_i = ChargingTime[int(j)]*NominalPlan.StationRechargePower
        else:
            Engery_i = a1 + (Energy[indx[0][0],indx[1][0]]-a1)*np.exp(-a2*ChargingTime[int(j)])
        plt.arrow(indx[0][0],0,0,max(Engery_i,1.0), color=colr, width= 0.1)
        plt.text(indx[0][0]+0.2,5+i*5,"{:.2f}".format(Engery_i), color=colr,fontsize=20)
plt.xlabel('Nodes')
# plt.savefig(SolverType+'.png')

############################################################################################################
# Run Monte-Carlo Simulation for solution:
Nmc = 10000
FinalTime = np.zeros((Nmc,NumberOfCars))
MinEnergyLevel = np.zeros((Nmc,NumberOfCars))
for n in range(Nmc):
    TimeVec = np.zeros((NumberOfCars,1))
    for m in range(NumberOfCars):
        EnergyVec = np.zeros((N+1,1)) + InitialChargeStage
        for i in range(NodesTrajectory.shape[0]):
            TimeVec[m] += NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m]] + np.random.normal(0,1)*NominalPlan.TravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
            EnergyVec[i+1] = EnergyVec[i] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m]]*NominalPlan.StationRechargePower + np.random.normal(0,1)*NominalPlan.NodesEnergyTravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
            EnergyVec[i+1] = min(EnergyVec[i+1],PltParams.BatteryCapacity)
            if i>0 and NodesTrajectory[i+1,m] == 0:
                break
        MinEnergyLevel[n,m] = np.min(EnergyVec)
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
    colr = col_vec[m] if m < len(col_vec) else 'k'
    plt.plot(FinalTime[:,m],'s',color=colr)
plt.legend(leg_str1)
plt.grid('on')
plt.ylabel('Time')
plt.title(str(SolutionProbabilityTimeReliability*100)+'% Cost is '+"{:.2f}".format(Cost))
plt.subplot(2,1,2)
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    plt.plot(MinEnergyLevel[:,m],'s',color=colr)
plt.grid('on')
plt.ylabel('Min Energy')
plt.xlabel('Monte-Carlo Simulation')
plt.title(str(SolutionProbabilityEnergyReliability*100)+'% Energy Reliability')
plt.legend(leg_str2)
plt.show()

