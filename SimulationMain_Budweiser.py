import numpy as np
import matplotlib.pyplot as plt
import SimDataTypes as DataTypes
import csv
from CVX_Solution import *
from CVX_Solution_ChargingStations import *
from Pyomo_Solution_ChargingStations import *
from Gurobi_Solution_ChargingStations import *
from Gurobi_Convex_ChargingStations import *
import os
import time
os.system('cls' if os.name == 'nt' else 'clear')

np.random.seed(10)

# Define Simulation Parameters:
##############################$
# Number of Nodes
NumberOfCars = 5
NumberOfChargeStations = 0
MaxMissionTime = 120
ReturnToBase = True
MustVisitAllNodes = True
SolutionProbabilityTimeReliability = 0.9
SolutionProbabilityEnergyReliability = 0.999
Scenario = 11 # -1: Mean and Std form data, 0-23: Specific Scenario
CostFunctionType = 2 # 1: Min Sum of Time Travelled, 2: ,Min Max Time Travelled by any car
MaxTotalTimePerVehicle = 150.0
MaxNumberOfNodesPerCar = 20

# If MustVisitAllNodes is True, then the mission time is set to a large number
if MustVisitAllNodes == True:
    MaxMissionTime = 10e5


# Load Nodes Database (budweiser)
NumberOfFiles = 24
Dir = './/Bay_Field//'
filetypes = ['d','tt','ec','ec_based_on_soc']
BudweiserData = []
for iFile in range(NumberOfFiles):
    Data = dict()
    for filetype in filetypes:
        filename = "Bay_Field_2022-09-24-"+str(iFile).zfill(2)+"-00"
        with open(Dir+filename+"_"+filetype+".csv", "r") as csvfile:
            csvreader = csv.DictReader(csvfile)
            Data[filetype] = np.zeros((len(csvreader.fieldnames)-1,len(csvreader.fieldnames)-1))
            i=0
            for row in csvreader:
                temp = row
                for j in range(len(csvreader.fieldnames)-1):
                    Data[filetype][i,j] = temp[str(j)]
                i += 1
    BudweiserData.append(Data)
N = len(Data['d'])

# Calc Mean and Std of Data:
BudweiserMeanData = dict()
BudweiserStdData = dict()
BudweiserMaxData = dict()
BudweiserMinData = dict()
for filetype in filetypes:
    BudweiserMeanData[filetype] = np.zeros((N,N))
    BudweiserStdData[filetype] = np.zeros((N,N))
    BudweiserMaxData[filetype] = BudweiserData[0][filetype]
    BudweiserMinData[filetype] = BudweiserData[0][filetype]
    for i in range(NumberOfFiles):
        BudweiserMeanData[filetype] += BudweiserData[i][filetype]
    BudweiserMeanData[filetype] = BudweiserMeanData[filetype]/NumberOfFiles
    for i in range(NumberOfFiles):
        BudweiserStdData[filetype] += (BudweiserData[i][filetype]-BudweiserMeanData[filetype])**2
    BudweiserStdData[filetype] = np.sqrt(BudweiserStdData[filetype]/NumberOfFiles)
    for i in range(1,NumberOfFiles):
        BudweiserMaxData[filetype] = np.maximum(BudweiserMaxData[filetype],BudweiserData[i][filetype])
        BudweiserMinData[filetype] = np.minimum(BudweiserMinData[filetype],BudweiserData[i][filetype])

# Platform Parameters:
PltParams = DataTypes.PlatformParams()
PltParams.BatteryCapacity = 100.0
PltParams.MinimalSOC = 0
PltParams.RechargeModel = 'ExponentialRate' # 'ExponentialRate' or 'ConstantRate'
PltParams.FullRechargeRateFactor = 0.5

# Set the Nomianl Time of Travel between any 2 nodes as the distance between 
# the nodes divided by the estimated travel velocity:
NominalPlan = DataTypes.NominalPlanning(N)
NominalPlan.N = N
if Scenario >= 0:
    NominalPlan.NodesTimeOfTravel = BudweiserData[Scenario]['tt']
    NominalPlan.NodesEnergyTravel = BudweiserData[Scenario]['ec']
    NominalPlan.NodesEnergyTravelBasedOnSOC = BudweiserData[Scenario]['ec_based_on_soc']
    NominalPlan.NodesDistance = BudweiserData[Scenario]['d']
else:
    NominalPlan.NodesTimeOfTravel = BudweiserMeanData['tt']
    NominalPlan.TravelSigma = BudweiserStdData['tt']
    NominalPlan.NodesEnergyTravel = BudweiserMeanData['ec']
    NominalPlan.NodesEnergyTravelSigma = BudweiserStdData['ec_based_on_soc']
    NominalPlan.NodesEnergyTravelBasedOnSOC = BudweiserMeanData['ec_based_on_soc']
    NominalPlan.NodesDistance = BudweiserMeanData['d']

NominalPlan.NodesEnergyTravel = NominalPlan.NodesEnergyTravelBasedOnSOC

## Nominal Solution to the problem:
#
#       max sum(Time_ij)
#
InitialChargeStage = 0.75* PltParams.BatteryCapacity
NominalPlan.TimeCoefInCost = 1.0
NominalPlan.PriorityCoefInCost = 100.0 if MustVisitAllNodes == False else 0.0
NominalPlan.ReturnToBase = ReturnToBase
NominalPlan.MustVisitAllNodes = MustVisitAllNodes
NominalPlan.NumberOfCars = NumberOfCars
NominalPlan.MaxNumberOfNodesPerCar = MaxNumberOfNodesPerCar
NominalPlan.ChargingStations = np.zeros((1,1), dtype=int)
NominalPlan.StationRechargePower = 10.0
NominalPlan.FullRechargeRateFactor = 0.5
NominalPlan.SolutionProbabilityTimeReliability = SolutionProbabilityTimeReliability
NominalPlan.SolutionProbabilityEnergyReliability = SolutionProbabilityEnergyReliability
NominalPlan.CostFunctionType = CostFunctionType
NominalPlan.MaxTotalTimePerVehicle = MaxTotalTimePerVehicle
NominalPlan.NumberOfChargeStations = NumberOfChargeStations

t = time.time()
BestPlan = DataTypes.BestPlan(N)
NodesTrajectory, Cost, EnergyEnteringNodes, ChargingTime, EnergyExitingNodes = SolveGurobi_Convex_MinMax(PltParams=PltParams,
                                                        NominalPlan= NominalPlan,
                                                        NodesWorkDone = np.zeros((N,1), dtype=int),
                                                        TimeLeft= MaxMissionTime,
                                                        PowerLeft= InitialChargeStage,
                                                        i_CurrentNode= 0,
                                                        NodesTrajectory= [],
                                                        NodesWorkSequence= [],
                                                        Cost= 0.0)

# NominalPlan.NodesTimeOfTravel = BudweiserMaxData['tt']

NumberOfCars = NodesTrajectory.shape[1]
print('Gurobi Calculation Time = ', time.time()-t)
BestPlan.NodesTrajectory = NodesTrajectory
BestPlan.PowerLeft = np.append(EnergyEnteringNodes,BestPlan.PowerLeft[-1] + NominalPlan.NodesEnergyTravel[NodesTrajectory[-2],0] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[-2]])
TimeVec = np.zeros((N+1,NumberOfCars))
TimeSigma2Vec = np.zeros((N+1,NumberOfCars)); TimeSigma2Vec[0,:] = 0
DistanceVec = np.zeros((N+1,NumberOfCars))
Energy =np.zeros((N+1,NumberOfCars)); Energy[0,:] = InitialChargeStage
EnergySigma2 =np.zeros((N+1,NumberOfCars)); EnergySigma2[0,:] = 0
a1 = PltParams.BatteryCapacity/PltParams.FullRechargeRateFactor
a2 = PltParams.FullRechargeRateFactor*NominalPlan.StationRechargePower/PltParams.BatteryCapacity
for m in range(NumberOfCars):
    i = 0
    while NodesTrajectory[i,m] > 0 or i==0:
        TimeVec[i+1,m] = TimeVec[i,m] + NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m]]
        TimeSigma2Vec[i+1,m] = TimeSigma2Vec[i,m] + NominalPlan.TravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
        DistanceVec[i+1,m] = DistanceVec[i,m] + NominalPlan.NodesDistance[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
        if PltParams.RechargeModel == 'ConstantRate':
            Energy[i+1,m] = Energy[i,m] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[i+1,m]]
            EnergySigma2[i+1,m] = EnergySigma2[i,m] + NominalPlan.NodesEnergyTravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]**2
        else:
            Energy[i+1,m] = NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + (a1 + (Energy[i,m]-a1)*np.exp(-a2*ChargingTime[NodesTrajectory[i,m]]))
        i += 1
EnergySigma = np.sqrt(EnergySigma2)
TimeSigma = np.sqrt(TimeSigma2Vec)
print('Gurobi Time = ', np.sum(np.max(TimeVec, axis=0)))
print('Best Traj = ',NodesTrajectory.T)

EnergyAlpha = norm.ppf(SolutionProbabilityEnergyReliability)
TimeAlpha = norm.ppf(SolutionProbabilityTimeReliability)
col_vec = ['m','y','b','r','g','c','k']
plt.figure()
plt.subplot(3,1,1)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(TimeVec[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+' Lap Time: '+"{:.2f}".format(np.max(TimeVec[:,m])))
    plt.legend(leg_str)
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    for i in range(indx):
        plt.text(i,TimeVec[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*TimeVec[i,m]+0.5*TimeVec[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
plt.plot(TimeVec[0:indx,m]+TimeAlpha*TimeSigma[0:indx,m],'-.',color=colr)
plt.grid('on')
plt.ylabel('Time')
plt.title('Time - Sum of all Cars is '+str(np.sum(np.max(TimeVec, axis=0)))+' [-]')
for i in range(NumberOfChargeStations):
    j = NominalPlan.ChargingStations[i]
    indx = np.where(BestPlan.NodesTrajectory == j)
    if ChargingTime[int(j)] > 0:
        colr = col_vec[indx[1][0]] if m < len(col_vec) else 'k'
        plt.arrow(indx[0][0],0,0,max(ChargingTime[int(j)],1.0), color=colr, width= 0.1)
        plt.text(indx[0][0]+0.2,5,"{:.2f}".format(ChargingTime[int(j)]), color=colr, fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*TimeVec[i,m]+0.5*TimeVec[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='k',fontsize=10)
plt.subplot(3,1,2)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(Energy[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+" Number of Nodes: {}".format(indx-2))
    for i in range(indx):
        plt.text(i,Energy[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*Energy[i,m]+0.5*Energy[i+1,m]-0.3,"{:.3}".format(NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
    
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(Energy[0:indx,m]-EnergyAlpha*EnergySigma[0:indx,m],'-.',color=colr)
plt.grid('on')
# plt.ylim((0,PltParams.BatteryCapacity))
plt.legend(leg_str)
plt.ylabel('Energy')
for i in range(NumberOfChargeStations):
    j = NominalPlan.ChargingStations[i]
    indx = np.where(BestPlan.NodesTrajectory == j)
    if ChargingTime[int(j)] > 0:
        colr = col_vec[indx[1][0]] if m < len(col_vec) else 'k'
        plt.arrow(indx[0][0],0,0,max(ChargingTime[int(j)]*NominalPlan.StationRechargePower,1.0), color=colr, width= 0.1)
        plt.text(indx[0][0]+0.2,5,"{:.2f}".format(ChargingTime[int(j)]*NominalPlan.StationRechargePower), color=colr, fontsize=20)
plt.subplot(3,1,3)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(DistanceVec[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+' Lap Distance: '+"{:.1f}".format(np.max(DistanceVec[:,m])))
    for i in range(indx):
        plt.text(i,DistanceVec[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*DistanceVec[i,m]+0.5*DistanceVec[i+1,m]-0.3,"{:.1}".format(NominalPlan.NodesDistance[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
plt.title('Distance - Sum of all Cars is '+str(np.sum(np.max(DistanceVec, axis=0))).zfill(1)+' [-]')
plt.grid('on')
plt.legend(leg_str)
plt.ylabel('Distance')
plt.tight_layout()


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



## "Current Solution"
# NominalPlan.NodesTimeOfTravel = BudweiserMinData['tt']

NumberOfCars=5
CurrentNodesTrajectory = np.zeros((N,NumberOfCars), dtype=int)
CurrentNodesTrajectory[1:9,0] = range(1,9)
CurrentNodesTrajectory[1:11,1] = range(9,19)
CurrentNodesTrajectory[1:9,2] = range(19,27)
CurrentNodesTrajectory[1:10,3] = range(27,36)
CurrentNodesTrajectory[1:12,4] = range(36,47)
NodesTrajectory = CurrentNodesTrajectory

BestPlan.NodesTrajectory = CurrentNodesTrajectory
TimeVec = np.zeros((N+1,NumberOfCars))
TimeSigma2Vec = np.zeros((N+1,NumberOfCars)); TimeSigma2Vec[0,:] = 0
DistanceVec = np.zeros((N+1,NumberOfCars))
Energy =np.zeros((N+1,NumberOfCars)); Energy[0,:] = InitialChargeStage
EnergySigma2 =np.zeros((N+1,NumberOfCars)); EnergySigma2[0,:] = 0
a1 = PltParams.BatteryCapacity/PltParams.FullRechargeRateFactor
a2 = PltParams.FullRechargeRateFactor*NominalPlan.StationRechargePower/PltParams.BatteryCapacity
for m in range(NumberOfCars):
    i = 0
    while NodesTrajectory[i,m] > 0 or i==0:
        TimeVec[i+1,m] = TimeVec[i,m] + NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + ChargingTime[NodesTrajectory[i,m]]
        TimeSigma2Vec[i+1,m] = TimeSigma2Vec[i,m] + NominalPlan.TravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
        DistanceVec[i+1,m] = DistanceVec[i,m] + NominalPlan.NodesDistance[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]
        if PltParams.RechargeModel == 'ConstantRate':
            Energy[i+1,m] = Energy[i,m] + NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + NominalPlan.StationRechargePower*ChargingTime[NodesTrajectory[i+1,m]]
            EnergySigma2[i+1,m] = EnergySigma2[i,m] + NominalPlan.NodesEnergyTravelSigma[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]**2
        else:
            Energy[i+1,m] = NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]] + (a1 + (Energy[i,m]-a1)*np.exp(-a2*ChargingTime[NodesTrajectory[i,m]]))
        i += 1
EnergySigma = np.sqrt(EnergySigma2)
TimeSigma = np.sqrt(TimeSigma2Vec)

Cost = np.sum(np.max(TimeVec,axis=0) + TimeAlpha*np.max(TimeSigma,axis=0))

plt.figure()
plt.subplot(3,1,1)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(TimeVec[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+' Lap Time: '+"{:.2f}".format(np.max(TimeVec[:,m])))
    for i in range(indx):
        plt.text(i,TimeVec[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*TimeVec[i,m]+0.5*TimeVec[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(TimeVec[0:indx,m]+TimeAlpha*TimeSigma[0:indx,m],'-.',color=colr)
plt.grid('on')
plt.ylabel('Time')
plt.legend(leg_str)
for i in range(NumberOfChargeStations):
    j = NominalPlan.ChargingStations[i]
    indx = np.where(BestPlan.NodesTrajectory == j)
    if ChargingTime[int(j)] > 0:
        colr = col_vec[indx[1][0]] if m < len(col_vec) else 'k'
        plt.arrow(indx[0][0],0,0,max(ChargingTime[int(j)],1.0), color=colr, width= 0.1)
        plt.text(indx[0][0]+0.2,5,"{:.2f}".format(ChargingTime[int(j)]), color=colr, fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*TimeVec[i,m]+0.5*TimeVec[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='k',fontsize=10)
plt.title('Time - Sum of all Cars is '+str(np.sum(np.max(TimeVec, axis=0)))+' [-]')
plt.subplot(3,1,2)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(Energy[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+" Number of Nodes: {}".format(indx-2))
    for i in range(indx):
        plt.text(i,Energy[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*Energy[i,m]+0.5*Energy[i+1,m]-0.3,"{:.3}".format(NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
plt.legend(leg_str)
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(Energy[0:indx,m]-EnergyAlpha*EnergySigma[0:indx,m],'-.',color=colr)
plt.grid('on')
# plt.ylim((0,PltParams.BatteryCapacity))
plt.ylabel('Energy')
for i in range(NumberOfChargeStations):
    j = NominalPlan.ChargingStations[i]
    indx = np.where(BestPlan.NodesTrajectory == j)
    if ChargingTime[int(j)] > 0:
        colr = col_vec[indx[1][0]] if m < len(col_vec) else 'k'
        plt.arrow(indx[0][0],0,0,max(ChargingTime[int(j)]*NominalPlan.StationRechargePower,1.0), color=colr, width= 0.1)
        plt.text(indx[0][0]+0.2,5,"{:.2f}".format(ChargingTime[int(j)]*NominalPlan.StationRechargePower), color=colr, fontsize=20)
plt.subplot(3,1,3)
leg_str = []
for m in range(NumberOfCars):
    colr = col_vec[m] if m < len(col_vec) else 'k'
    indx = np.argwhere(NodesTrajectory[:,m] > 0)
    indx = indx[-1][0] if ReturnToBase==False else indx[-1][0]+2
    plt.plot(DistanceVec[0:indx,m],'o-',color=colr)
    leg_str.append('Car '+str(m+1)+' Lap Distance: '+"{:.1f}".format(np.max(DistanceVec[:,m])))
    for i in range(indx):
        plt.text(i,DistanceVec[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
    for i in range(0,indx-1):
        plt.text(i+0.5,0.5*DistanceVec[i,m]+0.5*DistanceVec[i+1,m]-0.3,"{:.5}".format(NominalPlan.NodesDistance[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
plt.title('Distance - Sum of all Cars is '+str(np.sum(np.max(DistanceVec, axis=0))).zfill(5)+' [-]')
plt.grid('on')
plt.ylabel('Distance')
plt.tight_layout()
plt.legend(leg_str)


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
