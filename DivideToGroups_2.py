import numpy as np
import SimDataTypes as DataTypes
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from multiprocessing import Pool, Lock, Value


def CreateGroupMatrix(Group, NodesTimeOfTravel):
    GroupMatrix = np.zeros((len(Group),len(Group)))
    for i in range(len(Group)):
        for j in range(len(Group)):
            GroupMatrix[i,j] = NodesTimeOfTravel[Group[i],Group[j]]
    return GroupMatrix

def CalcEntropy(NodesGroups, NodesTimeOfTravel, Method, CS: list = []):
    NumberOfCars = len(NodesGroups)
    TimeEntropy_i = np.zeros(NumberOfCars)
    if Method == "Max_Eigenvalue":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(list(set(NodesGroups[iGroup]) - set({0})), NodesTimeOfTravel)
            w, _ = np.linalg.eig(GroupTimeMatrix)
            TimeEntropy_i[iGroup] = np.max(np.abs(w))**2 * len(w)
    elif Method == "Frobenius":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            TimeEntropy_i[iGroup] = np.linalg.norm(GroupTimeMatrix, 'fro')
    elif Method == "Mean_MaxRow":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            TimeEntropy_i[iGroup] = np.sum(np.max(GroupTimeMatrix, axis=1))**2
    elif Method == "Sum_AbsEigenvalue":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            w, v = np.linalg.eig(GroupTimeMatrix)
            TimeEntropy_i[iGroup] = np.sum(np.abs(w)**2)
    elif Method == "PartialMax_Eigenvalue":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(list(set(NodesGroups[iGroup]) - set({0})), NodesTimeOfTravel)
            w, _ = np.linalg.eig(GroupTimeMatrix)
            w = np.sort(np.abs(w))[::-1]
            ind = min(3,len(w))
            weights = np.array(range(1,ind+1))[::-1]
            TimeEntropy_i[iGroup] = len(w) * np.sum((np.abs(w[0:ind]))**2)
    elif Method == "Greedy_Method":
        for iGroup in range(NumberOfCars):
            for iNode in NodesGroups[iGroup]:
                TimeEntropy_i[iGroup] += np.mean(NodesTimeOfTravel[NodesGroups[iGroup],iNode])
            TimeEntropy_i[iGroup] = TimeEntropy_i[iGroup]**2
    else:
        print("Error: Unknown method for calculating Entropy")
    
    TimeEntropy = np.sum(TimeEntropy_i)
    return TimeEntropy

def DivideNodesToGroups(NominalPlan: DataTypes.NominalPlanning , 
                        Method = "Max_Eigenvalue", 
                        MaximizeGroupSize: bool=False,
                        MustIncludeNodeZero: bool = True, 
                        ChargingStations: list = [],
                        MaxGroupSize: int = 12,
                        LoadCapacity: float = np.inf,
                        isplot: bool = False):

    N = NominalPlan.N
    M = NominalPlan.NumberOfCars
    NodesGroups = list()
    i1 = 0 if MustIncludeNodeZero==False else 1
    NodesTimeOfTravel = NominalPlan.NodesTimeOfTravel + NominalPlan.TimeAlpha * NominalPlan.TravelSigma

    if MaximizeGroupSize == True:
        NodesSet = set(range(N))
        for iGroup in range(M-1):
            Group = []
            if iGroup == 0:
                Group.append(0)
                NodesSet.remove(0)
            while len(Group) < NominalPlan.MaxNumberOfNodesPerCar and len(NodesSet)>0:
                Entropy = np.zeros((len(NodesSet),))
                for i in range(len(NodesSet)):
                    iNode = list(NodesSet)[i]
                    Group.append(iNode)
                    Entropy[i] = CalcEntropy((Group,), NodesTimeOfTravel, Method, NominalPlan.ChargingStations)
                    Group.remove(iNode)
                Group.append(list(NodesSet)[np.argmin(Entropy)])
                NodesSet = NodesSet - set(Group)
                if len(NodesSet) == 2 and iGroup == M-2:
                    break
            NodesGroups.append(np.array(Group))
        
        NodesGroups.append(np.array(list(NodesSet)))


    else:
    # Initialize the groups by Closest Neighbor:
        for i in NominalPlan.CarsInDepots:
            NodesGroups.append(np.array([i]))
        CustomersNodes = set(range(N)) - set(NominalPlan.CarsInDepots) - set(NominalPlan.ChargingStations)
        TimeMat = NodesTimeOfTravel[list(CustomersNodes),:][:,list(CustomersNodes)]
        max_Time_i = np.argmax(np.max(TimeMat, axis=1))
        NodesGroups[i] = np.append(NodesGroups[i], list(CustomersNodes)[max_Time_i])
        CustomersNodes = CustomersNodes - set([list(CustomersNodes)[max_Time_i]])
        for i in range(1,M):
            Dist2 = np.zeros((len(CustomersNodes),))
            for j in range(i):
                Dist2 += np.sqrt(NodesTimeOfTravel[NodesGroups[j][1],list(CustomersNodes)])
            max_Dist2_i = np.argmax(Dist2)
            NodesGroups[i] = np.append(NodesGroups[i], list(CustomersNodes)[max_Dist2_i])
            CustomersNodes = CustomersNodes - set([list(CustomersNodes)[max_Dist2_i]])
            
        argDemand = np.argsort(NominalPlan.LoadDemand)[::-1]
        for iNode in argDemand:
            if iNode not in CustomersNodes:
                continue
            TimeToNode = np.zeros((M,))
            for iGroup in range(M):
                TimeToNode[iGroup] = np.mean(NodesTimeOfTravel[NodesGroups[iGroup],iNode])
            Group_MinTime = np.argsort(TimeToNode)
            for iGroup in Group_MinTime:
                if (NodesGroups[iGroup].shape[0] <= NominalPlan.MaxNumberOfNodesPerCar) and (NominalPlan.LoadDemand[iNode]+np.sum(NominalPlan.LoadDemand[NodesGroups[iGroup]])<=LoadCapacity):
                    NodesGroups[iGroup] = np.append(NodesGroups[iGroup], iNode)
                    break
                if iGroup == Group_MinTime[-1]:
                    print("Error: Node {:} can't be added to any group".format(iNode))

    # Minimize "Entropy"
    BestEntropy = CalcEntropy(NodesGroups, NodesTimeOfTravel, Method, NominalPlan.ChargingStations)
    t0 = time.time()
    print("Initial Entropy: ", BestEntropy)
    for iter in range(100):
        t0_iter = time.time()
        for iter2 in range(100):
            CurNodesGroups = NodesGroups.copy()
            GroupChanged = False
            Entropy = np.zeros((M,))
            for iGroup in range(M):
                for i in range(i1,len(CurNodesGroups[iGroup])):
                    CurNode = CurNodesGroups[iGroup][i]
                    for jGroup in range(M):
                        arg_iGroup = np.argwhere(NodesGroups[iGroup]==CurNode)
                        if iGroup == jGroup:
                            Entropy[jGroup] = BestEntropy
                            continue
                        NodesGroups[iGroup] = np.delete(NodesGroups[iGroup], arg_iGroup)
                        NodesGroups[jGroup] = np.append(NodesGroups[jGroup], CurNode)
                        if len(NodesGroups[jGroup]) > MaxGroupSize or np.sum(NominalPlan.LoadDemand[NodesGroups[jGroup]])>LoadCapacity:
                            Entropy[jGroup] = np.inf
                        else:
                            Entropy[jGroup] = CalcEntropy(NodesGroups, NodesTimeOfTravel, Method, NominalPlan.ChargingStations)
                        arg_jGroup = np.argwhere(NodesGroups[jGroup]==CurNode)
                        NodesGroups[jGroup] = np.delete(NodesGroups[jGroup], arg_jGroup)
                        NodesGroups[iGroup] = np.append(NodesGroups[iGroup], CurNode)
                        
                    Group_MinEntropy = np.argmin(Entropy)
                    BestEntropy = Entropy[Group_MinEntropy]
                    arg_iGroup = np.argwhere(NodesGroups[iGroup]==CurNode)
                    NodesGroups[iGroup] = np.delete(NodesGroups[iGroup], arg_iGroup)
                    NodesGroups[Group_MinEntropy] = np.append(NodesGroups[Group_MinEntropy], CurNode)
                    if Group_MinEntropy != iGroup:
                        GroupChanged = True
                NodesGroups[iGroup] = np.sort(NodesGroups[iGroup])
            if not GroupChanged:
                break
        print("iteration = "+str(iter)+", iteration2 = "+str(iter2)+", Entropy = {:}".format(BestEntropy), ", Iteration Clustering Time = {:}".format(time.time()-t0_iter)+"[sec]")

        CurNodesGroups = NodesGroups.copy()
        for iGroup in range(M):
            for jGroup in range(iGroup+1,M):
                if iGroup==jGroup: continue
                for iNode in NodesGroups[iGroup][i1:]:
                    for jNode in NodesGroups[jGroup][i1:]:
                        if iNode==0 or jNode==0: continue
                        arg_iNode = np.argwhere(CurNodesGroups[iGroup]==iNode)
                        arg_jNode = np.argwhere(CurNodesGroups[jGroup]==jNode)
                        CurNodesGroups[iGroup] = np.delete(CurNodesGroups[iGroup], arg_iNode)
                        CurNodesGroups[jGroup] = np.delete(CurNodesGroups[jGroup], arg_jNode)
                        CurNodesGroups[iGroup] = np.append(CurNodesGroups[iGroup], jNode)
                        CurNodesGroups[iGroup] = np.sort(CurNodesGroups[iGroup])
                        CurNodesGroups[jGroup] = np.append(CurNodesGroups[jGroup], iNode)
                        CurNodesGroups[jGroup] = np.sort(CurNodesGroups[jGroup])
                        if np.sum(NominalPlan.LoadDemand[CurNodesGroups[iGroup]])>LoadCapacity or np.sum(NominalPlan.LoadDemand[CurNodesGroups[jGroup]])>LoadCapacity:
                            CurEntropy = np.inf
                        else:
                            CurEntropy = CalcEntropy(CurNodesGroups, NodesTimeOfTravel, Method, NominalPlan.ChargingStations)
                        if CurEntropy >= BestEntropy:
                            CurNodesGroups = NodesGroups.copy()
                        else:
                            BestEntropy = CurEntropy
                            GroupChanged = True
                            for i in range(M):
                                CurNodesGroups[i] = np.sort(CurNodesGroups[i])
                            NodesGroups = CurNodesGroups.copy()
        print("iteration = "+str(iter)+", Entropy = {:}".format(BestEntropy), ", Iteration Clustering Time = {:}".format(time.time()-t0_iter)+"[sec]")
        if not GroupChanged:
            break
    print("Total Clustering Time = {:}".format(time.time()-t0)+"[sec]")

    # Make sure that the first group has the depot:
    if NodesGroups[0][0] != 0:
        for i in range(1,M):
            if NodesGroups[i][0] == 0:
                Temp = NodesGroups[0].copy()
                NodesGroups[0] = NodesGroups[i].copy()
                NodesGroups[i] = Temp.copy()
                break
    # Organize the groups:
    CurGroupIntegration = NodesGroups[0]
    for i in range(1,M-1):
        Entropy = np.zeros((M,))+np.inf
        for j in range(M-i):
            Group = np.append(CurGroupIntegration, NodesGroups[j+i])
            Entropy[i+j] = CalcEntropy([Group], NodesTimeOfTravel, Method, NominalPlan.ChargingStations)
        Group_MinEntropy = np.argmin(Entropy)
        # Set Group_MinEntropy as Group number i:
        Temp = NodesGroups[i].copy()
        NodesGroups[i] = NodesGroups[Group_MinEntropy].copy()
        NodesGroups[Group_MinEntropy] = Temp.copy()
        # Update CurGroupIntegration:
        CurGroupIntegration = np.append(CurGroupIntegration, NodesGroups[i])
            
    # Final Enthropy:
    for i in range(M):
        NodesGroups_i = []
        NodesGroups_i.append(NodesGroups[i])
        print("Final Entropy Group", i,": ", CalcEntropy(NodesGroups_i, NodesTimeOfTravel, Method, NominalPlan.ChargingStations))    

    # Print summary:
    print("Final Entropy: ", CalcEntropy(NodesGroups, NodesTimeOfTravel, Method, NominalPlan.ChargingStations))    
    for i in range(M):
        GroupTimeMatrix = CreateGroupMatrix(NodesGroups[i], NodesTimeOfTravel)
        w, v = np.linalg.eig(GroupTimeMatrix)
        w = np.sort(np.abs(w))[::-1]
        print("Group {:} - Number of Nodes: {:}, Entropy: {:}, Max Eigenvalue: {:}".format(i, len(NodesGroups[i]), CalcEntropy([NodesGroups[i]], NodesTimeOfTravel, Method, NominalPlan.ChargingStations), np.abs(w[0:3])))

    for i in range(M):
        NodesGroups[i] = list(NodesGroups[i])

    NodesWitoutCS = set(range(N)) - set(NominalPlan.ChargingStations)
    if len(NominalPlan.ChargingStations) > 0:
        for i in range(M):
            MaxEnergy = np.min(NominalPlan.NodesEnergyTravel[NodesGroups[i][0],list(NodesWitoutCS)])
            MaxEnergy += np.min(NominalPlan.NodesEnergyTravel[list(NodesWitoutCS),NodesGroups[i][0]])
            for j in range(1,len(NodesGroups[i])):
                MaxEnergy += np.mean(NominalPlan.NodesEnergyTravel[NodesGroups[i][j],list(NodesWitoutCS-set([j]))])
            
            NumberOfMaxCS = MaxEnergy + NominalPlan.InitialChargeStage

            # Mean Time to CS:
            MeanTimeToCS = np.zeros((len(NominalPlan.ChargingStations),))
            for j in range(len(NominalPlan.ChargingStations)):
                MeanTimeToCS[j] = np.mean(NominalPlan.NodesTimeOfTravel[NodesGroups[i],NominalPlan.ChargingStations[j]])

            NumOfCS = int(min(max(2.0,-0.5*NumberOfMaxCS/NominalPlan.InitialChargeStage),len(NominalPlan.ChargingStations)))
            j_CS = np.argsort(MeanTimeToCS)
            for j in range(NumOfCS):
                NodesGroups[i].append(NominalPlan.ChargingStations[j_CS[j]])




# Plot the groups
    if np.max(NominalPlan.NodesPosition)>0 and isplot==True:
        col_vec = ['m','y','b','r','g','c']
        markers = ['o','s','^','v','<','>','*']
        imarkers = 0
        leg_str = list()
        plt.figure()
        ax = plt.subplot(111)
        if MustIncludeNodeZero==True:
            plt.scatter(NominalPlan.NodesPosition[0:NominalPlan.NumberOfDepots,0], NominalPlan.NodesPosition[0:NominalPlan.NumberOfDepots,1], c='k', s=50)
            leg_str.append('Depot')
        for i in range(M):
            if 0 == i%len(col_vec) and i>0:
                imarkers += 1
            plt.scatter(NominalPlan.NodesPosition[NodesGroups[i][i1:],0], NominalPlan.NodesPosition[NodesGroups[i][i1:],1], s=50, c=col_vec[i%len(col_vec)], marker=markers[imarkers])
            leg_str.append('Group '+str(i)+" Number of Nodes: {}".format(len(NodesGroups[i])-1))
        for i in NominalPlan.ChargingStations:
            plt.scatter(NominalPlan.NodesPosition[i,0], NominalPlan.NodesPosition[i,1], c='k', s=15)
        leg_str.append('Charging Station')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(leg_str, loc=(1.05,0.5))
        for i in range(N):
            colr = 'r' if i in NominalPlan.CarsInDepots else 'c'
            colr = 'k' if i in NominalPlan.ChargingStations else colr
            plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=20)
        # plt.xlim((-100,100))
        # plt.ylim((-100,100))
        plt.grid()
        plt.show()

    return NodesGroups

def CreateSubPlanFromPlan (NominalPlan: DataTypes.NominalPlanning, NodesGroups):
    n = len(NodesGroups)
    NominalPlanGroup = DataTypes.NominalPlanning(n)
    NominalPlanGroup.ChargingStations = []
    for i in range(n):
        NominalPlanGroup.NodesPosition[i,:] = NominalPlan.NodesPosition[NodesGroups[i],:]
        NominalPlanGroup.LoadDemand[i] = NominalPlan.LoadDemand[NodesGroups[i]]
        for j in range(n):
            NominalPlanGroup.NodesVelocity[i,j] = NominalPlan.NodesVelocity[NodesGroups[i]][NodesGroups[j]]
            NominalPlanGroup.NodesDistance[i,j] = NominalPlan.NodesDistance[NodesGroups[i]][NodesGroups[j]]
            NominalPlanGroup.NodesTimeOfTravel[i,j] = NominalPlan.NodesTimeOfTravel[NodesGroups[i]][NodesGroups[j]]
            NominalPlanGroup.TravelSigma[i,j] = NominalPlan.TravelSigma[NodesGroups[i]][NodesGroups[j]]
            NominalPlanGroup.NodesEnergyTravel[i,j] = NominalPlan.NodesEnergyTravel[NodesGroups[i]][NodesGroups[j]]
            NominalPlanGroup.NodesEnergyTravelSigma[i,j] = NominalPlan.NodesEnergyTravelSigma[NodesGroups[i]][NodesGroups[j]]
        NominalPlanGroup.NodesTaskTime[i] = NominalPlan.NodesTaskTime[NodesGroups[i]]
        NominalPlanGroup.NodesTaskPower[i] = NominalPlan.NodesTaskPower[NodesGroups[i]]
        NominalPlanGroup.NodesPriorities[i] = NominalPlan.NodesPriorities[NodesGroups[i]]
        if NodesGroups[i] in NominalPlan.ChargingStations:
            NominalPlanGroup.ChargingStations.append(i)
    
    NominalPlanGroup.NodesEnergyTravelSigma2 = NominalPlanGroup.NodesEnergyTravelSigma**2
    NominalPlanGroup.TravelSigma2 = NominalPlanGroup.TravelSigma**2
    NominalPlanGroup.ChargingStations = np.array(NominalPlanGroup.ChargingStations)
    NominalPlanGroup.StationRechargePower = NominalPlan.StationRechargePower
    NominalPlanGroup.N = n
    NominalPlanGroup.TimeCoefInCost = NominalPlan.TimeCoefInCost
    NominalPlanGroup.PriorityCoefInCost = NominalPlan.PriorityCoefInCost
    NominalPlanGroup.ReturnToBase = NominalPlan.ReturnToBase
    NominalPlanGroup.MustVisitAllNodes = NominalPlan.MustVisitAllNodes
    NominalPlanGroup.NumberOfCars = 1
    NominalPlanGroup.MaxNumberOfNodesPerCar = NominalPlan.MaxNumberOfNodesPerCar
    NominalPlanGroup.SolutionProbabilityTimeReliability = NominalPlan.SolutionProbabilityTimeReliability
    NominalPlanGroup.SolutionProbabilityEnergyReliability = NominalPlan.SolutionProbabilityEnergyReliability
    NominalPlanGroup.CostFunctionType = NominalPlan.CostFunctionType
    NominalPlanGroup.MaxTotalTimePerVehicle = NominalPlan.MaxTotalTimePerVehicle
    NominalPlanGroup.NumberOfChargeStations = NominalPlan.NumberOfChargeStations
    NominalPlanGroup.EnergyAlpha = NominalPlan.EnergyAlpha
    NominalPlanGroup.TimeAlpha = NominalPlan.TimeAlpha
    NominalPlanGroup.InitialChargeStage = NominalPlan.InitialChargeStage
    NominalPlanGroup.CarsInDepots = [0]
    NominalPlanGroup.NumberOfDepots = 1
    NominalPlanGroup.NodesRealNames = NodesGroups

    return NominalPlanGroup




def ConnectSubGroups(PltParams: DataTypes.PlatformParams, NominalPlan: DataTypes.NominalPlanning, NodesTrajectoryGroup, NodesTrajectorySubGroup, isplot: bool = False):

    # Create the charging stations data
    ChargingStations = np.array([],dtype=int)
    for i in NominalPlan.ChargingStations:
        if i in NodesTrajectoryGroup or i in NodesTrajectorySubGroup:
            ChargingStations = np.append(ChargingStations,i)
    if ChargingStations.shape[0] == 0:
        ChargingStations = np.array([0])
    BestChargingStationsData = DataTypes.ChargingStations(ChargingStations)

    # Connect the subgroups
    CostGroup = np.inf
    BestNodesTrajectoryGroup = []
    NodesTrajectorySubGroup = NodesTrajectorySubGroup[0:-1]
    
    for reverse_i in range(2):
        if reverse_i == 1:
            NodesTrajectoryGroup = NodesTrajectoryGroup[::-1]
        for i in range(1,len(NodesTrajectoryGroup)):
            for j in range(len(NodesTrajectorySubGroup)):
                for reverse in range(2):
                    PotentialTrajectory = []
                    for k in range(len(NodesTrajectoryGroup)+len(NodesTrajectorySubGroup)-1):
                        if k < i:
                            PotentialTrajectory.append(NodesTrajectoryGroup[k])
                        elif k < i+len(NodesTrajectorySubGroup):
                            if reverse == 0:
                                PotentialTrajectory.append(NodesTrajectorySubGroup[(k-i+j)%len(NodesTrajectorySubGroup)])
                            elif reverse == 1:
                                PotentialTrajectory.append(NodesTrajectorySubGroup[(i+len(NodesTrajectorySubGroup)+j-k)%len(NodesTrajectorySubGroup)])
                        else:
                            PotentialTrajectory.append(NodesTrajectoryGroup[k-len(NodesTrajectorySubGroup)])
                    PotentialTrajectory.append(0)

                    #Calculate the cost of the potential trajectory:
                    TrajMeanTime = 0.0
                    TrajMeanEnergy = NominalPlan.InitialChargeStage
                    TrajSigmaTime2 = 0.0
                    TrajSigmaEnergy2 = 0.0
                    ChargingStationsData = DataTypes.ChargingStations(ChargingStations)
                    for iNode in range(len(PotentialTrajectory)-1):
                        i1 = PotentialTrajectory[iNode]
                        i2 = PotentialTrajectory[iNode+1]
                        TrajMeanTime += NominalPlan.NodesTimeOfTravel[i1,i2]
                        TrajSigmaTime2 += NominalPlan.TravelSigma2[i1,i2]
                        TrajMeanEnergy += NominalPlan.NodesEnergyTravel[i1,i2]
                        TrajSigmaEnergy2 += NominalPlan.NodesEnergyTravelSigma2[i1,i2]
                        # Check if the trajectory is feasible:
                        if TrajMeanEnergy + ChargingStationsData.MaxChargingPotential - NominalPlan.EnergyAlpha*np.sqrt(TrajSigmaEnergy2) < 0.0:
                            TrajMeanTime = np.inf
                            break
                        # Update the charging stations data:
                        if np.any(i2 == ChargingStationsData.ChargingStationsNodes):
                            arg_i = np.argwhere(i2 == ChargingStationsData.ChargingStationsNodes)[0][0]
                            ChargingStationsData.MaxChargingPotential += PltParams.BatteryCapacity - (TrajMeanEnergy+ChargingStationsData.MaxChargingPotential)
                            ChargingStationsData.EnergyEntered[arg_i] = TrajMeanEnergy

                    
                    ChargeTime = max(0.0,-(TrajMeanEnergy - NominalPlan.EnergyAlpha*np.sqrt(TrajSigmaEnergy2))/NominalPlan.StationRechargePower)
                    Cost = TrajMeanTime + ChargeTime + NominalPlan.TimeAlpha*np.sqrt(TrajSigmaTime2)
                    if Cost < CostGroup:
                        ChargeNeeded = ChargeTime*NominalPlan.StationRechargePower
                        iChargingStation = 0
                        ChargingNodeSquence = np.argsort(ChargingStationsData.EnergyEntered)[::-1]
                        while iChargingStation < len(ChargingStationsData.ChargingStationsNodes):
                            iCharge = ChargingNodeSquence[iChargingStation]
                            ChargingStationsData.EnergyExited[iCharge] = min(PltParams.BatteryCapacity*0.95,ChargingStationsData.EnergyEntered[iCharge] + ChargeNeeded)
                            ChargingStationsData.ChargingTime[iCharge] = (ChargingStationsData.EnergyExited[iCharge] - ChargingStationsData.EnergyEntered[iCharge])/NominalPlan.StationRechargePower
                            for ii in range(iChargingStation+1,len(ChargingStationsData.ChargingStationsNodes)):
                                ChargingStationsData.EnergyEntered[ChargingNodeSquence[ii]] += ChargingStationsData.ChargingTime[iCharge]*NominalPlan.StationRechargePower
                            ChargeNeeded -= ChargingStationsData.EnergyExited[iCharge] - ChargingStationsData.EnergyEntered[iCharge]
                            iChargingStation += 1
                        CostGroup = Cost
                        BestNodesTrajectoryGroup = PotentialTrajectory
                        BestChargingStationsData = ChargingStationsData
                        print("Connected Traj With Cost = {:}".format(Cost))
                        if isplot == True:
                            PlotSubGroups(NominalPlan, NodesTrajectoryGroup, NodesTrajectorySubGroup, PotentialTrajectory)



    return BestNodesTrajectoryGroup, CostGroup, BestChargingStationsData



def PlotSubGroups(NominalPlan: DataTypes.NominalPlanning, Group1, Group2, Group3 = []):

    plt.figure()
    leg_str = []
    leg_str.append('Group 1 - '+str(Group1))
    leg_str.append('Group 2 - '+str(Group2))
    leg_str.append('Group 3 - '+str(Group3))
    for i in Group1:
        plt.plot(NominalPlan.NodesPosition[i,0].T,NominalPlan.NodesPosition[i,1].T,'o',linewidth=10, color='r')
    for i in Group2:
        plt.plot(NominalPlan.NodesPosition[i,0].T,NominalPlan.NodesPosition[i,1].T,'o',linewidth=10, color='b')
    plt.grid('on')
    # plt.xlim((-100,100))
    # plt.ylim((-100,100))
    for i in range(len(Group1)-1):
        plt.arrow(NominalPlan.NodesPosition[Group1[i],0],NominalPlan.NodesPosition[Group1[i],1],NominalPlan.NodesPosition[Group1[i+1],0]-NominalPlan.NodesPosition[Group1[i],0],NominalPlan.NodesPosition[Group1[i+1],1]-NominalPlan.NodesPosition[Group1[i],1], width= 1, color='r')
    for i in range(len(Group2)-1):
        plt.arrow(NominalPlan.NodesPosition[Group2[i],0],NominalPlan.NodesPosition[Group2[i],1],NominalPlan.NodesPosition[Group2[i+1],0]-NominalPlan.NodesPosition[Group2[i],0],NominalPlan.NodesPosition[Group2[i+1],1]-NominalPlan.NodesPosition[Group2[i],1], width= 1, color='g')
    if len(Group3)>0:
        for i in range(len(Group3)-1):
            plt.arrow(NominalPlan.NodesPosition[Group3[i],0],NominalPlan.NodesPosition[Group3[i],1],NominalPlan.NodesPosition[Group3[i+1],0]-NominalPlan.NodesPosition[Group3[i],0],NominalPlan.NodesPosition[Group3[i+1],1]-NominalPlan.NodesPosition[Group3[i],1], width= 0.1, color='b')
    for i in Group1:
        colr = 'r' if i==0 else 'c'
        colr = 'k' if i in NominalPlan.ChargingStations else colr
        plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
    for i in Group2:
        colr = 'r' if i==0 else 'c'
        colr = 'k' if i in NominalPlan.ChargingStations else colr
        plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
    plt.legend(leg_str)

    plt.show()


def PlotGraph(iCar: list, NodesGroups: list, NodesTrajectory: list, Time: list, UncertainTime: list, Energy: list, UncertainEnergy:list, NominalPlan: DataTypes.NominalPlanning, PltParams: DataTypes.PlatformParams):

    col_vec = ['r','y','b','m','g','c','k']
    for m in range(len(iCar)):
        plt.figure()
        plt.subplot(3,1,(1,2))
        leg_str = []
        legi = []
        plt.plot(NominalPlan.NodesPosition[:,0].T,NominalPlan.NodesPosition[:,1].T,'o',linewidth=10, color=col_vec[m%len(col_vec)])
        for j in range(NominalPlan.NumberOfDepots):
            a = plt.plot(NominalPlan.NodesPosition[j,0].T,NominalPlan.NodesPosition[j,1].T,'o',linewidth=20, color='k')        
        legi.append(a[0])
        for j in NominalPlan.ChargingStations:
            a = plt.plot(NominalPlan.NodesPosition[j,0].T,NominalPlan.NodesPosition[j,1].T,'o',linewidth=20, color='g') 
        legi.append(a[0])      
        plt.grid('on')
        plt.xlim((np.min(NominalPlan.NodesPosition[:,0])-10,np.max(NominalPlan.NodesPosition[:,0])+10))
        plt.ylim((np.min(NominalPlan.NodesPosition[:,1])-10,np.max(NominalPlan.NodesPosition[:,1])+10))
        # if N<=10:
        #     for i in range(N):
        #         for j in range(i+1,N):
        #             plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),0.5*(NodesPosition[j,0]-NodesPosition[i,0]),0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
        #             plt.arrow(0.5*(NodesPosition[i,0]+NodesPosition[j,0]),0.5*(NodesPosition[i,1]+NodesPosition[j,1]),-0.5*(NodesPosition[j,0]-NodesPosition[i,0]),-0.5*(NodesPosition[j,1]-NodesPosition[i,1]), width= 0.01)
        for i in range(NominalPlan.N):
            colr = 'b' if i<NominalPlan.NumberOfDepots else 'c'
            colr = 'k' if i in NominalPlan.ChargingStations else colr
            plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=20)
        colr = col_vec[m%len(col_vec)]
        for i in range(len(NodesTrajectory)-1):
            j1 = NodesTrajectory[i,m]
            j2 = NodesTrajectory[i+1,m]
            if (NominalPlan.ReturnToBase==True and j1 > 0) or (NominalPlan.ReturnToBase==False and j2>0) or i==0:
                legi[m] = plt.arrow(NominalPlan.NodesPosition[j1,0],NominalPlan.NodesPosition[j1,1],NominalPlan.NodesPosition[j2,0]-NominalPlan.NodesPosition[j1,0],NominalPlan.NodesPosition[j2,1]-NominalPlan.NodesPosition[j1,1], width= 0.5, color=colr)
            if j2 < NominalPlan.NumberOfDepots:
                break
        if np.max(NodesTrajectory[:,m])>0:
            indx = np.argwhere(NodesTrajectory[:,m] > NominalPlan.NumberOfDepots)
            indx = indx[-1][0] if NominalPlan.ReturnToBase==False else indx[-1][0]+2
        else:
            indx = 2
        leg_str.append('Car '+str(iCar[m]+1)+" Number of Nodes: {}".format(indx-2))

        plt.legend(legi,leg_str)
        plt.title("Depot - Node 0, Car Num "+str(iCar[m]+1)+" Charging Stations: "+str(NominalPlan.ChargingStations))
        plt.subplot(3,2,5)
        leg_str = []
        colr = col_vec[m%len(col_vec)]
        if np.max(NodesTrajectory[:,m])>0:
            indx = np.argwhere(NodesTrajectory[:,m] > NominalPlan.NumberOfDepots)
            indx = indx[-1][0] if NominalPlan.ReturnToBase==False else indx[-1][0]+2
        else:
            indx = 0
        plt.plot(Energy[0:indx,m],'o-',color=colr)
        leg_str.append('Car '+str(m+1)+' Energy')
        for i in range(indx):
            plt.text(i,Energy[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
        for i in range(0,indx-1):
            plt.text(i+0.5,0.5*Energy[i,m]+0.5*Energy[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesEnergyTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)

        plt.title("Energy Profile - Car "+str(iCar[m]+1))
        plt.grid('on')
        plt.ylim((0,PltParams.BatteryCapacity))
        plt.legend(leg_str)
        plt.ylabel('Energy')
        colr = col_vec[m%len(col_vec)]
        if np.max(NodesTrajectory[:,m])>0:
            indx = np.argwhere(NodesTrajectory[:,m] > NominalPlan.NumberOfDepots)
            indx = indx[-1][0] if NominalPlan.ReturnToBase==False else indx[-1][0]+2
        else:
            indx = 0
        plt.plot(UncertainEnergy[0:indx,m],'-.',color=colr)


        plt.subplot(3,2,6)
        leg_str = []
        colr = col_vec[m%len(col_vec)]
        if np.max(NodesTrajectory[:,m])>0:
            indx = np.argwhere(NodesTrajectory[:,m] > NominalPlan.NumberOfDepots)
            indx = indx[-1][0] if NominalPlan.ReturnToBase==False else indx[-1][0]+2
        else:
            indx = 0
        plt.plot(Time[0:indx,m],'o-',color=colr)
        leg_str.append('Car '+str(m+1)+' Time')
        for i in range(indx):
            plt.text(i,Time[i,m]+0.1,"{:}".format(NodesTrajectory[i,m]), color=colr,fontsize=20)
        for i in range(0,indx-1):
            plt.text(i+0.5,0.5*Time[i,m]+0.5*Time[i+1,m]+0.1,"{:.3}".format(NominalPlan.NodesTimeOfTravel[NodesTrajectory[i,m],NodesTrajectory[i+1,m]]), color='c',fontsize=10)
        plt.title("Time Profile - Car "+str(iCar[m]+1))
        plt.grid('on')
        plt.legend(leg_str)
        plt.ylabel('Time [sec]')
        colr = col_vec[m%len(col_vec)]
        if np.max(NodesTrajectory[:,m])>0:
            indx = np.argwhere(NodesTrajectory[:,m] > NominalPlan.NumberOfDepots)
            indx = indx[-1][0] if NominalPlan.ReturnToBase==False else indx[-1][0]+2
        else:
            indx = 0
            plt.plot(UncertainTime[0:indx,m],'-.',color=colr)
