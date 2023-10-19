import numpy as np
import SimDataTypes as DataTypes
import matplotlib.pyplot as plt


def CreateGroupMatrix(Group, NodesTimeOfTravel):
    GroupMatrix = np.zeros((len(Group),len(Group)))
    for i in range(len(Group)):
        for j in range(len(Group)):
            GroupMatrix[i,j] = NodesTimeOfTravel[Group[i],Group[j]]
    return GroupMatrix

def CalcEntropy(NodesGroups, NodesTimeOfTravel, Method):
    NumberOfCars = len(NodesGroups)
    TimeEntropy_i = np.zeros(NumberOfCars)
    if Method == "Max_Eigenvalue":
        for iGroup in range(NumberOfCars):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
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
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            w, v = np.linalg.eig(GroupTimeMatrix)
            w = np.sort(np.abs(w))[::-1]
            ind = 3
            weights = np.array(range(1,ind+1))[::-1]
            TimeEntropy_i[iGroup] = len(w) * np.sum((weights*np.abs(w[0:ind]))**2)
    else:
        print("Error: Unknown method for calculating Entropy")
    
    TimeEntropy = np.sum(TimeEntropy_i)
    return TimeEntropy

def DivideNodesToGroups(NominalPlan: DataTypes.NominalPlanning , 
                        NumberOfCars: int, 
                        Method, 
                        MaximizeGroupSize: bool=False,
                        MustIncludeNodeZero: bool = True, 
                        ChargingStations: list = [],
                        MaxGroupSize: int = 12,
                        isplot: bool = False):

    N = NominalPlan.N
    NodesGroups = list()
    i1 = 0 if MustIncludeNodeZero==False else 1
    NodesTimeOfTravel = NominalPlan.NodesTimeOfTravel + NominalPlan.TimeAlpha * NominalPlan.TravelSigma

    if MaximizeGroupSize == True:
        NodesSet = set(range(N))
        for iGroup in range(NumberOfCars-1):
            Group = []
            if iGroup == 0:
                Group.append(0)
                NodesSet.remove(0)
            while len(Group) < MaxGroupSize and len(NodesSet)>0:
                Entropy = np.zeros((len(NodesSet),))
                for i in range(len(NodesSet)):
                    iNode = list(NodesSet)[i]
                    Group.append(iNode)
                    Entropy[i] = CalcEntropy((Group,), NodesTimeOfTravel, Method)
                    Group.remove(iNode)
                Group.append(list(NodesSet)[np.argmin(Entropy)])
                NodesSet = NodesSet - set(Group)
                if len(NodesSet) == 2 and iGroup == NumberOfCars-2:
                    break
            NodesGroups.append(np.array(Group))
        
        NodesGroups.append(np.array(list(NodesSet)))


    else:
    # Initialize the groups by random:
        RndGroups = np.random.randint(0,NumberOfCars,size=(N,))
        if MustIncludeNodeZero==True:
            dAngle = 2*np.pi/NumberOfCars
            NodesAngles = np.zeros((N,))
            NodesAngles[1:] = np.arctan2(NominalPlan.NodesPosition[1:,1],NominalPlan.NodesPosition[1:,0])
            RndGroups[0] = 0
            for j in range(1,N):
                for i in range(NumberOfCars):
                    if NodesAngles[j]>=i*dAngle and NodesAngles[j]<(i+1)*dAngle:
                        RndGroups[j] = i
                        break
        else:
            # Divide Groups by appending closest nodes:
            RndGroups = np.zeros((N,), dtype=int)
            NodesSet = set(range(NominalPlan.N))
            iGroup = 0
            GroupNumber = 0
            RndGroups[iGroup] = GroupNumber
            NodesSet.remove(iGroup)
            Ngroup = 1
            while len(NodesSet)>0:
                iGroup = list(NodesSet)[np.argmin(NodesTimeOfTravel[iGroup,:][list(NodesSet)])]
                RndGroups[iGroup] = GroupNumber
                NodesSet.remove(iGroup)
                Ngroup += 1
                if Ngroup >= np.ceil(N/NumberOfCars):
                    GroupNumber += 1
                    Ngroup = 0
            

        for i in range(NumberOfCars):
            if MustIncludeNodeZero==True:
                arg = np.argwhere(RndGroups[1:]==i)+1
            else:
                arg = np.argwhere(RndGroups==i)
            Group = np.asarray(arg.T,dtype=int).reshape(arg.shape[0],) # initialize the list
            if MustIncludeNodeZero==True:
                Group = np.append(Group,0) # add the depot
            elif 0 in Group:
                    Group = np.delete(Group, np.argwhere(Group==0))
            Group.sort()
            NodesGroups.append(Group) # the depot

        # Minimize "Entropy"
        print("Initial Entropy: ", CalcEntropy(NodesGroups, NodesTimeOfTravel, Method))
        for iter in range(100):
            CurNodesGroups = NodesGroups.copy()
            GroupChanged = False
            for iGroup in range(NumberOfCars):
                Entropy = np.zeros((NumberOfCars,))
                for i in range(i1,len(CurNodesGroups[iGroup])):
                    CurNode = CurNodesGroups[iGroup][i]
                    for jGroup in range(NumberOfCars):
                        arg_iGroup = np.argwhere(NodesGroups[iGroup]==CurNode)
                        NodesGroups[iGroup] = np.delete(NodesGroups[iGroup], arg_iGroup)
                        NodesGroups[jGroup] = np.append(NodesGroups[jGroup], CurNode)
                        if len(NodesGroups[jGroup]) > NominalPlan.MaxNumberOfNodesPerCar:# and len(CurNodesGroups[jGroup]) > len(NodesGroups[jGroup]):
                            Entropy[jGroup] = 1e10
                        else:
                            Entropy[jGroup] = CalcEntropy(NodesGroups, NodesTimeOfTravel, Method)
                        arg_jGroup = np.argwhere(NodesGroups[jGroup]==CurNode)
                        NodesGroups[jGroup] = np.delete(NodesGroups[jGroup], arg_jGroup)
                        NodesGroups[iGroup] = np.append(NodesGroups[iGroup], CurNode)
                        
                    Group_MinEntropy = np.argmin(Entropy)
                    arg_iGroup = np.argwhere(NodesGroups[iGroup]==CurNode)
                    NodesGroups[iGroup] = np.delete(NodesGroups[iGroup], arg_iGroup)
                    NodesGroups[Group_MinEntropy] = np.append(NodesGroups[Group_MinEntropy], CurNode)
                    if Group_MinEntropy != iGroup:
                        GroupChanged = True
                NodesGroups[iGroup] = np.sort(NodesGroups[iGroup])
            print("iteration = "+str(iter)+", Entropy = {:}".format(Entropy[Group_MinEntropy]))
            if not GroupChanged:
                break
        
        # Decide Which group should get the depot:
        if MustIncludeNodeZero==False:
            Entropy = np.zeros((NumberOfCars,))
            for i in range(NumberOfCars):
                Entropy[i] = np.mean(NodesTimeOfTravel[0,:][NodesGroups[i]])
            Group_MinEntropy = np.argmin(Entropy)
            NodesGroups[Group_MinEntropy] = np.append(NodesGroups[Group_MinEntropy], 0)
            NodesGroups[Group_MinEntropy] = np.sort(NodesGroups[Group_MinEntropy])

    # Try Switch Nodes between groups:
    for iter in range(100):
        GroupChanged = False
        CurNodesGroups = NodesGroups.copy()
        Entropy = CalcEntropy(NodesGroups, NodesTimeOfTravel, Method)
        for iGroup in range(NumberOfCars):
            for jGroup in range(iGroup+1,NumberOfCars):
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
                        CurEntropy = CalcEntropy(CurNodesGroups, NodesTimeOfTravel, Method)
                        if CurEntropy >= Entropy:
                            CurNodesGroups = NodesGroups.copy()
                        else:
                            Entropy = CurEntropy
                            GroupChanged = True
                            for i in range(NumberOfCars):
                                CurNodesGroups[i] = np.sort(CurNodesGroups[i])
                            NodesGroups = CurNodesGroups.copy()
        print("iteration = "+str(iter)+", Entropy = {:}".format(Entropy))
        if not GroupChanged:
            break

    # Make sure that the first group has the depot:
    if NodesGroups[0][0] != 0:
        for i in range(1,NumberOfCars):
            if NodesGroups[i][0] == 0:
                Temp = NodesGroups[0].copy()
                NodesGroups[0] = NodesGroups[i].copy()
                NodesGroups[i] = Temp.copy()
                break
    # Organize the groups:
    CurGroupIntegration = NodesGroups[0]
    for i in range(1,NumberOfCars-1):
        Entropy = np.zeros((NumberOfCars,))+np.inf
        for j in range(NumberOfCars-i):
            Group = np.append(CurGroupIntegration, NodesGroups[j+i])
            Entropy[i+j] = CalcEntropy([Group], NodesTimeOfTravel, Method)
        Group_MinEntropy = np.argmin(Entropy)
        # Set Group_MinEntropy as Group number i:
        Temp = NodesGroups[i].copy()
        NodesGroups[i] = NodesGroups[Group_MinEntropy].copy()
        NodesGroups[Group_MinEntropy] = Temp.copy()
        # Update CurGroupIntegration:
        CurGroupIntegration = np.append(CurGroupIntegration, NodesGroups[i])
            
    # Final Enthropy:
    for i in range(NumberOfCars):
        NodesGroups_i = []
        NodesGroups_i.append(NodesGroups[i])
        # print("Final Entropy Group", i,": ", CalcEntropy(NodesGroups_i, NodesTimeOfTravel, Method))    

    # Print summary:
    print("Final Entropy: ", CalcEntropy(NodesGroups, NodesTimeOfTravel, Method))    
    for i in range(NumberOfCars):
        GroupTimeMatrix = CreateGroupMatrix(NodesGroups[i], NodesTimeOfTravel)
        w, v = np.linalg.eig(GroupTimeMatrix)
        w = np.sort(np.abs(w))[::-1]
        print("Group {:} - Number of Nodes: {:}, Entropy: {:}, Max Eigenvalue: {:}".format(i, len(NodesGroups[i]), CalcEntropy([NodesGroups[i]], NodesTimeOfTravel, Method), np.abs(w[0:3])))

# Plot the groups
    if np.max(NominalPlan.NodesPosition)>0 and isplot==True:
        col_vec = ['m','y','b','r','g','c','k']
        leg_str = list()
        plt.figure()
        if MustIncludeNodeZero==True:
            plt.scatter(NominalPlan.NodesPosition[0,0], NominalPlan.NodesPosition[0,1], c='k', s=50)
            leg_str.append('Depot')
        for i in range(NumberOfCars):
            plt.scatter(NominalPlan.NodesPosition[NodesGroups[i][i1:],0], NominalPlan.NodesPosition[NodesGroups[i][i1:],1], s=50, c=col_vec[i%len(col_vec)])
            leg_str.append('Group '+str(i)+" Number of Nodes: {}".format(len(NodesGroups[i])-1))
        for i in ChargingStations:
            plt.scatter(NominalPlan.NodesPosition[i,0], NominalPlan.NodesPosition[i,1], c='k', s=15)
        leg_str.append('Charging Station')
        plt.legend(leg_str)
        for i in range(N):
            colr = 'r' if i==0 else 'c'
            colr = 'k' if i in ChargingStations else colr
            plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}".format(i), color=colr,fontsize=30)
        plt.xlim((-100,100))
        plt.ylim((-100,100))
        plt.grid()
        plt.show()

    return NodesGroups

def CreateSubPlanFromPlan (NominalPlan: DataTypes.NominalPlanning, NodesGroups):
    n = len(NodesGroups)
    NominalPlanGroup = DataTypes.NominalPlanning(n)
    NominalPlanGroup.ChargingStations = []
    for i in range(n):
        NominalPlanGroup.NodesPosition[i,:] = NominalPlan.NodesPosition[NodesGroups[i],:]
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
        if np.any(NodesGroups[i] == NominalPlan.ChargingStations):
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