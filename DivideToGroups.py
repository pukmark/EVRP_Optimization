import numpy as np
import SimDataTypes as DataTypes
import matplotlib.pyplot as plt
import time
from copy import deepcopy
from multiprocessing import Pool, Lock, Value
import itertools
import xmltodict as xml


def normalize(x):
    fac = np.linalg.norm(x)
    x_n = x / fac
    return fac, x_n

def CalcLargestEigenvalue(A):
    if A.shape[0] == 1:
        return A[0][0]
    v = np.ones((A.shape[0],1))/np.sqrt(A.shape[0])
    v = A@v
    lam_pre, v = normalize(v)
    while True:
        v = A@v
        lam, v = normalize(v)
        if abs((lam-lam_pre)/lam_pre) < 1e-3:
            return abs(lam)
        lam_pre = lam


def CalcTotalEntropy(Entropy_i: list):
    return np.linalg.norm(Entropy_i, ord=1)

def CreateGroupMatrix(Group, NodesTimeOfTravel):
    GroupMatrix = np.zeros((len(Group),len(Group)))
    for i in range(len(Group)):
        for j in range(len(Group)):
            GroupMatrix[i,j] = NodesTimeOfTravel[Group[i],Group[j]]
    return GroupMatrix

def CalcEntropy(NodesGroups, NodesTimeOfTravel, Method, GroupsChanged: list = [], Entropy_i: list = []):
    if len(GroupsChanged)== 0:
        GroupsChanged = range(len(NodesGroups))
    if len(Entropy_i) == 0:
        Entropy_i = np.zeros(len(NodesGroups))
    if Method == "Max_EigenvalueN":
        for iGroup in GroupsChanged:
            if len(NodesGroups[iGroup]) == 0:
                Entropy_i[iGroup] = 1e8
                continue
            if len(NodesGroups[iGroup]) == 1:
                Entropy_i[iGroup] = 0.0
                continue
            w = CalcLargestEigenvalue(NodesTimeOfTravel[NodesGroups[iGroup],:][:,NodesGroups[iGroup]])
            Entropy_i[iGroup] = w * len(NodesGroups[iGroup])
    elif Method == "Frobenius":
        for iGroup in GroupsChanged:
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            Entropy_i[iGroup] = np.linalg.norm(GroupTimeMatrix, 'fro')
    elif Method == "Mean_MaxRow":
        for iGroup in GroupsChanged:
            Entropy_i[iGroup] = np.sum(np.max(NodesTimeOfTravel[NodesGroups[iGroup],:][:,NodesGroups[iGroup]], axis=1))**2
    elif Method == "Sum_AbsEigenvalue":
        for iGroup in GroupsChanged:
            # GroupTimeMatrix = CreateGroupMatrix(NodesGroups[iGroup], NodesTimeOfTravel)
            w, v = np.linalg.eig(NodesTimeOfTravel[NodesGroups[iGroup],:][:,NodesGroups[iGroup]])
            Entropy_i[iGroup] = np.sum(np.abs(w))
    elif Method == "PartialMax_Eigenvalue":
        for iGroup in GroupsChanged:
            GroupTimeMatrix = CreateGroupMatrix(list(set(NodesGroups[iGroup]) - set({0})), NodesTimeOfTravel)
            w, _ = np.linalg.eig(GroupTimeMatrix)
            w = np.sort(np.abs(w))[::-1]
            ind = min(3,len(w))
            weights = np.array(range(1,ind+1))[::-1]
            Entropy_i[iGroup] = len(w) * np.sum((np.abs(w[0:ind]))**2)
    elif Method == "Mean_Method":
        for iGroup in GroupsChanged:
            Entropy_i[iGroup] = np.sum(np.mean(NodesTimeOfTravel[NodesGroups[iGroup],:][:,NodesGroups[iGroup]], axis=0))**2
    # elif Method == "Greedy_Method":
        
    #     for iGroup in GroupsChanged:
    #         Group = set(GroupsChanged)
    #         Time = 0.0
    #         iGroup = Group[0]
    #         CostMat = NodesTimeOfTravel[NodesGroups[iGroup],:][:,NodesGroups[iGroup]]
    #         while len(Group)>0:
    #             i
    #             Time += np.argmin(NodesTimeOfTravel[Group[0],Group])

    else:
        print("Error: Unknown method for calculating Entropy")
    
    return Entropy_i

def DivideNodesToGroups(NominalPlan: DataTypes.NominalPlanning , 
                        Method = "Max_Eigenvalue", 
                        ClusterSubGroups: bool=False,
                        MaxGroupSize: int = 12,
                        isplot: bool = False):

    N = NominalPlan.N
    M = NominalPlan.NumberOfCars
    NodesGroups = list()
    i1 = 0 if ClusterSubGroups==True else 1
    if len(NominalPlan.ChargingRateProfile) == 0:
        NodesTimeOfTravel = NominalPlan.NodesTimeOfTravel + NominalPlan.TimeAlpha * NominalPlan.TravelSigma
    else:
        NodesTimeOfTravel = NominalPlan.NodesTimeOfTravel + NominalPlan.TimeAlpha * NominalPlan.TravelSigma - (NominalPlan.NodesEnergyTravel - NominalPlan.EnergyAlpha * NominalPlan.NodesEnergyTravelSigma)/NominalPlan.ChargingRateProfile[0][0,-1]

    if ClusterSubGroups == True:
        NodesSet = set(range(N)) - set(NominalPlan.ChargingStations) - set(NominalPlan.CarsInDepots)
        NumSubGroups = int(len(NominalPlan.CarsInDepots))
        M = NumSubGroups
        for iGroup in range(NumSubGroups-1):
            Group = []
            while len(Group) < MaxGroupSize and len(NodesSet)>0:
                Entropy = np.zeros((len(NodesSet),))
                for i in range(len(NodesSet)):
                    iNode = list(NodesSet)[i]
                    Group.append(iNode)
                    Entropy[i] = CalcEntropy((Group,), NodesTimeOfTravel, Method)
                    Group.remove(iNode)
                Group.append(list(NodesSet)[np.argmin(Entropy)])
                NodesSet = NodesSet - set(Group)
                if len(NodesSet) == 2 and iGroup == NumSubGroups-2:
                    break
            NodesGroups.append(np.array(Group, dtype=int))
        
        NodesGroups.append(np.array(list(NodesSet), dtype=int))


    else:
    # Initialize the groups by Closest Neighbor:
        CustomersNodes = set(range(N)) - set(NominalPlan.CarsInDepots) - set(NominalPlan.ChargingStations)
        for i in NominalPlan.CarsInDepots:
            NodesGroups.append(np.array([i]))
            TimeMat = NodesTimeOfTravel[list(CustomersNodes),:][:,list(CustomersNodes)]
            max_Time_i = np.argmax(np.max(TimeMat, axis=1))
            NodesGroups[-1] = np.append(NodesGroups[-1], list(CustomersNodes)[max_Time_i])
            CustomersNodes = CustomersNodes - set([list(CustomersNodes)[max_Time_i]])
        for i in range(1,M):
            Dist2 = np.zeros((len(CustomersNodes),))
            for j in range(i):
                Dist2 += np.sqrt(NodesTimeOfTravel[NodesGroups[j][-1],list(CustomersNodes)])
            max_Dist2_i = np.argmax(Dist2)
            NodesGroups[i] = np.append(NodesGroups[i], list(CustomersNodes)[max_Dist2_i])
            CustomersNodes = CustomersNodes - set([list(CustomersNodes)[max_Dist2_i]])
            
        argDemand = np.argsort(NominalPlan.LoadDemand)[::-1]
        for iNode in argDemand:
            if iNode not in CustomersNodes:
                continue
            TimeToNode = np.zeros((M,))
            for iGroup in range(M):
                TimeToNode[iGroup] = CalcEntropy([np.append(NodesGroups[iGroup], iNode)], NodesTimeOfTravel, Method)
            Group_MinTime = np.argsort(TimeToNode)
            for iGroup in Group_MinTime:
                if (NodesGroups[iGroup].shape[0]+1 <= NominalPlan.MaxNumberOfNodesPerCar) and (NominalPlan.LoadDemand[iNode]+np.sum(NominalPlan.LoadDemand[NodesGroups[iGroup]])<=NominalPlan.LoadCapacity):
                    NodesGroups[iGroup] = np.append(NodesGroups[iGroup], iNode)
                    break
                if iGroup == Group_MinTime[-1]:
                    np.append(NodesGroups[Group_MinTime[0]], iNode)
                    NCust = len(set(range(N)) - set(NominalPlan.CarsInDepots) - set(NominalPlan.ChargingStations))
                    print("Error: Node {:} can't be added to any group".format(iNode)," Only {:3.2f}[%] nodes assingned".format(100*(NCust-len(CustomersNodes))/NCust))
                    return []

    # Load file for initial groups:
    if ClusterSubGroups==False:
        try:
            NodesGroupsGuess = []
            # open file and read the content in a list
            SumNodes = 0
            with open(r'./Groups.txt', 'r') as fp:
                for line in fp:
                    x = line[:-1]
                    # add current item to the list
                    x = x.strip('][').split(', ')
                    NodesGroupsGuess.append([int(i) for i in x])
                    SumNodes += len(NodesGroupsGuess[-1])
                    if np.sum(NominalPlan.LoadDemand[NodesGroupsGuess[-1]])>NominalPlan.LoadCapacity:
                        break
                if (len(NodesGroupsGuess) == len(NodesGroups)) and (SumNodes == N-len(NominalPlan.ChargingStations)+len(NodesGroups)-1):
                    NodesGroups = NodesGroupsGuess.copy()
        except:
            print("Error: Can't open file 'Groups.txt'")
    
    # if ClusterSubGroups==False and len(NominalPlan.ChargingStations)>0:
    #     NodesGroups = []
    #     NodesGroups.append([ 0 ,  5,7,4])
    #     NodesGroups.append([ 1,6,9,10,2])

    # Calculate the demand of each group:
    GroupDemand = np.zeros((M,))
    CustomersNodes = set(range(N)) - set(NominalPlan.CarsInDepots) - set(NominalPlan.ChargingStations)
    for i in range(M):
        GroupDemand[i] = np.sum(NominalPlan.LoadDemand[NodesGroups[i]])
        CustomersNodes = CustomersNodes - set(NodesGroups[i])
        if GroupDemand[i] > NominalPlan.LoadCapacity:
            print("Error: Group {:} has more load than capacity".format(i))
            return []
    if len(CustomersNodes) > 0:
        print("Error: Not all nodes are assigned to groups")
        return []
    


    BestEntropy_i = CalcEntropy(NodesGroups.copy(), NodesTimeOfTravel, Method)
    BestEntropy = CalcTotalEntropy(BestEntropy_i)
    Entropy_prev = BestEntropy
    EntropyChangeCriteria = 0.00
    t0 = time.time()
    if ClusterSubGroups==False:
        print("Initial Entropy: {:4.3e}".format(BestEntropy),", Number Of Custemers: ", N-len(NominalPlan.ChargingStations)," Number Of CS: ", len(NominalPlan.ChargingStations)," Number Of Cars: ", M)
    
    # Move nodes between groups:
    GroupChanged_new = np.ones((M,)).astype(bool)
    for iter in range(100):
        t0_iter = time.time()
        GroupChanged = GroupChanged_new.copy()
        GroupChanged_new = np.zeros((M,)).astype(bool)
        if Entropy_prev>0.0:
            BestEntropy_i = CalcEntropy(NodesGroups.copy(), NodesTimeOfTravel, Method)
            BestEntropy = CalcTotalEntropy(BestEntropy_i)
            CurNodesGroups = NodesGroups.copy()
            GroupEntropySort_i = np.argsort(BestEntropy_i)[::-1]
            for iGroup in GroupEntropySort_i:
                if len(CurNodesGroups[iGroup]) <= 2: continue
                for CurNode in CurNodesGroups[iGroup][i1:]:
                    Entropy = np.ones((M,))*np.inf
                    if len(CurNodesGroups[iGroup]) <= 2: continue
                    for jGroup in range(M):
                        if (iGroup == jGroup):
                            Entropy[jGroup] = BestEntropy
                            continue
                        if (GroupChanged[iGroup]==False and GroupChanged[jGroup]==False) or (len(CurNodesGroups[jGroup])+1 > MaxGroupSize) or GroupDemand[jGroup]+NominalPlan.LoadDemand[CurNode]>NominalPlan.LoadCapacity:
                            continue
                        arg_iGroup = np.argwhere(CurNodesGroups[iGroup]==CurNode)
                        CurNodesGroups[iGroup] = np.delete(CurNodesGroups[iGroup], arg_iGroup)
                        CurNodesGroups[jGroup] = np.append(CurNodesGroups[jGroup], CurNode)
                        Entropy[jGroup] = CalcTotalEntropy(CalcEntropy(CurNodesGroups.copy(), NodesTimeOfTravel, Method, [iGroup, jGroup], BestEntropy_i.copy()))
                        arg_jGroup = np.argwhere(CurNodesGroups[jGroup]==CurNode)
                        CurNodesGroups[jGroup] = np.delete(CurNodesGroups[jGroup], arg_jGroup)
                        CurNodesGroups[iGroup] = np.append(CurNodesGroups[iGroup], CurNode)
                    Group_MinEntropy = np.argmin(Entropy)
                    BestEntropy = Entropy[Group_MinEntropy]
                    arg_iGroup = np.argwhere(CurNodesGroups[iGroup]==CurNode)
                    CurNodesGroups[iGroup] = np.delete(CurNodesGroups[iGroup], arg_iGroup)
                    CurNodesGroups[Group_MinEntropy] = np.append(CurNodesGroups[Group_MinEntropy], CurNode)
                    if Group_MinEntropy != iGroup:
                        GroupDemand[iGroup] -= NominalPlan.LoadDemand[CurNode]
                        GroupDemand[Group_MinEntropy] += NominalPlan.LoadDemand[CurNode]
                        NodesGroups = CurNodesGroups.copy()
                        BestEntropy_i = CalcEntropy(NodesGroups.copy(), NodesTimeOfTravel, Method)
                        GroupChanged_new[iGroup] = True
                        GroupChanged_new[Group_MinEntropy] = True
                CurNodesGroups[iGroup] = np.sort(CurNodesGroups[iGroup])
            for i in range(M):
                GroupChanged[i] = True if (GroupChanged_new[i]==True or GroupChanged[i]==True) else False
                BestEntropy_i[i] = CalcEntropy([NodesGroups[i]], NodesTimeOfTravel, Method)
        # if BestEntropy < Entropy_prev:
        #     Entropy_prev = BestEntropy
        #     continue

        # Switch between groups:
        GroupEntropySort_i = np.argsort(BestEntropy_i)[::-1]
        CurNodesGroups = NodesGroups.copy()
        if Entropy_prev>0.0:
            for i in range(M):
                iGroup = GroupEntropySort_i[i]
                for j in range(i+1,M):
                    jGroup = GroupEntropySort_i[j]
                    if GroupChanged[iGroup]==False or GroupChanged[jGroup]==False:
                        continue
                    SubNodesGroups = []
                    SubNodesGroups.append(NodesGroups[iGroup])
                    SubNodesGroups.append(NodesGroups[jGroup])
                    CurSubNodesGroups = SubNodesGroups.copy()
                    BestEntropy = CalcTotalEntropy(CalcEntropy(SubNodesGroups, NodesTimeOfTravel, Method))
                    for iNode in SubNodesGroups[0]:
                        for jNode in SubNodesGroups[1]:
                            if iNode in NominalPlan.CarsInDepots and jNode not in NominalPlan.CarsInDepots:
                                continue
                            if jNode in NominalPlan.CarsInDepots and iNode not in NominalPlan.CarsInDepots:
                                continue
                            # if iNode==0 or jNode==0: continue
                            if jNode in CurSubNodesGroups[0] or iNode in CurSubNodesGroups[1]: 
                                continue
                            if GroupDemand[iGroup]-NominalPlan.LoadDemand[iNode]+NominalPlan.LoadDemand[jNode]>NominalPlan.LoadCapacity or GroupDemand[jGroup]-NominalPlan.LoadDemand[jNode]+NominalPlan.LoadDemand[iNode]>NominalPlan.LoadCapacity:
                                continue
                            arg_iNode = np.argwhere(CurSubNodesGroups[0]==iNode)
                            arg_jNode = np.argwhere(CurSubNodesGroups[1]==jNode)
                            CurSubNodesGroups[0] = np.delete(CurSubNodesGroups[0], arg_iNode)
                            CurSubNodesGroups[1] = np.delete(CurSubNodesGroups[1], arg_jNode)
                            CurSubNodesGroups[0] = np.append(CurSubNodesGroups[0], jNode)
                            CurSubNodesGroups[1] = np.append(CurSubNodesGroups[1], iNode)
                            CurEntropy = CalcTotalEntropy(CalcEntropy(CurSubNodesGroups, NodesTimeOfTravel, Method))
                            if CurEntropy >= BestEntropy:
                                CurSubNodesGroups = SubNodesGroups.copy()
                            else:
                                BestEntropy = CurEntropy
                                for i in range(2):
                                    SubNodesGroups[i] = np.sort(CurSubNodesGroups[i])
                                NodesGroups[iGroup] = SubNodesGroups[0]
                                NodesGroups[jGroup] = SubNodesGroups[1]
                                GroupChanged_new[iGroup] = True
                                GroupChanged_new[jGroup] = True
                                GroupDemand[iGroup] -= NominalPlan.LoadDemand[iNode] - NominalPlan.LoadDemand[jNode]
                                GroupDemand[jGroup] += NominalPlan.LoadDemand[iNode] - NominalPlan.LoadDemand[jNode]
            
            for i in range(M):
                GroupChanged[i] = True if (GroupChanged_new[i]==True or GroupChanged[i]==True) else False
                BestEntropy_i[i] = CalcEntropy([NodesGroups[i]], NodesTimeOfTravel, Method)
            BestEntropy = CalcTotalEntropy(BestEntropy_i)

        # if Entropy_prev > BestEntropy:
        #     Entropy_prev = BestEntropy
        #     continue
        # BestEntropy = CalcTotalEntropy(CalcEntropy(NodesGroups, NodesTimeOfTravel, Method))

        # Switch between groups 1 to 2:
        CurNodesGroups = NodesGroups.copy()
        Group_i = np.argsort(BestEntropy_i)[::-1]
        Group_j = np.argsort(BestEntropy_i)
        for iGroup in Group_i:
            for jGroup in Group_j:
                if len(NodesGroups[jGroup]) <= 2: continue
                if (iGroup == jGroup) or (GroupChanged[iGroup]==False and GroupChanged[jGroup]==False): continue
                SubNodesGroups = []
                SubNodesGroups.append(NodesGroups[iGroup])
                SubNodesGroups.append(NodesGroups[jGroup])
                CurSubNodesGroups = SubNodesGroups.copy()
                BestEntropy = CalcTotalEntropy(CalcEntropy(SubNodesGroups, NodesTimeOfTravel, Method))
                premuts_j = list(itertools.combinations(CurSubNodesGroups[1],2))
                for iNode in SubNodesGroups[0]:
                    for premut in premuts_j:
                            jNode = premut[0]
                            kNode = premut[1]
                            if jNode==kNode: continue
                            if iNode in NominalPlan.CarsInDepots and (jNode not in NominalPlan.CarsInDepots and kNode not in NominalPlan.CarsInDepots):
                                continue
                            if (jNode in NominalPlan.CarsInDepots or kNode in NominalPlan.CarsInDepots) and iNode not in NominalPlan.CarsInDepots:
                                continue
                            if jNode in CurSubNodesGroups[0] or kNode in CurSubNodesGroups[0] or iNode in CurSubNodesGroups[1]: 
                                continue
                            if GroupDemand[iGroup]-NominalPlan.LoadDemand[iNode]+NominalPlan.LoadDemand[jNode]+NominalPlan.LoadDemand[kNode]>NominalPlan.LoadCapacity or GroupDemand[jGroup]-NominalPlan.LoadDemand[jNode]-NominalPlan.LoadDemand[kNode]+NominalPlan.LoadDemand[iNode]>NominalPlan.LoadCapacity:
                                continue
                            arg_iNode = np.argwhere(CurSubNodesGroups[0]==iNode)
                            arg_jNode = np.argwhere(CurSubNodesGroups[1]==jNode)
                            CurSubNodesGroups[0] = np.delete(CurSubNodesGroups[0], arg_iNode)
                            CurSubNodesGroups[1] = np.delete(CurSubNodesGroups[1], arg_jNode)
                            arg_kNode = np.argwhere(CurSubNodesGroups[1]==kNode)
                            CurSubNodesGroups[1] = np.delete(CurSubNodesGroups[1], arg_kNode)
                            CurSubNodesGroups[0] = np.append(CurSubNodesGroups[0], jNode)
                            CurSubNodesGroups[0] = np.append(CurSubNodesGroups[0], kNode)
                            CurSubNodesGroups[1] = np.append(CurSubNodesGroups[1], iNode)
                            CurEntropy = CalcTotalEntropy(CalcEntropy(CurSubNodesGroups, NodesTimeOfTravel, Method))
                            if CurEntropy >= BestEntropy:
                                CurSubNodesGroups = SubNodesGroups.copy()
                            else:
                                BestEntropy = CurEntropy
                                GroupChanged_new[iGroup] = True
                                GroupChanged_new[jGroup] = True
                                for i in range(2):
                                    SubNodesGroups[i] = np.sort(CurSubNodesGroups[i])
                                NodesGroups[iGroup] = SubNodesGroups[0].copy()
                                NodesGroups[jGroup] = SubNodesGroups[1].copy()
                                GroupDemand[iGroup] -= NominalPlan.LoadDemand[iNode] - NominalPlan.LoadDemand[jNode] - NominalPlan.LoadDemand[kNode]
                                GroupDemand[jGroup] += NominalPlan.LoadDemand[iNode] - NominalPlan.LoadDemand[jNode] - NominalPlan.LoadDemand[kNode]
                                break
        BestEntropy = CalcTotalEntropy(CalcEntropy(NodesGroups, NodesTimeOfTravel, Method))
        if ClusterSubGroups==False:
            print("iteration = "+str(iter)+", Entropy = {:4.3e}".format(BestEntropy), ", Iteration Clustering Time = {:3.2f}".format(time.time()-t0_iter)+"[sec]")
        if (Entropy_prev-BestEntropy)/max(Entropy_prev,1e-4) <= EntropyChangeCriteria:
            break
        Entropy_prev = BestEntropy

    if ClusterSubGroups==False:
        print("Total Clustering Time = {:3.2f}".format(time.time()-t0)+"[sec]. All groups have less load than capacity")


    # if ClusterSubGroups==False:
    #     NodesGroups = []
    #     NodesGroups.append([ 0 ,10,  11])
    #     NodesGroups.append([ 0, 13,  9,  8,  5,  4, 21,  7])
    #     NodesGroups.append([ 0 ,18 ,19, 20, 22 ,17, 14, 15 ,16 , 3, 2 , 1 , 6, 12])

    # Save the groups to file for future use:
    if ClusterSubGroups==False:
        with open(r'./Groups.txt', 'w') as fp:
            for Group in NodesGroups:
                # write each item on a new line
                if type(Group) == np.ndarray:
                    fp.write("%s\n" % Group.tolist())
                else:
                    fp.write("%s\n" % Group)
            fp.close()

    # Print summary:
    if isplot==True:
        print("Final Entropy: {:4.3e}".format(CalcTotalEntropy(CalcEntropy(NodesGroups, NodesTimeOfTravel, Method))))    
        for i in range(M):
            GroupTimeMatrix = CreateGroupMatrix(NodesGroups[i], NodesTimeOfTravel)
            w, v = np.linalg.eig(GroupTimeMatrix)
            w = np.sort(np.abs(w))[::-1]
            print("Group {:} - Number of Nodes: {:}, Entropy: {:}, Max Eigenvalue: {:}".format(i, len(NodesGroups[i]), CalcEntropy([NodesGroups[i]], NodesTimeOfTravel, Method), np.abs(w[0:3])))

    for i in range(M):
        NodesGroups[i] = list(NodesGroups[i])

    if ClusterSubGroups==False:
        NodesWitoutCS = set(range(N)) - set(NominalPlan.ChargingStations)
        if len(NominalPlan.ChargingStations) > 0:
            for i in range(M):
                NodesWitoutCS = set(NodesGroups[i]) - set(NominalPlan.ChargingStations)
                MaxEnergy = np.min(NominalPlan.NodesEnergyTravel[NodesGroups[i][0],list(NodesWitoutCS-set([NodesGroups[i][0]]))])
                MaxEnergy += np.min(NominalPlan.NodesEnergyTravel[list(NodesWitoutCS-set([NodesGroups[i][0]])),NodesGroups[i][0]])
                for j in range(1,len(NodesGroups[i])):
                    MaxEnergy += np.min(NominalPlan.NodesEnergyTravel[NodesGroups[i][j],list(NodesWitoutCS-set([NodesGroups[i][0],j]))])
                
                NumberOfMaxCS = MaxEnergy + NominalPlan.BatteryCapacity
                NumOfCS = np.round(min(max(2.0,-NumberOfMaxCS/NominalPlan.BatteryCapacity),len(NominalPlan.ChargingStations)))

                CS_list = []
                for j in range(int(NumOfCS)):
                    if j==0:
                        NodesList = NodesGroups[i][1:].copy()
                    else:
                        NodesList = NodesGroups[i].copy()
                # Mean Time to CS:
                    MeanTimeToCS = np.inf*np.ones((len(NominalPlan.ChargingStations),))
                    for iCS in range(len(NominalPlan.ChargingStations)):
                        if NominalPlan.ChargingStations[iCS] in NodesGroups[i]:
                            continue
                        TimeFromNodesToCS = NodesTimeOfTravel[NodesList,NominalPlan.ChargingStations[iCS]]
                        if j>1:
                            for k in range(j):
                                for iNode in range(len(NodesList)):
                                    if NodesTimeOfTravel[NodesList[iNode],CS_list[k]] < TimeFromNodesToCS[iNode]:
                                        TimeFromNodesToCS[iNode] = TimeFromNodesToCS[iNode]
                        if j>0 and np.any(TimeFromNodesToCS==NodesTimeOfTravel[NodesList,NominalPlan.ChargingStations[iCS]])==False:
                            continue
                        MeanTimeToCS[iCS] = np.linalg.norm(TimeFromNodesToCS,ord=2)

                    j_CS = np.argsort(MeanTimeToCS)
                    if MeanTimeToCS[j_CS[0]] == np.inf:
                        break
                    NodesGroups[i].append(NominalPlan.ChargingStations[j_CS[0]])
                    CS_list.append(NominalPlan.ChargingStations[j_CS[0]])

    if isplot==True:
        PlotCluster(NominalPlan, NodesGroups, NominalPlan.LoadCapacity)
    
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
            iCS = np.argwhere(NodesGroups[i] == np.array(NominalPlan.ChargingStations))[0][0]
            NominalPlanGroup.ChargingProfile.append(NominalPlan.ChargingProfile[iCS])
            NominalPlanGroup.ChargingRateProfile.append(NominalPlan.ChargingRateProfile[iCS])
    
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
    NominalPlanGroup.NumberOfChargeStations = len(NominalPlanGroup.ChargingStations)
    NominalPlanGroup.LoadCapacity = NominalPlan.LoadCapacity
    NominalPlanGroup.BatteryCapacity = NominalPlan.BatteryCapacity

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
        plt.subplot(4,1,(1,2))
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
        plt.subplot(4,1,3)
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


        plt.subplot(4,1,4)
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


def AddChargingStations(NominalPlan: DataTypes.NominalPlanning, i):
    iCS = NominalPlan.ChargingStations[i]
    N = NominalPlan.N
    NominalPlan.NodesTimeOfTravel = np.block([[NominalPlan.NodesTimeOfTravel, NominalPlan.NodesTimeOfTravel[:,iCS].reshape(N,1)],[NominalPlan.NodesTimeOfTravel[iCS,:].reshape(1,N), 999.0*np.ones((1,1))]])
    NominalPlan.NodesEnergyTravel = np.block([[NominalPlan.NodesEnergyTravel, NominalPlan.NodesEnergyTravel[:,iCS].reshape(N,1)],[NominalPlan.NodesEnergyTravel[iCS,:].reshape(1,N), 999.0*np.ones((1,1))]])
    NominalPlan.NodesEnergyTravelSigma = np.block([[NominalPlan.NodesEnergyTravelSigma, NominalPlan.NodesEnergyTravelSigma[:,iCS].reshape(N,1)],[NominalPlan.NodesEnergyTravelSigma[iCS,:].reshape(1,N), 999.0*np.ones((1,1))]])
    NominalPlan.TravelSigma = np.block([[NominalPlan.TravelSigma, NominalPlan.TravelSigma[:,iCS].reshape(N,1)],[NominalPlan.TravelSigma[iCS,:].reshape(1,N), 999.0*np.ones((1,1))]])
    NominalPlan.TravelSigma2 = NominalPlan.TravelSigma**2
    NominalPlan.NodesEnergyTravelSigma2 = NominalPlan.NodesEnergyTravelSigma**2
    NominalPlan.ChargingStations = np.append(NominalPlan.ChargingStations, N)
    NominalPlan.NodesPosition = np.vstack([NominalPlan.NodesPosition, NominalPlan.NodesPosition[iCS,:].reshape(1,2)])
    NominalPlan.N = N+1
    NominalPlan.NodesRealNames.append(NominalPlan.NodesRealNames[iCS])
    NominalPlan.NumberOfChargeStations += 1
    NominalPlan.ChargingProfile.append(NominalPlan.ChargingProfile[i])
    NominalPlan.ChargingRateProfile.append(NominalPlan.ChargingRateProfile[i])
    NominalPlan.LoadDemand = np.append(NominalPlan.LoadDemand, NominalPlan.LoadDemand[iCS])

    return NominalPlan

def PlotCluster(NominalPlan: DataTypes.NominalPlanning, NodesGroups, LoadCapacity: float = 0.0, TrajSol: list = []):

# Plot the groups
    if np.max(NominalPlan.NodesPosition)>0:
        col_vec = ['m','y','b','r','g','c']
        markers = ['o','s','^','v','<','>','*']
        imarkers = 0
        leg_str = list()
        M = len(NodesGroups)
        N = NominalPlan.N
        OtherNodes = set(range(NominalPlan.NumberOfDepots,N))
        plt.figure()
        ax = plt.subplot(111)
        plt.scatter(NominalPlan.NodesPosition[0:NominalPlan.NumberOfDepots,0], NominalPlan.NodesPosition[0:NominalPlan.NumberOfDepots,1], c='k', s=50)
        OtherNodes = OtherNodes - set(range(NominalPlan.NumberOfDepots))
        leg_str.append('Depot')
        for i in range(M):
            if 0 == i%len(col_vec) and i>0:
                imarkers += 1
            i_toplot = list(set(NodesGroups[i]) - set(NominalPlan.ChargingStations) - set(NominalPlan.CarsInDepots))
            plt.scatter(NominalPlan.NodesPosition[i_toplot,0], NominalPlan.NodesPosition[i_toplot,1], s=50, c=col_vec[i%len(col_vec)], marker=markers[imarkers%len(markers)])
            OtherNodes = OtherNodes - set(i_toplot)
            leg_str.append("Group {:}, Nodes: {}, Left Load: {:}".format(i,len(i_toplot),LoadCapacity-np.sum(NominalPlan.LoadDemand[i_toplot])))
        for i in NominalPlan.ChargingStations:
            if i in NodesGroups[0] and len(NodesGroups)==1:
                plt.scatter(NominalPlan.NodesPosition[i,0], NominalPlan.NodesPosition[i,1], marker='x', c=col_vec[0], s=50)
            else:
                plt.scatter(NominalPlan.NodesPosition[i,0], NominalPlan.NodesPosition[i,1], marker='x', c='k', s=15)
            OtherNodes = OtherNodes - set([i])
        leg_str.append('Charging Station')
        if len(OtherNodes)>0:
            OtherNodes = list(set(range(NominalPlan.NumberOfDepots,N)) - set(NominalPlan.ChargingStations) - set(NodesGroups[0]))
            plt.scatter(NominalPlan.NodesPosition[OtherNodes,0], NominalPlan.NodesPosition[OtherNodes,1], c='c', s=10)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
        plt.legend(leg_str, loc=(1.01,0.05))
        for i in range(N):
            colr = 'r' if i in NominalPlan.CarsInDepots else 'c'
            colr = 'k' if i in NominalPlan.ChargingStations else colr
            plt.text(NominalPlan.NodesPosition[i,0]+1,NominalPlan.NodesPosition[i,1]+1,"{:}({:})".format(i,NominalPlan.LoadDemand[i]), color=colr,fontsize=10)
        if len(TrajSol)>0:
            for m in range(M):
                colr = col_vec[m%len(col_vec)]
                for i in range(len(TrajSol)-1):
                    j1 = TrajSol[i,m]
                    j2 = TrajSol[i+1,m]
                    plt.arrow(NominalPlan.NodesPosition[j1,0],NominalPlan.NodesPosition[j1,1],NominalPlan.NodesPosition[j2,0]-NominalPlan.NodesPosition[j1,0],NominalPlan.NodesPosition[j2,1]-NominalPlan.NodesPosition[j1,1], width= 2.0, color=colr)
        plt.xlim((np.min(NominalPlan.NodesPosition[:,0])-10,np.max(NominalPlan.NodesPosition[:,0])+10))
        plt.ylim((np.min(NominalPlan.NodesPosition[:,1])-10,np.max(NominalPlan.NodesPosition[:,1])+10))
        plt.grid()
        plt.show()


def LoadScenarioFileEvrp(ScenarioFileName: str, NominalPlan: DataTypes.NominalPlanning, PltParams: DataTypes.PlatformParams):
    # Load Scenario File:
    NominalPlan.NumberOfChargeStations = 0
    PltParams.BatteryCapacity = 100.0
    EnergyConsumption = 0.0
    NominalPlan.StationRechargePower = 1.0e5
    f = open(ScenarioFileName, "r")
    for x in f:
        if x[0:9] == 'VEHICLES:':
            NominalPlan.NumberOfCars = int(x[9:])
            NominalPlan.CarsInDepots = np.zeros((NominalPlan.NumberOfCars,), dtype=int).tolist()
        if x[0:14] == 'OPTIMAL_VALUE:':
            OptVal = float(x[14:])
        if x[0:10] == 'DIMENSION:':
            NCustemers = int(x[10:])
        if x[0:9] == 'STATIONS:':
            NominalPlan.NumberOfChargeStations = int(x[10:])
        if x[0:9] == 'CAPACITY:':
            PltParams.LoadCapacity = float(x[10:])        
        if x[0:16] == 'ENERGY_CAPACITY:':
            PltParams.BatteryCapacity = float(x[16:])  
        if x[0:19] == 'ENERGY_CONSUMPTION:':
            EnergyConsumption = float(x[19:])  
        if x[0:18] == 'NODE_COORD_SECTION':
            NominalPlan.N = NCustemers + NominalPlan.NumberOfChargeStations
            NominalPlan.NodesPosition = np.zeros((NominalPlan.N,2))
            for i in range(NominalPlan.N):
                x = f.readline()
                x = x.split()
                NominalPlan.NodesPosition[int(x[0])-1,0] = float(x[1])
                NominalPlan.NodesPosition[int(x[0])-1,1] = float(x[2])
        if x[0:14] == 'DEMAND_SECTION':
            NominalPlan.LoadDemand = np.zeros((NominalPlan.N,))
            for i in range(NCustemers):
                x = f.readline()
                x = x.split()
                NominalPlan.LoadDemand[int(x[0])-1,] = float(x[1])
        if x[0:22] == 'STATIONS_COORD_SECTION':
            NominalPlan.ChargingStations = np.zeros((NominalPlan.NumberOfChargeStations,1), dtype=int).tolist()
            for i in range(NominalPlan.NumberOfChargeStations):
                x = f.readline()
                x = x.split()
                NominalPlan.ChargingStations[i] = int(x[0])-1
                NominalPlan.ChargingProfile.append(np.array([[0.0, 0.0],[PltParams.BatteryCapacity,PltParams.BatteryCapacity/NominalPlan.StationRechargePower]]))
                NominalPlan.ChargingRateProfile.append(np.array([[0.0, NominalPlan.StationRechargePower],[PltParams.BatteryCapacity, 0.0]]))

    NominalPlan.NodesTimeOfTravel = np.zeros((NominalPlan.N,NominalPlan.N))
    NominalPlan.NodesEnergyTravel = np.zeros((NominalPlan.N,NominalPlan.N))
    NominalPlan.NodesEnergyTravelSigma = np.zeros((NominalPlan.N,NominalPlan.N))
    NominalPlan.TravelSigma = np.zeros((NominalPlan.N,NominalPlan.N))
    for i in range(NominalPlan.N):
        for j in range(i,NominalPlan.N):
            if i==j: continue
            NominalPlan.NodesTimeOfTravel[i,j] = np.linalg.norm(NominalPlan.NodesPosition[i,:]-NominalPlan.NodesPosition[j,:])
            NominalPlan.NodesTimeOfTravel[j,i] = NominalPlan.NodesTimeOfTravel[i,j]
            NominalPlan.NodesEnergyTravel[i,j] = -NominalPlan.NodesTimeOfTravel[i,j] * EnergyConsumption
            NominalPlan.NodesEnergyTravel[j,i] = NominalPlan.NodesEnergyTravel[i,j]

    # NominalPlan.LoadDemand = NominalPlan.LoadDemand*0
    NominalPlan.TravelSigma = NominalPlan.NodesTimeOfTravel * 0.1
    NominalPlan.NodesEnergyTravelSigma = -NominalPlan.NodesEnergyTravel * 0.1
    NominalPlan.TravelSigma2 = NominalPlan.TravelSigma**2
    NominalPlan.NodesEnergyTravelSigma2 = NominalPlan.NodesEnergyTravelSigma**2

    NominalPlan.NumberOfDepots = 1
    NominalPlan.NodesVelocity = np.ones((NominalPlan.N,NominalPlan.N))
    NominalPlan.NodesDistance = NominalPlan.NodesTimeOfTravel
    NominalPlan.NodesTaskTime = np.zeros((NominalPlan.N,1))
    NominalPlan.NodesTaskPower = np.zeros((NominalPlan.N,1))
    NominalPlan.NodesPriorities = np.zeros((NominalPlan.N,1))
    NominalPlan.LoadCapacity = PltParams.LoadCapacity
    NominalPlan.BatteryCapacity = PltParams.BatteryCapacity


    return NominalPlan, PltParams, OptVal

def LoadScenarioFileXml(ScenarioFileName: str, NominalPlan: DataTypes.NominalPlanning, PltParams: DataTypes.PlatformParams):
    # Load Scenario File:
    NominalPlan.NumberOfChargeStations = 0
    PltParams.BatteryCapacity = 100.0
    EnergyConsumption = 0.0
    NominalPlan.StationRechargePower = 1.0e5
    NominalPlan.NumberOfDepots = 0
    NominalPlan.NumberOfChargingStations = 0

    # Read the xml file:
    with open(ScenarioFileName) as fd:
        doc = xml.parse(fd.read())
    
    Nodes = doc['instance']['network']['nodes']['node']
    CS = doc['instance']['fleet']['vehicle_profile']['custom']['charging_functions']['function']
    NominalPlan.N = len(Nodes)
    NominalPlan.LoadDemand = np.zeros((NominalPlan.N,))
    NominalPlan.NodesPosition = np.zeros((NominalPlan.N,2))
    for i in range(NominalPlan.N):
        NominalPlan.NodesPosition[int(Nodes[i]['@id']),0] = float(Nodes[i]['cx'])
        NominalPlan.NodesPosition[int(Nodes[i]['@id']),1] = float(Nodes[i]['cy'])
        
        if Nodes[i]['@type'] == '0':
            NominalPlan.NumberOfDepots += 1
        if Nodes[i]['@type'] == '1':
            NominalPlan.LoadDemand[int(Nodes[i]['@id'])] = 1.0
        elif Nodes[i]['@type'] == '2':
            NominalPlan.ChargingStations.append(int(Nodes[i]['@id']))
            NominalPlan.NumberOfChargingStations += 1
            cs_type = doc['instance']['network']['nodes']['node'][i]['custom']['cs_type']
            for j in range(len(CS)):
                if CS[j]['@cs_type'] == cs_type:
                    NominalPlan.ChargingProfile.append(np.zeros((len(CS[j]['breakpoint']),2)))
                    NominalPlan.ChargingRateProfile.append(np.zeros((len(CS[j]['breakpoint']),2)))
                    for k in range(1,len(CS[j]['breakpoint'])):
                        NominalPlan.ChargingProfile[-1][k,0] = float(CS[j]['breakpoint'][k]['battery_level'])
                        NominalPlan.ChargingRateProfile[-1][k,0] = float(CS[j]['breakpoint'][k]['battery_level'])
                        NominalPlan.ChargingProfile[-1][k,1] = float(CS[j]['breakpoint'][k]['charging_time'])
                        NominalPlan.ChargingRateProfile[-1][k-1,1] = (NominalPlan.ChargingProfile[-1][k,0]-NominalPlan.ChargingProfile[-1][k-1,0])/(NominalPlan.ChargingProfile[-1][k,1]-NominalPlan.ChargingProfile[-1][k-1,1])

                    break
        
    PltParams.BatteryCapacity = float(doc['instance']['fleet']['vehicle_profile']['custom']['battery_capacity'])
    PltParams.LoadCapacity = NominalPlan.N
    EnergyConsumption = float(doc['instance']['fleet']['vehicle_profile']['custom']['consumption_rate'])
    Velocity = float(doc['instance']['fleet']['vehicle_profile']['speed_factor'])


    NominalPlan.NodesTimeOfTravel = np.zeros((NominalPlan.N,NominalPlan.N))
    NominalPlan.NodesEnergyTravel = np.zeros((NominalPlan.N,NominalPlan.N))
    NominalPlan.NodesEnergyTravelSigma = np.zeros((NominalPlan.N,NominalPlan.N))
    NominalPlan.TravelSigma = np.zeros((NominalPlan.N,NominalPlan.N))
    for i in range(NominalPlan.N):
        for j in range(i,NominalPlan.N):
            if i==j: continue
            NominalPlan.NodesTimeOfTravel[i,j] = np.linalg.norm(NominalPlan.NodesPosition[i,:]-NominalPlan.NodesPosition[j,:])/Velocity
            NominalPlan.NodesTimeOfTravel[j,i] = NominalPlan.NodesTimeOfTravel[i,j]
            NominalPlan.NodesEnergyTravel[i,j] = -NominalPlan.NodesTimeOfTravel[i,j] * EnergyConsumption * Velocity
            NominalPlan.NodesEnergyTravel[j,i] = NominalPlan.NodesEnergyTravel[i,j]
            NominalPlan.NodesEnergyTravelSigma[i,j] = 0
            NominalPlan.NodesEnergyTravelSigma[j,i] = NominalPlan.NodesEnergyTravelSigma[i,j]
            NominalPlan.TravelSigma[i,j] = 0
            NominalPlan.TravelSigma[j,i] = NominalPlan.TravelSigma[i,j]

    NominalPlan.NumberOfDepots = 1
    NominalPlan.NodesVelocity = np.ones((NominalPlan.N,NominalPlan.N))
    NominalPlan.NodesDistance = NominalPlan.NodesTimeOfTravel
    NominalPlan.NodesTaskTime = np.zeros((NominalPlan.N,1))
    NominalPlan.NodesTaskPower = np.zeros((NominalPlan.N,1))
    NominalPlan.NodesPriorities = np.zeros((NominalPlan.N,1))

    OptVal = 0.0
    return NominalPlan, PltParams, OptVal
