import numpy as np
import SimDataTypes



def SolveRecursive(PltParams, NominalPlan, NodesWorkDone, TimeLeft, PowerLeft, i_CurrentNode, NodesTrajectory, NodesWorkSequence, Cost, BestPlan):
    # Check how many days need to recharge before starting Task
    if PowerLeft < NominalPlan.NodesTaskPower[i_CurrentNode]:
        TimeToRecharge = np.ceil((NominalPlan.NodesTaskPower[i_CurrentNode]-PowerLeft)/PltParams.RechargePowerPerDay)
    else:
        TimeToRecharge = 0
    if TimeLeft > NominalPlan.NodesTaskTime[i_CurrentNode] + TimeToRecharge:
        NodesWorkSequence.append(i_CurrentNode) # Mark That Current Node Visited
        NodesTrajectory.append(i_CurrentNode)
        if NodesWorkDone[i_CurrentNode]==0:
            NodesWorkDone[i_CurrentNode] = 1
            Cost += NominalPlan.PriorityCoefInCost*NominalPlan.NodesPriorities[i_CurrentNode]**2 - NominalPlan.TimeCoefInCost*NominalPlan.NodesTaskTime[i_CurrentNode]
            TimeLeft -= (NominalPlan.NodesTaskTime[i_CurrentNode] + TimeToRecharge)
            PowerLeft -= NominalPlan.NodesTaskPower[i_CurrentNode]


        # Check if all nodes are visited:
        if np.sum(NodesWorkDone) == NominalPlan.NodesPriorities.shape[0]:
            if i_CurrentNode == 0 or NominalPlan.ReturnToBase==False:
                return BestPlan, NodesTrajectory, NodesWorkDone, Cost
            if NominalPlan.ReturnToBase==True:
                PwrTravelNode_0 = NominalPlan.NodesTimeOfTravel[i_CurrentNode,0]*PltParams.VelConstPowerConsumption + PltParams.VelEnergyConsumptionCoef*NominalPlan.NodesVelocity[i_CurrentNode,0]**2
                if PowerLeft < PwrTravelNode_0:
                    DaysToRecharge = np.ceil((PwrTravelNode_0-PowerLeft)/PltParams.RechargePowerPerDay)
                else:
                    DaysToRecharge = 0
                if NominalPlan.NodesTimeOfTravel[i_CurrentNode,0]+DaysToRecharge<=TimeLeft:
                    BestPlan, Cur_NodesTrajectory, NodesWorkDone, Cur_Cost = SolveRecursive(PltParams=PltParams,
                                                                            NominalPlan= NominalPlan, 
                                                                            NodesWorkDone= NodesWorkDone.copy(), 
                                                                            TimeLeft= TimeLeft - NominalPlan.NodesTimeOfTravel[i_CurrentNode,0] - DaysToRecharge, 
                                                                            PowerLeft= PowerLeft - PwrTravelNode_0 + DaysToRecharge*PltParams.RechargePowerPerDay, 
                                                                            i_CurrentNode= 0, 
                                                                            NodesTrajectory= NodesTrajectory.copy(), 
                                                                            NodesWorkSequence= NodesWorkSequence.copy(),
                                                                            Cost= Cost - NominalPlan.TimeCoefInCost*(DaysToRecharge+NominalPlan.NodesTimeOfTravel[i_CurrentNode,0]),
                                                                            BestPlan=BestPlan)
                return BestPlan, Cur_NodesTrajectory, NodesWorkDone, Cur_Cost
        else:
            for j in range(NominalPlan.NodesPriorities.shape[0]):
                if NodesWorkDone[j] == 1:
                    continue
                else:
                    PwrTravelNode_j = NominalPlan.NodesTimeOfTravel[i_CurrentNode,j]*PltParams.VelConstPowerConsumption + PltParams.VelEnergyConsumptionCoef*NominalPlan.NodesVelocity[i_CurrentNode,j]**2
                    if PowerLeft < PwrTravelNode_j:
                        DaysToRecharge = np.ceil((PwrTravelNode_j-PowerLeft)/PltParams.RechargePowerPerDay)
                    else:
                        DaysToRecharge = 0
                    if NominalPlan.NodesTimeOfTravel[i_CurrentNode,j]+DaysToRecharge>TimeLeft:
                        continue
                    BestPlan, Cur_NodesTrajectory, Cur_NodesWorkDone, Cur_Cost = SolveRecursive(PltParams=PltParams,
                                                                            NominalPlan= NominalPlan, 
                                                                            NodesWorkDone= NodesWorkDone.copy(), 
                                                                            TimeLeft= TimeLeft - NominalPlan.NodesTimeOfTravel[i_CurrentNode,j] - DaysToRecharge, 
                                                                            PowerLeft= PowerLeft - PwrTravelNode_j + DaysToRecharge*PltParams.RechargePowerPerDay, 
                                                                            i_CurrentNode= j, 
                                                                            NodesTrajectory= NodesTrajectory.copy(), 
                                                                            NodesWorkSequence= NodesWorkSequence.copy(),
                                                                            Cost= Cost - NominalPlan.TimeCoefInCost*(DaysToRecharge+NominalPlan.NodesTimeOfTravel[i_CurrentNode,j]),
                                                                            BestPlan=BestPlan)
                    if ((np.sum(Cur_NodesWorkDone) == NominalPlan.NodesPriorities.shape[0] and NominalPlan.MustVisitAllNodes==True) or (NominalPlan.MustVisitAllNodes==False)) and (Cur_Cost >= BestPlan.BestCost):
                        BestPlan.BestCost = Cur_Cost
                        BestPlan.BestNodesTrajectory = Cur_NodesTrajectory
                    
            return BestPlan, BestPlan.BestNodesTrajectory, NodesWorkDone, BestPlan.BestCost
        
    ## No TimeLeft to complete the current task
    return BestPlan, BestPlan.BestNodesTrajectory, NodesWorkDone, Cost