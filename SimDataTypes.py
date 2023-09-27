import numpy as np

class PlatformParams():
    def __init__(self):
        self.Vmax = 0
        self.MinVelReductionCoef = 0
        self.MaxVelReductionCoef = 0
        self.VelEnergyConsumptionCoef = 0
        self.VelConstPowerConsumption = 0
        self.MinPowerConsumptionPerTask = 0
        self.MaxPowerConsumptionPerTask = 0
        self.MinTimePerTask = 0
        self.MaxTimePerTask = 0
        self.RechargePowerPerDay = 0
        self.BatteryCapacity = 0
        self.TimeCoefInCost = 0


## Nomianl Planning Class
class NominalPlanning():
    def __init__(self,N):
        self.N = 0
        self.NodesVelocity = np.zeros((N,N))
        self.NodesDistance = np.zeros((N,N))
        self.NodesTimeOfTravel = np.zeros((N,N))
        self.NodesEnergyTravel = np.zeros((N,N))
        self.NodesEnergyTravelSigma = np.zeros((N,N))
        self.TravelSigma = np.zeros((N,N))
        self.NodesTaskTime = np.zeros((N,1))
        self.NodesTaskPower = np.zeros((N,1))
        self.NodesPriorities = np.zeros((N,1))

class BestPlan():
    def __init__(self,N):
        self.Cost = -10e10
        self.NodesTrajectory = np.zeros((N,1), dtype=int)
        self.PowerLeft = np.zeros((N,1))
        self.ChargingSequence = np.zeros((1,3)) # [Node, Time Spent, Power at the end of the charging]