#include<iostream>
#include<cstdlib>
#include <ctime>
#include <string.h>
#include "EVRP.hpp"
#include <fstream>
#include <math.h>
#include "/usr/include/eigen3/Eigen/Dense"
#include <boost/math/distributions/normal.hpp>
#include <vector>
#include <set>
#include <algorithm>
#include <numeric>

double inverseCDF(double p) {
    boost::math::normal_distribution<double> distribution(0.0, 1.0); // Normal distribution with mean 0 and standard deviation 1
    return boost::math::quantile(distribution, p);
}

#define MAX_PATH_LENGTH 256
#define MAX_FILES 100

using namespace std;

// char ScenarioFileNames[MAX_FILES][MAX_PATH_LENGTH];
int numFiles = 0;
void DivideNodesIntoGroups(ScenarioType* Scenario, int** NodesGroups, int* NumberOfNodesInGroups);
double ArraySum(double* Array, int* NodesGroup, int NodesGroupSize)
{
    double sum = 0;
    for (int i = 0; i < NodesGroupSize; i++)
    {
        sum += Array[NodesGroup[i]];
    }
    return sum;
}

ScenarioType Read_EVRP_Instance()
{
    // Create Scenario
    ScenarioType Scenario;
    char filename[] = "./../VRP_Instances/evrp-benchmark-set/E-n22-k4.evrp";
    // Open file
    ifstream myfile(filename);
    char delimiters[] = " :=\n\t\r\f\v";
    int temp_i, n_customers = 0, n_CS = 0;
    double temp_d,energy_consumption;

    char line[CHAR_LEN];
    while ((myfile.getline(line, CHAR_LEN - 1)))
    {
        char* token;

        if (!(token = strtok(line, delimiters)))
            continue;

        if (!strcmp(token, "DIMENSION")) {
            if (!sscanf(strtok(NULL, delimiters), "%d", &n_customers)) {
                cout << "DIMENSION error" << endl;
                exit(0);
            }
        } else if (!strcmp(token, "EDGE_WEIGHT_TYPE")) {
            char *tempChar;
            if (!(tempChar = strtok(NULL, delimiters))) {
                cout << "EDGE_WEIGHT_TYPE error" << endl;
                exit(0);
            }
            if (strcmp(tempChar, "EUC_2D")) {
                cout << "not EUC_2D" << endl;
                exit(0);
            }
        } else if (!strcmp(token, "CAPACITY")) {
            if (!sscanf(strtok(NULL, delimiters), "%lf", &Scenario.LoadCapacity)) {
                cout << "CAPACITY error" << endl;
                exit(0);
            }
        } else if (!strcmp(token, "ENERGY_CAPACITY")) {
            if (!sscanf(strtok(NULL, delimiters), "%lf", &Scenario.BatteryCapacity)) {
                cout << "ENERGY_CAPACITY error" << endl;
                exit(0);
            }
        } else if (!strcmp(token, "VEHICLES")) {
            if (!sscanf(strtok(NULL, delimiters), "%d", &Scenario.NumberOfCars)) {
                cout << "VEHICLES error" << endl;
                exit(0);
            }
        } else if (!strcmp(token, "ENERGY_CONSUMPTION")) {
            if (!sscanf(strtok(NULL, delimiters), "%lf", &energy_consumption)) {
                cout << "ENERGY_CONSUMPTION error" << endl;
                exit(0);
            }
        } else if (!strcmp(token, "STATIONS")) {
            if (!sscanf(strtok(NULL, delimiters), "%d", &n_CS)) {
                cout << "STATIONS error" << endl;
                exit(0);
            }
        } else if (!strcmp(token, "OPTIMAL_VALUE")) {
            if (!sscanf(strtok(NULL, delimiters), "%lf", &Scenario.OptimalValue)) {
                cout << "OPTIMAL_VALUE error" << endl;
                exit(0);
            }
        } else if (!strcmp(token, "NODE_COORD_SECTION")) {
            if (n_customers != 0) {
                /*prroblem_size is the number of customers plus the depot*/
                Scenario.N = n_customers + n_CS;

                Scenario.NodesPosition = new double*[Scenario.N];
                Scenario.NodesEnergyTravel = new double*[Scenario.N];
                Scenario.NodesEnergyTravelSigma = new double*[Scenario.N];
                Scenario.NodesEnergyTravelSigma2 = new double*[Scenario.N];
                Scenario.TravelSigma = new double*[Scenario.N];
                Scenario.TravelSigma2 = new double*[Scenario.N];
                Scenario.NodesTimeOfTravel = new double*[Scenario.N];
                Scenario.NodesRealNames = new int[Scenario.N];
                Scenario.LoadDemand = new double[Scenario.N];

                for(int i = 0; i < Scenario.N; ++i) {
                    Scenario.NodesPosition[i] = new double[2];
                    Scenario.NodesEnergyTravel[i] = new double[Scenario.N];
                    Scenario.NodesEnergyTravelSigma[i] = new double[Scenario.N];
                    Scenario.NodesEnergyTravelSigma2[i] = new double[Scenario.N];
                    Scenario.TravelSigma[i] = new double[Scenario.N];
                    Scenario.TravelSigma2[i] = new double[Scenario.N];
                    Scenario.NodesTimeOfTravel[i] = new double[Scenario.N];
                }
                
                for (int i = 0; i < Scenario.N; i++) {
                    //store initial objects
                    myfile >> temp_i;
                    myfile >> Scenario.NodesPosition[i][0] >> Scenario.NodesPosition[i][1];
                }
            } else {
                cout << "wrong problem instance file" << endl;
                exit(1);
            }
        } else if (!strcmp(token, "DEMAND_SECTION")) {
            if (n_customers != 0) {

                int temp;
                //masked_demand = new int[problem_size];
                for (int i = 0; i < Scenario.N; i++) {
                    myfile >> temp;
                    myfile >> Scenario.LoadDemand[temp - 1];
                }
            }
        } else if (!strcmp(token, "DEPOT_SECTION")) {
            myfile >> temp_i;
        }

    }
    myfile.close();
    if (n_customers == 0) {
        cout << "wrong problem instance file" << endl;
        exit(1);
    } else {
        for (int i = 0; i < Scenario.N; i++) 
            for (int j = 0; j < Scenario.N; j++) {
                Scenario.NodesTimeOfTravel[i][j] = sqrt(pow(Scenario.NodesPosition[i][0] - Scenario.NodesPosition[j][0], 2) + pow(Scenario.NodesPosition[i][1] - Scenario.NodesPosition[j][1], 2));
                Scenario.NodesEnergyTravel[i][j] = Scenario.NodesTimeOfTravel[i][j] * energy_consumption;
                Scenario.NodesEnergyTravelSigma[i][j] = 0*Scenario.NodesEnergyTravel[i][j] * 0.1;
                Scenario.NodesEnergyTravelSigma2[i][j] = Scenario.NodesEnergyTravelSigma[i][j]*Scenario.NodesEnergyTravelSigma[i][j];
                Scenario.TravelSigma[i][j] = 0*Scenario.NodesTimeOfTravel[i][j] * 0.1;
                Scenario.TravelSigma2[i][j] = Scenario.TravelSigma[i][j] * Scenario.TravelSigma[i][j];
        }
    }
    Scenario.NumberOfDepots = 1;
    Scenario.ChargingStations = new int[n_CS];
    Scenario.NumberOfChargingStations = n_CS;
    for (int i = 0; i < n_CS; i++) {
        Scenario.ChargingStations[i] = i + n_customers;
    }
    Scenario.CarsInDepots = new int[Scenario.NumberOfCars];
    for (int i = 0; i < Scenario.NumberOfCars; i++) 
        Scenario.CarsInDepots[i] = 0;

    Scenario.PhiTime = inverseCDF(0.9);
    Scenario.PhiEnergy = inverseCDF(0.999);

        /* code */
    // Close file    
    myfile.close();
    return Scenario;
}


int main()
{
    srand(20);
    // Add file paths to the array
    // strncpy(ScenarioFileNames[numFiles], "./VRP_Instances/evrp-benchmark-set/E-n22-k4.evrp", MAX_PATH_LENGTH - 1); ScenarioFileNames[numFiles][MAX_PATH_LENGTH - 1] = '\0';numFiles++; // Ensure null-terminated string
    // strncpy(ScenarioFileNames[numFiles], "./VRP_Instances/evrp-benchmark-set/E-n23-k3.evrp", MAX_PATH_LENGTH - 1); ScenarioFileNames[numFiles][MAX_PATH_LENGTH - 1] = '\0';numFiles++; // Ensure null-terminated string
    // strncpy(ScenarioFileNames[numFiles], "./VRP_Instances/evrp-benchmark-set/E-n30-k3.evrp", MAX_PATH_LENGTH - 1); ScenarioFileNames[numFiles][MAX_PATH_LENGTH - 1] = '\0';numFiles++; // Ensure null-terminated string
    // strncpy(ScenarioFileNames[numFiles], "./VRP_Instances/evrp-benchmark-set/E-n33-k4.evrp", MAX_PATH_LENGTH - 1); ScenarioFileNames[numFiles][MAX_PATH_LENGTH - 1] = '\0';numFiles++; // Ensure null-terminated string
    // strncpy(ScenarioFileNames[numFiles], "./VRP_Instances/evrp-benchmark-set/E-n51-k5.evrp", MAX_PATH_LENGTH - 1); ScenarioFileNames[numFiles][MAX_PATH_LENGTH - 1] = '\0';numFiles++; // Ensure null-terminated string
    // strncpy(ScenarioFileNames[numFiles], "./VRP_Instances/evrp-benchmark-set/E-n76-k7.evrp", MAX_PATH_LENGTH - 1); ScenarioFileNames[numFiles][MAX_PATH_LENGTH - 1] = '\0';numFiles++; // Ensure null-terminated string
    // strncpy(ScenarioFileNames[numFiles], "./VRP_Instances/evrp-benchmark-set/E-n101-k8.evrp", MAX_PATH_LENGTH - 1); ScenarioFileNames[numFiles][MAX_PATH_LENGTH - 1] = '\0';numFiles++; // Ensure null-terminated string

    char ScenarioFileNames[] = "./VRP_Instances/evrp-benchmark-set/E-n22-k4.evrp";
    numFiles = 1;
    // strncpy(ScenarioFileNames, "./VRP_Instances/evrp-benchmark-set/E-n22-k4.evrp", MAX_PATH_LENGTH - 1); ScenarioFileNames[numFiles][MAX_PATH_LENGTH - 1] = '\0';numFiles++; // Ensure null-terminated string
    for (int i = 0; i < numFiles; i++)
    {
        ScenarioType Scenario;
        // Call Read_EVRP_Instance
        Scenario = Read_EVRP_Instance();

        // Cluster the nodes:
        int **NodesGroups = new int*[Scenario.NumberOfCars];
        for(int i = 0; i < Scenario.NumberOfCars; ++i) 
            NodesGroups[i] = new int[Scenario.N];
        int *NumberOfNodesInGroups = new int[Scenario.NumberOfCars];
        for(int i = 0; i < Scenario.NumberOfCars; ++i) 
            NumberOfNodesInGroups[i] = 0;
        
        DivideNodesIntoGroups(&Scenario, NodesGroups, NumberOfNodesInGroups);



        // Free memory
        for(int i = 0; i < Scenario.N; ++i) {
            delete[] Scenario.NodesPosition[i];
            delete[] Scenario.NodesEnergyTravel[i];
            delete[] Scenario.NodesEnergyTravelSigma[i];
            delete[] Scenario.NodesEnergyTravelSigma2[i];
            delete[] Scenario.TravelSigma[i];
            delete[] Scenario.TravelSigma2[i];
            delete[] Scenario.NodesTimeOfTravel[i];
        }    
    
    }


    return 0;
}

double CalcTotalEntropy(ScenarioType* Scenario, double* Entropy_i)
{
    double TotalEntropy = 0;
    for (int i = 0; i < Scenario->NumberOfCars; i++)
    {
        TotalEntropy += Entropy_i[i];
    }
    return TotalEntropy;
}


void CalcEntropy(ScenarioType* Scenario, int** NodesGroups, int* NumberOfNodesInGroups, double** CostMatrix, bool* GroupsChanged, double* Entropy_i)
{
    for (int i = 0; i < Scenario->NumberOfCars; i++)
    {
        if (GroupsChanged[i])
        {
            if (NumberOfNodesInGroups[i] == 1)
            {
                Entropy_i[i] = 0;
                continue;
            }
            // Calc Matrix Eigenvalues:
            Eigen::MatrixXd m(NumberOfNodesInGroups[i], NumberOfNodesInGroups[i]);
            for (int j = 0; j < NumberOfNodesInGroups[i]; j++)
                for (int k = 0; k < NumberOfNodesInGroups[i]; k++)
                    m(j,k) = CostMatrix[NodesGroups[i][j]][NodesGroups[i][k]];

            Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(m);
            Eigen::VectorXd eigenvalues = eigenSolver.eigenvalues().real(); // Get the real part of eigenvalues

            // Calc Entropy:
            Entropy_i[i] = 0;
            for (int j = 0; j < NumberOfNodesInGroups[i]; j++)
            {
                Entropy_i[i] += abs(eigenvalues(j));
            }

        }
    }
    return;

}

void DivideNodesIntoGroups(ScenarioType* Scenario, int** NodesGroups, int* NumberOfNodesInGroups)
{
    set<int> CustomersNodes;
    for (int i = Scenario->NumberOfDepots; i < Scenario->N; i++)
    {
        CustomersNodes.insert(i);
    }
    for (int i=0 ; i < Scenario->NumberOfChargingStations ; i++) {
        CustomersNodes.erase(Scenario->ChargingStations[i]);
    }
    bool* GroupsChanged = new bool[Scenario->NumberOfCars];
    for (int i = 0; i < Scenario->NumberOfCars; i++)
    {
        GroupsChanged[i] = true;
    }
    double* Entropy_i = new double[Scenario->NumberOfCars];
    int** NodesGroupsTemp = new int*[Scenario->NumberOfCars];
    for(int i = 0; i < Scenario->NumberOfCars; ++i) {
        NodesGroupsTemp[i] = new int[Scenario->N];
        for (int j = 0; j < Scenario->N; j++)
            NodesGroupsTemp[i][j] = -1;
    }
    int* NumberOfNodesInGroupsTemp = new int[Scenario->NumberOfCars];

    // Calc CostMatrix
    double** CostMatrix = new double*[Scenario->N];
    for(int i = 0; i < Scenario->N; ++i) {
        CostMatrix[i] = new double[Scenario->N];
        for (int j = 0; j < Scenario->N; j++)
        {
            CostMatrix[i][j] = Scenario->NodesTimeOfTravel[i][j] + Scenario->TravelSigma[i][j] * Scenario->PhiTime;
        }
    }

    //Initiate NodesGroups and NumberOfNodesInGroups
    for (int i = 0; i < Scenario->NumberOfCars; i++)
    {
        NodesGroups[i][0] = Scenario->CarsInDepots[i];
        for (int j = 1; j < Scenario->N; j++)
        {
            NodesGroups[i][j] = -1;
        }
        NumberOfNodesInGroups[i] = 1;
    }

    while (!CustomersNodes.empty()) {
            int max_Load_Node = 0;
            double maxTime = 0.0;
            int i = 0;

            // find the node left with the largest demand:
            double maxDemand = 0.0;
            for (int i=0 ; i<CustomersNodes.size() ; i++) {
                int node = *next(CustomersNodes.begin(), i);
                if (Scenario->LoadDemand[node] >= maxDemand) {
                    maxDemand = Scenario->LoadDemand[node];
                    max_Load_Node = node;
                }
            }
            double* Entropy_i = new double[Scenario->NumberOfCars];
            double* TotalEntropy_i = new double[Scenario->NumberOfCars];
            memcpy(NumberOfNodesInGroupsTemp, NumberOfNodesInGroups, Scenario->NumberOfCars*sizeof(int));
            for (int i = 0; i < Scenario->NumberOfCars; i++)
                memcpy(NodesGroupsTemp[i], NodesGroups[i], Scenario->N*sizeof(int));
            for (int i = 0; i < Scenario->NumberOfCars; i++)
            {
                NodesGroupsTemp[i][NumberOfNodesInGroups[i]] = max_Load_Node;
                memcpy(NumberOfNodesInGroupsTemp, NumberOfNodesInGroups, Scenario->NumberOfCars*sizeof(int));
                NumberOfNodesInGroupsTemp[i]++;
                CalcEntropy(Scenario, NodesGroupsTemp, NumberOfNodesInGroupsTemp, CostMatrix, GroupsChanged, Entropy_i);
                TotalEntropy_i[i] = CalcTotalEntropy(Scenario, Entropy_i);
                memcpy(NodesGroupsTemp[i], NodesGroups[i], Scenario->N*sizeof(int));
            }
            // Find min Entropy:
            for (int i = 0; i < Scenario->NumberOfCars; i++)
            {
                double minEntropy = 100000;
                int minEntropy_i = 0;
                for (int i = 0; i < Scenario->NumberOfCars; i++)
                {
                    if (TotalEntropy_i[i] < minEntropy)
                    {
                        minEntropy = TotalEntropy_i[i];
                        minEntropy_i = i;
                    }
                }
                // check if node can be added due to demand:
                if (Scenario->LoadDemand[max_Load_Node] + ArraySum(Scenario->LoadDemand, NodesGroups[minEntropy_i], NumberOfNodesInGroups[minEntropy_i]) <= Scenario->LoadCapacity)
                {
                    // Add node to the group with the min Entropy:
                    NodesGroups[minEntropy_i][NumberOfNodesInGroups[minEntropy_i]] = max_Load_Node;
                    NumberOfNodesInGroups[minEntropy_i]++;
                    CustomersNodes.erase(max_Load_Node);
                    break;
                }
                else
                {
                    TotalEntropy_i[minEntropy_i] = 100000;
                }
            }
    }
 



    // Calc Entropy

    CalcEntropy(Scenario, NodesGroups, NumberOfNodesInGroups, CostMatrix, GroupsChanged, Entropy_i);
    double TotalEntropy = CalcTotalEntropy(Scenario, Entropy_i);

    cout << "Total Initial Entropy: " << TotalEntropy << " Number Of Customers" << Scenario->N - Scenario->NumberOfChargingStations - Scenario->NumberOfDepots << endl;

    double Entropy_Prev = TotalEntropy;
    for (int iter = 0 ; iter<100 ; iter++)
    {
        bool* GroupsChanged_iter = new bool[Scenario->NumberOfCars];
        for (int i = 0; i < Scenario->NumberOfCars; i++)
            GroupsChanged_iter[i] = false;
        
        if (Entropy_Prev == 0)
            break;
        
        CalcEntropy(Scenario, NodesGroups, NumberOfNodesInGroups, CostMatrix, GroupsChanged, Entropy_i);
        TotalEntropy = CalcTotalEntropy(Scenario, Entropy_i);
        memcpy(NumberOfNodesInGroupsTemp, NumberOfNodesInGroups, Scenario->NumberOfCars*sizeof(int));
        for (int i = 0; i < Scenario->NumberOfCars; i++)
            memcpy(NodesGroupsTemp[i], NodesGroups[i], Scenario->N*sizeof(int));

        for (int i = 0; i < Scenario->NumberOfCars; i++)
        {
            if (NumberOfNodesInGroups[i] == 1)
                continue;
            for (int j = 1; j < NumberOfNodesInGroups[i]; j++)
                int node = NodesGroups[i][j];
        }

    }


    // free memory
    for(int i = 0; i < Scenario->N; ++i) {
        delete[] CostMatrix[i];
    }
    delete[] CostMatrix;
    delete[] GroupsChanged;
    return;
}