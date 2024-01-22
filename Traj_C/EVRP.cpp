#include <iostream>
#include <cstdlib>
#include <math.h>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include "EVRP.hpp"


using namespace std;


// Read File and store data in EVRP_Instance
ScenarioType Read_EVRP_Instance()
{
    // Create Scenario
    ScenarioType Scenario;
    char filename[] = "./VRP_Instances/evrp-benchmark-set/E-n22-k4.evrp";
    // Open file
    ifstream myfile;
    myfile.open(filename);
    char delimiters[] = ":\n";

    char line[CHAR_LEN];
    while (myfile.getline(line, CHAR_LEN-1))
    {
        char* token = strtok(line, delimiters);
        /* code */
    }
    
    myfile.close();
    return Scenario;
}