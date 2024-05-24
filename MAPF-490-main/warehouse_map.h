//
//  warehouse_map.h
//  PathFinder_WareHouseMap
//
//  Created by Tiancheng Xu on 2021/10/17.
//

#ifndef warehouse_map_h
#define warehouse_map_h
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <unordered_map>
#include "Vertex.h"


using namespace std;

class warehouse_map{
public:
    warehouse_map();
    ~warehouse_map();
    void readFile(ifstream& inputfile);
    int getnumofRows();
    int getnumofCols();
    vector<vector<MapVertex*>> getmap();
    bool isinvalid();
    void setvalid(bool b);
protected:
    vector<vector<MapVertex*>> mymap;
    unordered_map<string, vector<MapVertex *>> adjacent_list;
    int numrows;
    int numcols;
};


#endif /* warehouse_map_h */
