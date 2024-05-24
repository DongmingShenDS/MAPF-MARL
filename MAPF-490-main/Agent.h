#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#ifndef AGENT_H
#define AGENT_H
#include "Lock.h"
using namespace std;

class Agent{
    public:
        Agent(int agent_id, int town_id, pair<int, int> current_grid);
        int get_agent_id();
        int get_town_id();
        pair<int, int> get_goal();
        pair<int, int> get_temp_goal();
        pair<int, int> get_current_grid(); 
        Lock* get_lock_inT(); 
        Lock* get_lock_outT(); 
        int get_goal_town_id();
        vector<pair<int, int>> get_past_path();  
        void set_agent_id(int);
        void set_town_id(int);
        void set_goal(pair<int, int>);
        void set_temp_goal(pair<int, int>);
        void set_goal_town_id(int);
        void set_current_grid(pair<int, int>);
        void set_lock_inT(Lock*);
        void set_lock_outT(Lock*);
        void set_past_path(vector<pair<int, int>>);
        void add_past_path(pair<int, int> p);
        void to_string();

    private:
        int agent_id;
        int town_id;    // which town agent currently in, if on road then -1
        pair<int, int> goal;
        pair<int,int> temp_goal;
        int goal_town_id;
        pair<int, int> current_grid;
        Lock* lock_inT;
        Lock* lock_outT;
        vector<pair<int,int>> past_path;    
};
#endif