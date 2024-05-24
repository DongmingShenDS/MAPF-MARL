#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#ifndef GLOBAL_H
#define GLOBAL_H
#include "Agent.h"
#include "Lock.h"
#include "Road.h"
#include "Message.h"
#include "Town.h"


using namespace std;

class Town;
class Road;
class Global{
    public:
        Global(); 
        void run();
        void lock(Lock* l);
        void unlock(Lock* l);
        void send_alltowns_message(int);    // int for message type sent from Global to Towns
        void wait_one_town_message();
        void wait_one_road_message();
        void wait_message_from_all_towns();
        void wait_message_from_all_roads();
        void wait_ready_message_from_road();
        void wait_one_ready_message();
        void town_to_global(int, Message*); 
        void road_to_global(Message*);
        // getter and setters
        void set_tasks(unordered_map<int, deque<pair<int, int>>>);
        void set_road(std::shared_ptr<Road>);
        void set_all_towns(unordered_map<int, std::shared_ptr<Town>>);
        void set_all_agents(unordered_map<int, std::shared_ptr<Agent>>);
        void set_all_locks(unordered_map<int, Lock*>);
        void set_town_locks(unordered_map<int, vector<Lock*>>);
        unordered_map<int, int> get_gridid_to_townid();
        void set_gridid_to_townid(unordered_map<int, int>);
        void add_gridid_to_townid(int, int);  // 1=grid id, 2=town id
        void set_town_grids(unordered_map<int, vector<int>>);
        void set_lock_reachability(unordered_map<int, unordered_set<int>>);
        // other
        void print_agent_path(unordered_set<int>);
        void print_agent_path_single();
        int get_col_dim();
        void vis_to_json();
        unordered_map<int, vector<Lock*>> town_locks;
    private:
        int col_dim;
        unordered_map<int, deque<pair<int, int>>> tasks;    // tasks queue for each agent_id
        std::shared_ptr<Road> road;
        unordered_map<int, std::shared_ptr<Town>> all_towns;            // town_id to town, for all towns
        unordered_map<int, std::shared_ptr<Agent>> all_agents;          // agent_id to agent, for all agents
        unordered_map<int, Lock*> all_locks;            // lock_id to lock, for all locks
        // unordered_map<int, vector<Lock*>> town_locks;   // all locks at town_id  
        unordered_map<int, int> gridid_to_townid;       // all grids (not including locks) at town_id  
        unordered_map<int, vector<int>> town_grids;     // all grids (not including locks) at town_id
        unordered_map<int, unordered_set<int>> lock_reachability;    // lock_id to all reachable lock_id

        unordered_map<int, std::shared_ptr<Agent>> new_agents;          // agent with new goals
        unordered_map<int, std::shared_ptr<Agent>> new_accept_agents;   // newly accept agents (with new locks assigned)
        unordered_map<int, std::shared_ptr<Agent>> new_acc_agents;  // all new agents, including without locks
        unordered_map<int, std::shared_ptr<Agent>> lock_inT_agents;     // all agents currently still in lock_inT
        unordered_map<int, std::shared_ptr<Agent>> at_lock_outT_agents;     // all agents newly arrived lock_outT (from T)

        queue<pair<int, Message*>> msg1_from_town;      // agents newly arrived in lock_outT [1]
        queue<pair<int, Message*>> msg2_from_town;      // agents at lock_inT's status [2]
        queue<pair<int, Message*>> msg3_from_town;      // all new free agents [1]
        queue<pair<int, Message*>> temp_msg_from_town;  // temp buffer to store message 
        Message* msg1_from_road;                        // agents out of lock_outT [1]
        Message* msg2_from_road;                        // agents currently at lock_inT [1]
        queue<Message*> temp_msg_from_road;             // temp buffer to store message 
        queue<Message*> temp_road_ready_queue;          // temp buffer to store message 

        unordered_map<int, Message*> msg1_to_town;      // agents with new goal & lock [3]
        unordered_map<int, Message*> msg2_to_town;      // agents currently at lock_inT [1]
        Message* msg1_to_road;                          // all agents newly arrived at lock_outT [1]
        Message* msg2_to_road;                          // agents at lock_inT's status [2]
        bool ready_msg = false;

        shared_ptr<mutex> m;                            // for message passing
        shared_ptr<condition_variable> cv;              // for message passing
        int timestep;                                   

};

#endif