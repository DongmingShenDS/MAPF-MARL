#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <thread>
#include <chrono>
#include <queue>
#include <climits>
#include <cstdlib>
#include <set>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <condition_variable>
#include <mutex>
#ifndef ROAD_H
#define ROAD_H
#include "Agent.h"
#include "Lock.h"
#include "Message.h"
#include "Global.h"
#include "Vertex.h"
#include "warehouse_map.h"

using namespace std;

class Global;
class Road{
    public:
        Road(ifstream &map);
        std::shared_ptr<Global> get_global();
        unordered_map<int, std::shared_ptr<Agent>> get_all_agents_on_road();
        vector<std::shared_ptr<Agent>> get_new_arrived_agents_at_road();
        vector<std::shared_ptr<Agent>> get_agents_leaving_lock_outT();
        unordered_map<int, std::shared_ptr<Agent>> get_agents_at_lock_inT();
        warehouse_map get_warehouse_map();
        void set_global(std::shared_ptr<Global>);
        void add_all_agents_on_road(std::shared_ptr<Agent>);
        void set_new_arrived_agents_at_road(vector<std::shared_ptr<Agent>>);
        void set_agents_leaving_lock_outT(vector<std::shared_ptr<Agent>>);
        void add_agents_at_lock_inT(std::shared_ptr<Agent>);
        unordered_map<int, vector<pair<int, int>>> get_agents_remaining_path();
        void StarSearch(std::shared_ptr<Agent>);
        void run();
        unordered_map<int, MapVertex *> singleStepSolver(Message* msg, vector<std::shared_ptr<Agent>>& at_lock_in);
        Message *wait_message1_from_global();
        Message *wait_message2_from_global();
        void send_msg1_to_road(Message *message);
        void send_msg2_to_road(Message *message);
        void set_moved(unordered_map<int, MapVertex *> &thisStep, vector<pair<int, int>> &occupied, vector<int> &occupied_num, unordered_map<int, bool> &havetowait, int num1, int num2, int num3);
        //Decide which agent should move first when conflicts occur, now use their waiting time to compare.
        int decide_priority_2(vector<int> conflict_agent);
        int decide_priority_3(vector<int> conflict_agent);
        double get_road_load();

    private:
        std::shared_ptr<Global> global;
        int timestep = 0;
        warehouse_map mywarehouse;
        unordered_map<int, std::shared_ptr<Agent>> all_agents_on_road;
        vector<std::shared_ptr<Agent>> new_arrived_agents_at_road;
        vector<std::shared_ptr<Agent>> agents_leaving_lock_outT;
        unordered_map<int, std::shared_ptr<Agent>> agents_at_lock_inT;
        unordered_map<int, vector<pair<int,int>>> agents_remaining_path;
        unordered_map<int, int> waiting_time;
        vector<std::shared_ptr<Agent>> all_agents_at_lock_in;
        //All agents that arrived lock_out at the end of last timestep, receive at the start of this timestep.
        queue<Message*> msg1_from_global_queue;
        //All agents leaving lock_in at this timestep, receive at the end of this timestep
        queue<Message*> msg2_from_global_queue;
        shared_ptr<mutex> m;
        shared_ptr<condition_variable> cv;
        int **weight_score;
        int agent_on_road;
};
#endif