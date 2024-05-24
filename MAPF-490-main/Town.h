#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <queue>
#include <condition_variable>
#include <mutex>
#include <cmath>
#ifndef TOWN_H
#define TOWN_H
#include "Agent.h"
#include "Lock.h"
#include "Message.h"
#include "Global.h"

using namespace std;
class Global;
class Town{
    public:
        Town(int townId, std::shared_ptr<Global> global);
        std::shared_ptr<Global> get_global();
        int getTownId();
        void setAgentList(unordered_map<int, std::shared_ptr<Agent>> agentList);
        void send_message_to_town(Message* message);
        Message* wait_message_from_global();
        void start_the_town();
        void wait_start_signal_from_global();
        void read_path(string fileName);
        void generate_scene_file(string scen_file_name,vector<std::shared_ptr<Agent>> new_agents,vector<std::shared_ptr<Agent>> agents_from_lock_in, bool is_init, unordered_set<int> deleted_From_Run);
        void EECBS(string map_file_name, string scen_file_name);
        void run();
        bool isValid(int row, int col);
        
    private:
        //data variables
        int startX;
        int startY;
        int townId;
        int row_dimension;
        int col_dimension;
        bool start_signal;
        int timestep = 0;
        shared_ptr<Global> global;
        shared_ptr<mutex> m;
        shared_ptr<condition_variable> cv;
        vector<string> town_map;
        unordered_map<int,std::shared_ptr<Agent>> agent_map;
        unordered_map<int,int> agent_in_grid;
        unordered_map<int,int> lineId_to_agentId;
        unordered_set<int> locks;
        unordered_set<int> agents_in_lock_in;
        unordered_map<int,pair<int,int>> agents_to_lock_out;
        queue<Message*> messages_from_global;
        vector<vector<int>> directions = {{1,0},{0,1},{-1,0},{0,-1},{0,0}};

        //private functions
        void initMap();
        std::shared_ptr<Agent> getAgentByLineNumber(int line_count);
        bool checkAgentExistByLineNumber(int line_count);
        pair<int,int> getLocalCoordFromGlobal(pair<int,int> grid);
        pair<int,int> getGlobalCoordFromLocal(pair<int,int> grid);
        int convertLocToId(pair<int,int> grid);
        void removeAgentFromGridMap(int locId);
        void setTempGoal();
        void sendMessages(unordered_map<int, bool>& content,vector<std::shared_ptr<Agent>>& agents_to_lock_out_message,vector<std::shared_ptr<Agent>>& agent_list);
        void sendLockInMessage(unordered_map<int, bool>& content);
        void sendLockOutMessage(vector<std::shared_ptr<Agent>>& agents_to_lock_out_message);
        void sendNewFreeAgentMessage(vector<std::shared_ptr<Agent>>& agent_list);
        pair<int,int> findAcessCell(pair<int,int> grid,set<pair<int,int>>& existing_goal);
        pair<int,int> findLockAcessCell(pair<int,int> grid,set<pair<int,int>>& existing_goal);
        bool checkIfCanPlaceAgent(int row, int col);

};      

#endif