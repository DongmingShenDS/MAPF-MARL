#include <iostream>
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <fstream>
#include "ScenarioLoader.h"
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <condition_variable>
#include "Town.h"

using namespace std;

const string MAP_NAME_PREFIX = "maps128-9/town_info_";
/***
generate the scene file based on the previous agent's location
***/
void Town::generate_scene_file(string scen_file_name,vector<std::shared_ptr<Agent>> new_agents,vector<std::shared_ptr<Agent>> agents_from_lock_in, bool is_init, unordered_set<int> deleted_From_Run){
    ScenarioLoader sl = ScenarioLoader(scen_file_name.c_str());

    unordered_map<int,std::shared_ptr<Agent>> new_agents_map; 
    for(std::shared_ptr<Agent> agent : new_agents){
        new_agents_map[agent->get_agent_id()] = agent;
    }
    for(auto& a : agent_map){
        new_agents_map[a.first] = a.second;
    }

    unordered_set<int> agent_in_file;
    set<pair<int,int>> existing_goal;
    //mutate the scen_file based on previous results
    for(int i=0;i<sl.GetNumExperiments();i++){
        Experiment e = sl.GetNthExperiment(i);
        if(agent_map.count(e.GetBucket())!=0){
            //check if the agents with new goals already exist here
            if(new_agents_map.count(e.GetBucket())!=0){
                std::shared_ptr<Agent> agent = agent_map[e.GetBucket()];
                //if the goal only has lock out, move to the nearest access cell 
                if(agent->get_lock_outT()==nullptr){
                    //since the goal is in the same town, global goal can be used
                    if(agent->get_goal_town_id()==townId && existing_goal.count({agent->get_goal().first-startX,agent->get_goal().second-startY})==0){
                        pair<int,int> goal = getLocalCoordFromGlobal(agent->get_goal());
                         e.SetGoal(goal.first,goal.second);
                         existing_goal.insert(getLocalCoordFromGlobal(agent->get_goal()));
                    }else{
                        pair<int,int> local= getLocalCoordFromGlobal(agent->get_current_grid());
                        pair<int,int> access_grid = findAcessCell(local,existing_goal);
                        if(access_grid!=std::make_pair<int,int>(-1,-1)){
                            //find the access cell
                            int row = access_grid.first;
                            int col = access_grid.second;
                            e.SetGoal(row,col);
                            existing_goal.insert({row,col});
                            agent->set_temp_goal({row+startX,col+startY});
                        }                   
                    }
                }else{
                    if(agents_to_lock_out.count(agent->get_agent_id())==0){
                        bool find_status = false;
                        pair<int,int> local = getLocalCoordFromGlobal(agent->get_lock_outT()->get_lock_location());
                        pair<int,int> access_grid = findLockAcessCell(local,existing_goal);
                        if(access_grid!=std::make_pair<int,int>(-1,-1)){
                            //find the access cell
                            int row = access_grid.first;
                            int col = access_grid.second;
                            e.SetGoal(row,col);
                            existing_goal.insert({row,col});
                            find_status = true;
                            agents_to_lock_out[agent->get_agent_id()] = {row+startX,col+startY};
                        }
                        if(!find_status){
                            pair<int,int> local= getLocalCoordFromGlobal(agent->get_current_grid());
                            pair<int,int> access_grid = findAcessCell(local,existing_goal);
                            if(access_grid!=std::make_pair<int,int>(-1,-1)){
                                //find the access cell
                                int row = access_grid.first;
                                int col = access_grid.second;
                                e.SetGoal(row,col);
                                existing_goal.insert({row,col});
                                agent->set_temp_goal({row+startX,col+startY});
                            }  
                        }
                    }
                }
                sl.SetNthExperiment(i,e);
                //if the goal has no lock in and no lockout, move to the goal
            }
            m->lock();
            Experiment e = sl.GetNthExperiment(i);
            if(checkAgentExistByLineNumber(i)){
                e.SetStart(agent_map[e.GetBucket()]->get_current_grid().first-startX,agent_map[e.GetBucket()]->get_current_grid().second-startY);
            }
            existing_goal.insert({e.GetGoalX(),e.GetGoalY()});
            m->unlock();
            sl.SetNthExperiment(i,e);
        }
    }

    //tobedeleted 
    unordered_set<int> deleted;
    //append to the scen file
    for(std::shared_ptr<Agent> agent : new_agents){
        if(agent->get_goal()==make_pair(-1,-1)){ 
            deleted.insert(agent->get_agent_id());
            continue;
        }
        //if scen file does not contain the agents, do the initialization
        if(is_init){
            lineId_to_agentId[sl.GetNumExperiments()] = agent->get_agent_id();
            pair<int,int> agent_cur_pos = getLocalCoordFromGlobal(agent->get_current_grid());
            pair<int,int> agent_goal = agent->get_goal();
            agent->set_temp_goal(agent_goal);
            if(agent->get_lock_outT()==nullptr){
                if(agent->get_goal_town_id()==townId){
                    agent_goal = getLocalCoordFromGlobal(agent->get_goal());
                }else{
                    pair<int,int> local= getLocalCoordFromGlobal(agent->get_current_grid());
                    pair<int,int> access_grid = findAcessCell(local,existing_goal);
                    if(access_grid!=std::make_pair<int,int>(-1,-1)){
                        //find the access cell
                        int row = access_grid.first;
                        int col = access_grid.second;
                        agent_goal.first = row;
                        agent_goal.second = col;
                        agent->set_temp_goal({row+startX,col+startY});
                    }  
                }
                
            }else{
                if(agents_to_lock_out.count(agent->get_agent_id())==0){
                 m->lock();
                
                    bool find_status = false;
                    pair<int,int> local = getLocalCoordFromGlobal(agent->get_lock_outT()->get_lock_location());
                    pair<int,int> access_grid = findLockAcessCell(local,existing_goal);
                    if(access_grid!=std::make_pair<int,int>(-1,-1)){
                        //find the access cell
                        int row = access_grid.first;
                        int col = access_grid.second;
                        agent_goal.first = row;
                        agent_goal.second = col;
                        find_status = true;
                        agents_to_lock_out[agent->get_agent_id()] = {row+startX,col+startY};
                    }
                    if(!find_status){
                        pair<int,int> local= getLocalCoordFromGlobal(agent->get_current_grid());
                        pair<int,int> access_grid = findAcessCell(local,existing_goal);
                        if(access_grid!=std::make_pair<int,int>(-1,-1)){
                            //find the access cell
                            int row = access_grid.first;
                            int col = access_grid.second;
                            agent_goal.first = row;
                            agent_goal.second = col;
                            agent->set_temp_goal({row+startX,col+startY});
                        }  
                    }
                }
                m->unlock();
            }
            int bucket = agent->get_agent_id();
            double distance = sqrt((agent_goal.first-agent_cur_pos.first)*(agent_goal.first-agent_cur_pos.first) + (agent_goal.second-agent_cur_pos.second)*(agent_goal.second-agent_cur_pos.second));
            Experiment e = Experiment(agent_cur_pos.first,agent_cur_pos.second,agent_goal.first,agent_goal.second,bucket,distance,"maps128-9/town_"+to_string(townId)+".map");
            existing_goal.insert({e.GetGoalX(),e.GetGoalY()});
            sl.AddExperiment(e); 
        }
    }

    for(std::shared_ptr<Agent> agent:agents_from_lock_in){
        //if scen file does not contain the agents, do the initialization
        if(agent_in_file.count(agent->get_agent_id())==1) continue;

        agent_in_file.insert(agent->get_agent_id());
        lineId_to_agentId[sl.GetNumExperiments()] = agent->get_agent_id();
        pair<int,int> agent_cur_pos = getLocalCoordFromGlobal(agent->get_current_grid());
        pair<int,int> agent_goal = agent->get_goal();
        agent->set_temp_goal(agent_goal);
        if(agent->get_lock_outT()==nullptr){
            if(agent->get_goal_town_id()==townId && existing_goal.count(getLocalCoordFromGlobal(agent->get_goal()))==0){
                agent_goal = getLocalCoordFromGlobal(agent_goal);
                existing_goal.insert(agent_goal);
            }else{
                pair<int,int> local= getLocalCoordFromGlobal(agent->get_current_grid());
                pair<int,int> access_grid = findAcessCell(local,existing_goal);
                if(access_grid!=std::make_pair<int,int>(-1,-1)){
                    //find the access cell
                    int row = access_grid.first;
                    int col = access_grid.second;
                    agent_goal.first = row;
                    agent_goal.second = col;
                    existing_goal.insert({row,col});
                    agent->set_temp_goal({row+startX,col+startY});
                } 
            }
        } else {
            m->lock();
            if(agents_to_lock_out.count(agent->get_agent_id())==0){
                bool find_status = false;
                pair<int,int> local = getLocalCoordFromGlobal(agent->get_lock_outT()->get_lock_location());
                pair<int,int> access_grid = findLockAcessCell(local,existing_goal);
                if(access_grid!=std::make_pair<int,int>(-1,-1)){
                    //find the access cell
                    int row = access_grid.first;
                    int col = access_grid.second;
                    agent_goal.first = row;
                    agent_goal.second = col;
                    existing_goal.insert({row,col});
                    find_status = true;
                    agents_to_lock_out[agent->get_agent_id()] = {row+startX,col+startY};
                }

                if(!find_status){
                    pair<int,int> local= getLocalCoordFromGlobal(agent->get_current_grid());
                    pair<int,int> access_grid = findAcessCell(local,existing_goal);
                    if(access_grid!=std::make_pair<int,int>(-1,-1)){
                        //find the access cell
                        int row = access_grid.first;
                        int col = access_grid.second;
                        agent_goal.first = row;
                        agent_goal.second = col;
                        existing_goal.insert({row,col});
                        agent->set_temp_goal({row+startX,col+startY});
                    } 
                }
            }
                m->unlock();
            }
        int bucket = agent->get_agent_id();
        double distance = sqrt((agent_goal.first-agent_cur_pos.first)*(agent_goal.first-agent_cur_pos.first) + (agent_goal.second-agent_cur_pos.second)*(agent_goal.second-agent_cur_pos.second));
        Experiment e = Experiment(agent_cur_pos.first,agent_cur_pos.second,agent_goal.first,agent_goal.second,bucket,distance,"maps128-9/town_"+to_string(townId)+".map");
        existing_goal.insert({e.GetGoalX(),e.GetGoalY()});
        sl.AddExperiment(e);
    
    }
     
    unordered_map<int,int> temp_lineId_to_agentId;
    ScenarioLoader sl_temp = ScenarioLoader();

    int line =0;
    for(int i=0;i<sl.GetNumExperiments();i++){
        //current line number is line
        //update the new line number
        //add the experiment that does not match the line id
        Experiment e = sl.GetNthExperiment(i);
        if(deleted.count(e.GetBucket())==0 && deleted_From_Run.count(e.GetBucket())==0){
            sl_temp.AddExperiment(e);   
            temp_lineId_to_agentId[line] =e.GetBucket();
            line++; 
        }else{
            agent_map.erase(e.GetBucket());
        }
    }
    existing_goal.clear();
    lineId_to_agentId = temp_lineId_to_agentId;
    for(int i=0;i<sl_temp.GetNumExperiments();i++){
        Experiment e = sl_temp.GetNthExperiment(i);
        std::shared_ptr<Agent> agent = agent_map[e.GetBucket()]; 
        if(existing_goal.count({static_cast<int>(e.GetGoalX()),static_cast<int>(e.GetGoalY())})>0){
            pair<int,int> local= getLocalCoordFromGlobal(agent->get_current_grid());
            pair<int,int> access_grid = findAcessCell(local,existing_goal);
            if(access_grid!=std::make_pair<int,int>(-1,-1)){
                //find the access cell
                int row = access_grid.first;
                int col = access_grid.second;
                 e.SetGoal(row,col);
                existing_goal.insert({row,col});
                agent->set_temp_goal({row+startX,col+startY});
                sl_temp.SetNthExperiment(i,e);
            } 
        } else {
            existing_goal.insert({static_cast<int>(e.GetGoalX()),static_cast<int>(e.GetGoalY())});   // SDMSDMSDM
        }
    }
    //delete the agents at goal
    sl_temp.Save(scen_file_name.c_str());

}

/***
each eecbs instance schedules the path in one town
at each timestamp
    - it calculates the latest path based on previous agent's location stored in the updated scen file
    - it reads the latest path
    - it updates the global coordinator's information to the path
    - it build the scen file according to previous agent's coordinates information
***/
void Town::EECBS(string map_file_name, string scen_file_name){
    string pathName = map_file_name+"_path.txt";
    int number_of_agents = agent_map.size();
    if(number_of_agents>0){
        std::stringstream stream;    
        stream << "./eecbs"  // eecbs if on Tigo's Laptop // "../EECBS/eecbs" if on AWS
            << " " // don't forget a space between the path and the arguments
            << "-m "
            << map_file_name
            <<" -a "
            <<scen_file_name
            <<" --outputPaths="
            <<pathName
            <<" -k "
            << to_string(number_of_agents)
            <<" -t 5 --suboptimality=2";
        system(stream.str().c_str());
    }
}


void Town::run(){

    string map_file_name = "maps128-9/town_"+to_string(townId)+".map";
    string scene_file_name = "scene_"+to_string(townId)+".scen";
    string pathName = map_file_name+"_path.txt";
    vector<std::shared_ptr<Agent>> agents_to_lock_out_message;
    unordered_map<int, bool> content;
    sendLockInMessage(content);
    sendLockOutMessage(agents_to_lock_out_message);

    //stuck here and wait for the start message
    wait_start_signal_from_global();

    // send the all new free agents message to the global
    vector<std::shared_ptr<Agent>> agent_list;
    for(auto& pair : agent_map){
        std::shared_ptr<Agent> agent = pair.second;
        agent_list.push_back(agent);
    }
    
    sendNewFreeAgentMessage(agent_list);

    ScenarioLoader sl = ScenarioLoader();
    sl.Save(scene_file_name.c_str());
    vector<std::shared_ptr<Agent>> agents_from_lock_in;
    unordered_set<int> deleted;
    while(true){
        timestep++;
        // read message to get agents with new goal & locks
        Message* message = nullptr;
        while(!message || message->get_message_type()!=AGENTS_W_NEW_GOAL_LOCKS_GTT){
            message = wait_message_from_global();
        }
        
        vector<std::shared_ptr<Agent>> new_agents = ((AgentListMessage*) message)->get_content();
        vector<std::shared_ptr<Agent>> agent_list;
        for (auto& a : new_agents) {
            if (a->get_current_grid() == a->get_goal()) {
                agent_list.push_back(a);  // already at goal
            }
        }

        generate_scene_file(scene_file_name,new_agents,agents_from_lock_in, timestep==1,deleted);
        
        deleted.clear();
        agents_from_lock_in.clear();
        
        // read message to get agents currently in lock_in
        while(message->get_message_type()!=AGENTS_IN_LOCK_INT_GTT){
            message = wait_message_from_global();
        }
        vector<std::shared_ptr<Agent>> agents_in_lock_in = ((AgentListMessage*) message)->get_content();
    
        // run eecbs
        EECBS(map_file_name,scene_file_name);

        read_path(pathName);
 
        unordered_map<int, bool> content;

        for(std::shared_ptr<Agent> agent: agents_in_lock_in){
            //check the access cells
            std::random_shuffle(directions.begin(), directions.end());
            for(vector<int>& direction : directions){
                //find the access cells location
                int row = agent->get_current_grid().first+direction[0]-startX;
                int col = agent->get_current_grid().second + direction[1]-startY;
                if(isValid(row,col)){  // TODO: @ TIGO
                    //check if the agent_in_grid has an agent and if it is a valid cell
                    if(checkIfCanPlaceAgent(row,col)){
                        //no agent, success message, add this to the town
                        // 1. agent_map 2. agent position 3. grid info
                        agents_from_lock_in.push_back(agent);
                        agent_map[agent->get_agent_id()] = agent;
                        agent_in_grid[convertLocToId(getGlobalCoordFromLocal({row,col}))] = agent->get_agent_id();
                        agent->set_current_grid({row+startX,col+startY});
                        agent->add_past_path({row+startX,col+startY}); 
                        content[agent->get_agent_id()] = true;
                        agent->set_lock_outT(nullptr);
                        agent->set_town_id(townId);
                       
                        break;
                    }
                }
                
            }
            if(agent_map.count(agent->get_agent_id())==0){
                content[agent->get_agent_id()] = false;
                vector<pair<int,int>> past_path = agent->get_past_path();
                past_path.push_back(agent->get_current_grid());
                agent->set_past_path(past_path);
            }
        }

        // send to the global the agents in lock out
        vector<std::shared_ptr<Agent>> agents_to_lock_out_message;

        for(auto& agent_pair : agents_to_lock_out){
            //if it reaches at the access cell, tell the global
            if(agent_map[agent_pair.first]->get_current_grid() == agent_pair.second){
                std::shared_ptr<Agent> agent = agent_map[agent_pair.first];
                agents_to_lock_out_message.push_back(agent_map[agent_pair.first]);
                //remove from the local town map
                agent_map.erase(agent_pair.first);
                deleted.insert(agent->get_agent_id());
                agent_in_grid[agent_pair.second.first*row_dimension + agent_pair.second.second] = -1;
            }
        }
        for(auto& i:deleted){
            agents_to_lock_out.erase(i);
        }
        
        for(auto& pair : agent_map){
            std::shared_ptr<Agent> agent = pair.second;
            //if agent position == agent's goal
            if(agent->get_current_grid()==agent->get_goal() && find(agent_list.begin(), agent_list.end(), agent) == agent_list.end()){
                //add this to the message 
                agent_list.push_back(agent);
            }
            
        }
        
       sendMessages(content,agents_to_lock_out_message,agent_list);
        
        setTempGoal();
    }

}

/***
    * initialize the data structures by reading the map information files
***/
void Town::initMap(){
//read the file to store the map
    fstream mapfile;
    string tp;
    mapfile.open(MAP_NAME_PREFIX+to_string(townId)+".map",ios::in); 
    int i =0;
    if (mapfile.is_open()){   
        while(getline(mapfile, tp)){
            if(i==1){
                tp = tp.substr(7);
                col_dimension = stoi(tp);
            }
            else if(i==2){
                tp = tp.substr(6);
                row_dimension = stoi(tp);
            }
            else if(i==3){
                //start location
                stringstream stream(tp);
                string startXStr,startYStr;
                stream>>startXStr >> startYStr;
                startX = stoi(startXStr);
                startY = stoi(startYStr);
            }
            if(i>=4){
                for(int j=0;j<tp.size();j++){
                    if(town_map.size()<=j){
                        town_map.push_back("");
                    }
                    if(tp[j]=='L'){
                        town_map[j]+="@";
                    }else{
                        town_map[j]+=tp[j];
                    }
                    if(tp[j]=='.'){
                        int y_loc = col_dimension-(i-4)+startY-1;
                        int x_loc = j + startX;
                        global->add_gridid_to_townid(y_loc*global->get_col_dim()+x_loc,townId);
                    }
                   
                }
            };
            i++;
        }
        mapfile.close(); //close the file object.
    }
}
/***
    * Constructor
***/ 
Town::Town( int townId, std::shared_ptr<Global> global) : townId(townId),global(global),start_signal(false),m( make_shared<mutex>()),cv(make_shared<condition_variable>())

{
    initMap();
}

/***
    * getters and setters
***/ 
int Town::getTownId(){
    return townId;
}

std::shared_ptr<Global> Town::get_global(){
    return this->global;
}

void Town::setAgentList(unordered_map<int,std::shared_ptr<Agent>> agentList){
    for(auto& pair:agentList){
        if(pair.second->get_town_id()==this->townId){
            agent_map[pair.first] = pair.second;
            std::pair<int,int> loc = pair.second->get_current_grid();
            agent_in_grid[loc.first*row_dimension+loc.second] = pair.second->get_agent_id();
        }
    }
}

/***
    * read the messages from global
    * read from the from_global messagequeue
***/
Message* Town::wait_message_from_global(){
    unique_lock<mutex> l(*(this->m));
    while(this->messages_from_global.empty()){
        cv->wait(l);
    }
    Message* temp = messages_from_global.front();
    messages_from_global.pop();
    return temp;
}

/***
    * global send the message to town
    * add to the from_global message queue
***/
void Town::send_message_to_town(Message* message){
    m->lock();
    this->messages_from_global.push(message);
    cv->notify_all();
    m->unlock();
}

/***
    * town needs to wait for the global to set up the agent list
***/
void Town::wait_start_signal_from_global(){
    unique_lock<mutex> l(*(this->m));
    while(!this->start_signal){
        cv->wait(l);
    }
}

/***
    * global tells the town to start the list
***/
void Town::start_the_town(){
    m->lock();
    this->start_signal = true;
    cv->notify_all();
    m->unlock();
}

/***
    * get the agent by the current line number in path file
***/
std::shared_ptr<Agent> Town::getAgentByLineNumber(int line_count){
    return agent_map[lineId_to_agentId[line_count]];
}
bool Town::checkAgentExistByLineNumber(int line_count){
    return lineId_to_agentId.count(line_count)==1 && agent_map.count(lineId_to_agentId[line_count])==1;
}


/***
    * calculate the local grid based on the global grid
***/
pair<int,int> Town::getLocalCoordFromGlobal(pair<int,int> grid){
    return {grid.first-startX,grid.second-startY};
}
/***
    * calculate the local grid based on the global grid
***/
pair<int,int> Town::getGlobalCoordFromLocal(pair<int,int> grid){
    return {grid.first+startX,grid.second+startY};
}
/***
    * calculate the location id based on the location
    * note the grid location is global
***/
int Town::convertLocToId(pair<int,int> grid){
    return grid.first*row_dimension+grid.second;
}

/***
    * remove an agent from a grid
***/
void Town::removeAgentFromGridMap(int locId){
    agent_in_grid[locId] = -1;
}
/***
read the path planned
genere the latest scenary file based on the path planned
***/
void Town::read_path(string fileName){
    std::ifstream myfile; 
    myfile.open(fileName);
    int line_count = 0;
    size_t pos = 0;
    if ( myfile.is_open() ) { 
        string line;
        while(getline(myfile,line)){
            if ((pos = line.find(':')) != std::string::npos){
                string agent_num = line.substr(0, pos);
                string coord = line.substr(pos+2);
                //convert the coord to pair 
                pair<int,int> coord_pair;
                if((pos = coord.find(',')) != string::npos){
                    string x_coord = coord.substr(1,pos);
                    string y_coord = coord.substr(pos+1);
                    coord_pair.first = stoi(y_coord);
                    coord_pair.second = stoi(x_coord);                    
                }else{
                     if(lineId_to_agentId.count(line_count)==1 && agent_map.count(lineId_to_agentId[line_count])==1){
                         std::shared_ptr<Agent> agent = getAgentByLineNumber(line_count);
                         pair<int,int> grid = agent->get_current_grid();
                         coord_pair = getLocalCoordFromGlobal(grid);
                    }
                }
                if(checkAgentExistByLineNumber(line_count)){
                    std::shared_ptr<Agent> agent = getAgentByLineNumber(line_count);
                    pair<int,int> before_grid = agent->get_current_grid();
                    int locId = convertLocToId(before_grid);
                    m->lock();
                    if(agent_in_grid[locId]==agent->get_agent_id()){
                        removeAgentFromGridMap(locId);
                    } 
                    m->unlock();
                    coord_pair = getGlobalCoordFromLocal(coord_pair);
                    agent->add_past_path(coord_pair);
                    agent_in_grid[coord_pair.first*row_dimension+coord_pair.second]  = agent->get_agent_id();
                    agent->set_current_grid(coord_pair);
                } 
            }
            line_count++;
        }
    }
}

bool Town::isValid(int row, int col){
    return row>=0 && row<row_dimension && col>=0 && col<col_dimension;
}



/*
    set up temporary goal for each agent 
*/
void Town::setTempGoal(){
    for(auto& pair : agent_map){
        std::shared_ptr<Agent> agent = pair.second;
        agent->set_temp_goal(agent->get_goal());
    }
}
void Town::sendMessages(unordered_map<int, bool>& content,vector<std::shared_ptr<Agent>>& agents_to_lock_out_message,vector<std::shared_ptr<Agent>>& agent_list){
    sendLockInMessage(content);
    sendLockOutMessage(agents_to_lock_out_message);
    sendNewFreeAgentMessage(agent_list);
}
void Town::sendLockInMessage(unordered_map<int, bool>& content){
    Message* agents_in_lock_inT_status_message = new LockinStatusResponseMessage(AGENTS_STATUS_IN_LOCK_INT_TTG,content);
    global->town_to_global(townId,agents_in_lock_inT_status_message);
}
void Town::sendLockOutMessage(vector<std::shared_ptr<Agent>>& agents_to_lock_out_message){
     Message* agents_in_lock_outT_status_message = new AgentListMessage(AGENTS_IN_LOCK_OUT_TTG,agents_to_lock_out_message);
    global->town_to_global(townId,agents_in_lock_outT_status_message);
}
void Town::sendNewFreeAgentMessage(vector<std::shared_ptr<Agent>>& agent_list){
    Message* new_free_agents = new AgentListMessage(NEW_FREE_AGENTS_TTG,agent_list);
    global->town_to_global(townId,new_free_agents);
}
/**
 * find local access cell, make sure the grid pass in is lcoal coord
*/
pair<int,int> Town::findAcessCell(pair<int,int> grid,set<pair<int,int>>& existing_goal){
    std::random_shuffle(directions.begin(), directions.end());
    for(int j=0;j<(int)directions.size();j++){
        int row = grid.first + directions[j][0];
        int col = grid.second + directions[j][1];
        if(isValid(row, col) && existing_goal.count({row,col})==0){
            if(town_map[row][col_dimension-col-1]=='.'){
                return {row,col};
            }
        } 
    }   
    return {-1,-1};
}

pair<int,int> Town::findLockAcessCell(pair<int,int> grid,set<pair<int,int>>& existing_goal){
    std::random_shuffle(directions.begin(), directions.end());
    for(vector<int>& direction : directions){
        //find the access cells location
        int row = grid.first+ direction[0];
        int col = grid.second + direction[1];
        if(isValid(row, col)&& existing_goal.count({row,col})==0){
            //check if the agent_in_grid has an agent and if it is a valid cell
            if(checkIfCanPlaceAgent(row,col)){
                return {row,col};
            }
        } 
    }
    return {-1,-1};
}

bool Town::checkIfCanPlaceAgent(int row, int col){
    int global = convertLocToId(getGlobalCoordFromLocal({row,col}));
    return (town_map[row][col_dimension-col-1]=='.' && ((agent_in_grid.count(global)==0) || (agent_in_grid.count(global)!=0 && agent_in_grid[global]==-1)));
}
