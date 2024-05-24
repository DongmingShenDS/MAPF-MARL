#include "Agent.h"

Agent::Agent(int agent_id, int town_id, pair<int, int> current_grid){
    this->agent_id = agent_id;
    this->town_id = town_id;
    this->current_grid = current_grid;
    this->lock_inT = nullptr;
    this->lock_outT = nullptr;
}

void Agent::to_string() {
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
    cout << "\tagent id " << agent_id << endl;
    cout << "\ttown_id " << town_id << endl;
    cout << "\tgoal town_id " << goal_town_id << endl;
    cout << "\tgoal (" << goal.first << "," << goal.second << ")" << endl;
    cout << "\tcurrent_grid (" << current_grid.first << "," << current_grid.second << ")" << endl;
    if (lock_inT) 
        cout << "\tlock_inT (" << lock_inT->get_lock_location().first << "," << lock_inT->get_lock_location().second << ")" << endl;
    else 
        cout << "\tlock_inT (null)" << endl;
    if (lock_outT) 
        cout << "\tlock_outT (" << lock_outT->get_lock_location().first << "," << lock_outT->get_lock_location().second << ")" << endl;
    else 
        cout << "\tlock_outT (null)" << endl;
    cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
}

int Agent::get_agent_id(){
    return agent_id;
}

int Agent::get_town_id(){
    return town_id;
}

pair<int, int> Agent::get_goal(){
    return goal;
}

pair<int, int> Agent::get_temp_goal(){
    return temp_goal;
}

pair<int, int> Agent::get_current_grid(){
    return current_grid;
}

Lock* Agent::get_lock_inT(){
    return lock_inT;
}

Lock* Agent::get_lock_outT(){
    return lock_outT;
}

int Agent::get_goal_town_id(){
    return this->goal_town_id;
}

vector<pair<int, int>> Agent::get_past_path() {
    return past_path;
}

void Agent::set_agent_id(int agent_id){
    this->agent_id = agent_id;
}

void Agent::set_town_id(int town_id){
    this->town_id = town_id;
}

void Agent::set_goal(pair<int, int> goal){
    this->goal = goal;
}

void Agent::set_temp_goal(pair<int, int> goal){
    this->temp_goal = goal;
}

void Agent::set_goal_town_id(int gtid) {
    this->goal_town_id = gtid;
}

void Agent::set_current_grid(pair<int, int> curr){
    this->current_grid = curr;
}

void Agent::set_lock_inT(Lock* in){
    this->lock_inT = in;
}

void Agent::set_lock_outT(Lock* out){
    this->lock_outT = out;
}

void Agent::set_past_path(vector<pair<int, int>> pp){
    this->past_path = pp;
}

void Agent::add_past_path(pair<int, int> p)
{
    this->past_path.push_back(p);
}
