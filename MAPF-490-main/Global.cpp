#include "Global.h"
#include <map>
#include <nlohmann/json.hpp>
using json = nlohmann::json;  // convicience
using namespace std;
Global::Global(){ 
    this->col_dim = 128;
    this->m = make_shared<mutex>();
    this->cv = make_shared<condition_variable>();
}

void Global::run(){
    bool debug = true;

    // 0. pre-process (hard-coding for now)
    pair<int,int> null_pair = make_pair(-1, -1);
    bool finish = false;
    for (int i=0;i<all_towns.size();i++){
        this->msg1_to_town[i] = new AgentListMessage(AGENTS_W_NEW_GOAL_LOCKS_GTT,{});
        this->msg2_to_town[i] = new AgentListMessage(AGENTS_IN_LOCK_INT_GTT,{});
    }
    // wait town message, handle town message
    this->wait_message_from_all_towns();
    while(!msg1_from_town.empty()){  // 5.1
        AgentListMessage* pair = (AgentListMessage*) msg1_from_town.front().second;
        msg1_from_town.pop();
        vector<std::shared_ptr<Agent>> content = pair->get_content();
        for (auto& agent : content){
            this->at_lock_outT_agents[agent->get_agent_id()] = agent;
        }
    }
    vector<std::shared_ptr<Agent>> temp1;  // 5.2
    msg2_to_road = new AgentListMessage(AGENTS_IN_LOCK_OUTT_GTR, temp1);
        while(!msg2_from_town.empty()){
        LockinStatusResponseMessage* message = (LockinStatusResponseMessage*) msg2_from_town.front().second;
        unordered_map<int, bool> content = message->get_content();
        msg2_from_town.pop();
        for (auto& msg_pair : content){
            int agent_id = msg_pair.first;
            bool if_accepted = msg_pair.second;
            if (if_accepted) {
                std::shared_ptr<Agent> agent = this->all_agents[agent_id];
                ((AgentListMessage*)(this->msg2_to_road))->add_content(agent);
                int lock_to_free_id = agent->get_lock_inT()->get_lock_id(); 
                // cout << "unlock lock_inT for agent id " << agent->get_agent_id() << endl;
                this->all_locks[lock_to_free_id]->unlock();
            }  
        }
        this->lock_inT_agents.clear();
    }
    while(!msg3_from_town.empty()){  // 5.3
        AgentListMessage* message = (AgentListMessage*) msg3_from_town.front().second;
        msg3_from_town.pop();
            vector<std::shared_ptr<Agent>> content = message->get_content();
        for (auto& agent : content){
            this->new_agents[agent->get_agent_id()] = agent;
        }
    }
    // send road first message to get started
    vector<std::shared_ptr<Agent>> temp;  // 6.1
    msg1_to_road = new AgentListMessage(AGENTS_IN_LOCK_OUTT_GTR, temp);
    msg2_to_road = new AgentListMessage(AGENTS_STATUS_IN_LOCK_INT_GTR, temp);
    this->road->send_msg1_to_road(this->msg1_to_road);
    this->road->send_msg2_to_road(this->msg2_to_road);  // 6.2


    while (!finish) {
        for(int i=0;i<all_towns.size();i++){
            ((AgentListMessage*)(this->msg1_to_town[i]))->set_content({});
            ((AgentListMessage*)(this->msg2_to_town[i]))->set_content({});
        }
        this->msg2_to_road=nullptr;  // clear msg1_to_road for next round 
        this->msg1_to_road=nullptr;  // clear msg1_to_road for next round    

        this->timestep += 1;
        cout << "\nTIME STEP " << this->timestep << endl;
        // 1. get new free agents' next goal
        // cout << "timestep " << timestep << "new agent size " << this->new_agents.size() << endl;
        cout << "global get new agent ids ";
        for (auto& pair : this->new_agents) {
            int agent_id = pair.first;
            cout << agent_id << ", ";
            std::shared_ptr<Agent> agent = pair.second;
            // deque<std::pair<int,int>> agent_task_queue = this->tasks[agent_id];
            if (!this->tasks[agent_id].empty()) {
                std::pair<int, int> goal =  this->tasks[agent_id].front();
                this->tasks[agent_id].pop_front();
                agent->set_goal(goal);
                // cout << agent_id << goal.first << goal.second << "SDMSDMSDMSDMSDMSDMSDMSD " << timestep << endl;
            } else {  // if no next goal, assigne (-1,-1) as goal
                agent->set_goal(null_pair);
                agent->set_current_grid({-1*agent->get_agent_id()-1,-1});
                // cout << "global agent goal complete for agent id " << agent_id << "SDMSDMSDMSDMSDMSDMSDMSD" << endl;
            }
        }
        cout << "from town" << ",\n ";

        // cout << timestep<<" "<<"global step 2: "<< endl;
        // 2. assign locks to each of the above agents, lock the locks
        // cout << timestep<<" "<<"global step 2 number new agents = "<< this->new_agents.size() << endl;
        for (auto& pair : this->new_agents){
            // cout << "\t\t agent index " << pair.first << endl;
            // cout << timestep<<" "<<"global step 2 for loop: agent id" << pair.first << endl;
            std::shared_ptr<Agent> agent = pair.second;
            std::pair<int, int> goal = agent->get_goal();
            int townid_goal = 0;
            if (agent->get_goal() == null_pair) {
                townid_goal = agent->get_town_id();
            }
            else {
                townid_goal = this->gridid_to_townid[goal.second * this->col_dim + goal.first];  
            }
            int townid_start = agent->get_town_id();
            agent->set_goal_town_id(townid_goal);
            if (townid_start != townid_goal){  // different towns, assign locks
                // cout << "size of town_locks[townid_start] = " << town_locks[townid_start].size() << "for town id " << townid_start << endl;
                // cout << "size of town_locks[townid_goal] = " << town_locks[townid_goal].size() << "for town id " << townid_goal << endl;
                for (auto l1 : this->town_locks[townid_start]) {
                    for (auto l2 : this->town_locks[townid_goal]) {
                        // reachable is true = if l1 can reach l2
                        bool reachable = false;
                        if (this->lock_reachability[l1->get_lock_id()].count(l2->get_lock_id())) {
                            // cout << "lock pair reachable" << l1->get_lock_id() << l2->get_lock_id() << endl;
                            reachable = true;
                        }
                        if (reachable && !l1->get_is_locked() && !l2->get_is_locked()) {  
                            // cout << "locked" << l1->get_lock_id() << l2->get_lock_id() << "for agent id " << pair.first << endl;
                            agent->set_lock_inT(l2);
                            agent->set_lock_outT(l1);
                            l1->lock();
                            l2->lock();
                            goto break_loop1;
                        }
                    }
                }
                // cout << "no available locks (so null) for agent id " << pair.first << endl;
                this->tasks[pair.first].push_front(goal);  
                agent->set_lock_inT(nullptr);  // TODO: NEW
                agent->set_lock_outT(nullptr);
                agent->set_goal(agent->get_current_grid());
                agent->set_goal_town_id(gridid_to_townid[agent->get_goal().second * this->col_dim + agent->get_goal().first]);
                this->new_acc_agents[pair.first] = agent;
                this->all_agents[pair.first] = agent;
                goto break_loop2;
            }
            else {  // same town, no need to assign locks
                agent->set_lock_inT(nullptr);
                agent->set_lock_outT(nullptr);
                this->new_accept_agents[pair.first] = agent;
                this->new_acc_agents[pair.first] = agent;
                this->all_agents[pair.first] = agent;
            }
            break_loop1: {
                this->new_accept_agents[pair.first] = agent;
                this->new_acc_agents[pair.first] = agent;
                this->all_agents[pair.first] = agent;
            } 
            break_loop2: {}
        }
        cout << "GLOABL NEW ACCEPT AGENTS " << "timestep " << timestep << endl; 
        for (auto& pair : this->new_accept_agents){  // remove agent from this->new_agents (passed)
            // cout << "add to new_accept_agents id " << pair.first << endl;
            this->new_agents.erase(pair.first);
            // if (!pair.second->get_lock_inT()) {
            //     // cout << "agent id " << pair.first << "lock id null " << endl;
            // } else {
            //     // cout << "agent id " << pair.first << "lock id " << pair.second->get_lock_inT()->get_lock_id() << endl;
            // }
            // pair.second->to_string();
        }
        // new_agents.clear();

        // cout<< timestep<<" "<< "global step 3: "<< endl;
        // 3.1 send to each town newly arrived agents there, each with (a_id, a_state)
        for (auto& pair : this->new_acc_agents){  // passed - TODO, need to keep track of all, not just new_accept
            // if(!pair.second) cout << "null agent" << endl; 
            std::shared_ptr<Agent> agent = pair.second;
            
            // agent->to_string();
            int town_id = agent->get_town_id();
            ((AgentListMessage*)(this->msg1_to_town[town_id]))->add_content(agent);
            // cout << "size" << ((AgentListMessage*)(this->msg1_to_town[town_id]))->get_content().size() << endl;
        }
        
        // cout << "before clear"<<endl;
        this->new_accept_agents.clear();  // clear new_accepted_agents for next round
        this->new_acc_agents.clear();
        // cout << "clear" << endl;
        this->send_alltowns_message(1);
        // cout << "successfully send send_alltowns_message(1)" << endl;
        
        // 3.2. send to each town all agents currently in lock_inT
        for (auto& pair : this->lock_inT_agents){
            std::shared_ptr<Agent> agent = pair.second;
            if (agent->get_lock_inT() != nullptr) {
                // cout << "?????????????????????agent in lock in " << agent->get_agent_id()<<endl;
                // agent->to_string();
                int town_id = agent->get_lock_inT()->get_town_id();
                ((AgentListMessage*)(this->msg2_to_town[town_id]))->add_content(agent);
            }
        }
        this->send_alltowns_message(2);
        // cout << "successfully send send_alltowns_message(2)" << endl;

        // cout << timestep<<" "<< "global step 4: "<< endl;
        // 4. wait message from town...
        this->wait_message_from_all_towns();

        // cout << timestep<<" "<< "global step 5: "<< endl;
        // 5.1 receive from towns all agents newly arrived in lock_outT
        // cout << "timestep " << timestep << "SDMSDMSDMSDMSDMSDMSDMSDM msg1_from_town size " << msg1_from_town.size() << endl;
        while(!msg1_from_town.empty()){
            AgentListMessage* pair = (AgentListMessage*) msg1_from_town.front().second;
            msg1_from_town.pop();
            vector<std::shared_ptr<Agent>> content = pair->get_content();
            for (auto& agent : content){
                // cout << "timestep " << timestep << "SDMSDMSDMSDMSDMSDMSDMSDM msg1_from_town agent " << agent->get_agent_id() << endl;
                this->at_lock_outT_agents[agent->get_agent_id()] = agent;
            }
        }
        // cout << "timestep " << timestep << "SDMSDMSDMSDMSDMSDMSDMSDM at_lock_outT_agents size " << at_lock_outT_agents.size() << endl;

        // 5.2 receive from towns all agents in lock_inT's status
        vector<std::shared_ptr<Agent>> temp1;
        msg2_to_road = new AgentListMessage(AGENTS_STATUS_IN_LOCK_INT_GTR, temp1);
         while(!msg2_from_town.empty()){
            LockinStatusResponseMessage* message = (LockinStatusResponseMessage*) msg2_from_town.front().second;
            unordered_map<int, bool> content = message->get_content();
            msg2_from_town.pop();
            for (auto& msg_pair : content){
                int agent_id = msg_pair.first;
                bool if_accepted = msg_pair.second;
                if (if_accepted) {
                    // cout << "SSSSSSSSS timestep " << this->timestep << " agent id " << agent_id << "accepted from town lockINT" << endl;
                    std::shared_ptr<Agent> agent = this->all_agents[agent_id];
                    ((AgentListMessage*)(this->msg2_to_road))->add_content(agent);
                    int lock_to_free_id = agent->get_lock_inT()->get_lock_id(); 
                    // cout << "unlock lockint" << endl;
                    this->all_locks[lock_to_free_id]->unlock();
                    agent->set_lock_inT(nullptr);
                } 
            }
            this->lock_inT_agents.clear();
        }

        // 5.3 receive from towns all new free agents [timestep finish for all towns]
        while(!msg3_from_town.empty()){
            AgentListMessage* message = (AgentListMessage*) msg3_from_town.front().second;
            // cout << "timestep " << timestep << "town send global new free agent size " << message->get_content().size() << endl;
            msg3_from_town.pop();
            vector<std::shared_ptr<Agent>> content = message->get_content();
            for (auto& agent : content){
                this->new_agents[agent->get_agent_id()] = agent;
            }
        }
        
        // cout << timestep<<" "<< "global step 6.0: "<< endl;
        // 6.0 wait ready message from road (passed)
        while (!this->ready_msg) {
            // cout << "GLOBAL in while" << endl;
            this->wait_ready_message_from_road();
        }
        this->ready_msg = false;

        for (auto & a : all_agents) {
            if (a.second->get_goal() == null_pair) {
                a.second->add_past_path(a.second->get_current_grid());
            }
        }
        
        cout << "ROAD LOAD #AGENTS = " << road->get_road_load() << endl;
        // if (road->get_road_load() > 1000) {
        //     cout << "FALSE" << endl;
        //     return;
        // }
        // HERE EVERYTHING SHOULD SYNC
        vis_to_json();
        if (timestep < 0) usleep(500000);

        if (debug) {    
            this->m->lock();
            for (auto& agent1 : this->all_agents) {
                //cout << "agent " << agent1.first << "'s path size at timestep " << timestep << " is " << agent1.second->get_past_path().size()<<endl;
            }
            this->m->unlock();
        }   
        if (debug) {
            for (auto& agent1 : this->all_agents) {
                for (auto& agent2 : this->all_agents) {
                    if (agent1.first == agent2.first) {
                        continue;
                    }
                    if (agent1.second->get_current_grid().first == agent2.second->get_current_grid().first && agent1.second->get_current_grid().second == agent2.second->get_current_grid().second) {
                        m->lock();
                        unordered_set<int> temp;
                        cout << "collision" << endl;
                        temp.insert(agent1.first);
                        temp.insert(agent2.first);
                        agent1.second->to_string();
                        agent2.second->to_string();
                        print_agent_path(temp);
                        cout << "COLLISIONCOLLISIONCOLLISIONCOLLISIONCOLLISIONCOLLISIONCOLLISION" << endl;
                        cout << "COLLISIONCOLLISIONCOLLISIONCOLLISIONCOLLISIONCOLLISIONCOLLISION" << endl;
                        cout << "timestep: " << this->timestep << endl;
                        m->unlock();
                        return;
                    }
                }
            }
        }
        if (debug && false) {
            unordered_set<int> small_ids;
            unordered_set<int> all;
            for (auto& agent1 : this->all_agents) {
                for (auto& agent2 : this->all_agents) {
                    if (agent1.first == agent2.first) {
                        continue;
                    }
                    if (agent1.second->get_past_path().size() < timestep || agent2.second->get_past_path().size() < timestep) {
                        continue;
                    }
                    if (agent1.second->get_past_path().size() > agent2.second->get_past_path().size()) {
                        small_ids.insert(agent2.first);
                        // return;
                    }
                }
                all.insert(agent1.first);
            }
            if (small_ids.size() > 0) {
                cout << "MMMMMMMM" << endl;
                cout << "MMMMMMMM" << endl;
                cout << "MMMMMMMM" << endl;
                cout << "timestep: " << this->timestep << endl;
                for(auto a: small_ids){
                    cout << "small id is " <<a<< endl;
                }
                for (auto a : all) {
                    all_agents[a]->to_string();
                }
                this->print_agent_path(all);
                return;
            }
        }
        // if (timestep==2) break; 
        // cout << timestep<<" "<< "global step 6.1: "<< endl;
        // 6.1 send to road newly arrived agents in lock_outT
        // cout << "6.1 start " << this->at_lock_outT_agents.size() << endl;
        vector<std::shared_ptr<Agent>> temp;
        msg1_to_road = new AgentListMessage(AGENTS_IN_LOCK_OUTT_GTR, temp);
        for (auto& pair : this->at_lock_outT_agents){
            // cout << "6.1 loop" << endl;
            std::shared_ptr<Agent> agent = pair.second;
            ((AgentListMessage*)(this->msg1_to_road))->add_content(agent);
        }
        this->road->send_msg1_to_road(this->msg1_to_road);
        // cout << "before 6.2" << endl;
        if(((AgentListMessage*)this->msg2_to_road)->get_content().size() > 0){
            m->lock();
            for(auto& p:((AgentListMessage*)this->msg2_to_road)->get_content()){
            // cout << "timestep " <<timestep << " lock in status success agents " << p->get_agent_id() << endl;
            m->unlock();
         }
        }
         
        // 6.2 send to road all agents in lock_inT's status
        this->road->send_msg2_to_road(this->msg2_to_road);
        
        // cout << timestep<<" "<< "global step 7: "<< endl;
        // 7. wait message from road...
        this->wait_message_from_all_roads();

        // cout << timestep<<" "<< "global step 8: "<< endl;
        // 8.1 receive from road all agents newly get out of lock_outT
        // cout << "before for 1" << endl;
        for (auto& agent : ((AgentListMessage*)(this->msg1_from_road))->get_content()) {
            int lock_to_free_id = agent->get_lock_outT()->get_lock_id(); 
            // cout << "unlock lock_outT for agent id " << agent->get_agent_id() << endl;
            this->all_locks[lock_to_free_id]->unlock();
            agent->set_lock_outT(nullptr);
        }
        this->msg1_from_road = nullptr;
        // cout << "after for 1" << endl;
        // 8.2 receive from road all agents currently in lock_inT
        // cout << "before for 2" << endl;
        for (auto& agent : ((AgentListMessage*)(this->msg2_from_road))->get_content()) {
            this->lock_inT_agents[agent->get_agent_id()] = agent;
        }
        // cout << "after for 21" << endl;
        this->msg2_from_road = nullptr;
        
        this->at_lock_outT_agents.clear();  // clear lock_outT_agents for next round
        // cout << "after for 2" << endl;
        // this->print_agent_path();
        
        if(timestep==15000) {
            unordered_set <int> tmp;
            for(auto & agent1:this->all_agents){
                for(int i=0;i<agent1.second->get_past_path().size()-1;i++){
                    if(abs(agent1.second->get_past_path()[i].first-agent1.second->get_past_path()[i].first) + abs(agent1.second->get_past_path()[i].second-agent1.second->get_past_path()[i].second)>1){
                        cout << "agent id " << agent1.first << " skip a grid" << endl;
                        return;
                    }
                }
                tmp.insert(agent1.second->get_agent_id());
            }
            cout << "NO AGENT SKIP GRIDS" << endl;
            // this->print_agent_path(tmp);
            unordered_set<int> conflict_agent_ids;
            for(int i=0;i<timestep;i++){
                map<pair<int,int>,int> s;
                for(auto & agent1 :this->all_agents){
                    if(agent1.second->get_past_path().size() <= i) continue;
                    if(s.count(agent1.second->get_past_path()[i])==1){
                        conflict_agent_ids.insert(agent1.second->get_agent_id());
                        conflict_agent_ids.insert(s[agent1.second->get_past_path()[i]]);  // what is this?
                        cout << "collision find at timestep " << i << " with location " << agent1.second->get_past_path()[i].first <<" "<<agent1.second->get_past_path()[i].second <<" between agents " << agent1.second->get_agent_id() <<" " << s[agent1.second->get_past_path()[i]] <<endl;
                    }
                    s[agent1.second->get_past_path()[i]] = agent1.second->get_agent_id();
                }
            }
            cout << "conflict_agent_ids size " << conflict_agent_ids.size() << endl;
            // conflict_agent_ids.clear();
            // conflict_agent_ids.insert(30);

            this->print_agent_path(conflict_agent_ids);
        
            // this->print_agent_path();   // print all agents' path
            return;  
        };


        // if (debug) {
        //     for (auto& agent1 : this->all_agents) {
        //         if (agent1.second->get_current_grid() != agent1.second->get_past_path().back()) {
        //             cout << "MISMATCH" << endl;
        //             cout << "MISMATCH" << endl;
        //             cout << "MISMATCH" << endl;
        //             cout << "timestep: " << this->timestep << endl;
        //             cout << "MISMATCH For Agent " << agent1.second->get_agent_id() << endl;
        //             cout << "Its current grid is " << agent1.second->get_current_grid().first << ", " << agent1.second->get_current_grid().second << endl;
        //             cout << "Its past path back is " << agent1.second->get_past_path().back().first << ", " << agent1.second->get_past_path().back().second << endl;

        //             return;
        //         }
        //     }
        // }
        
    }
}

void Global::lock(Lock* l){
    l->lock();
}

void Global::unlock(Lock* l){
    l->unlock();
}

void Global::send_alltowns_message(int msg_index){
    if (msg_index == 1) {
        for (auto& pair : this->msg1_to_town){
            // cout << "send towns message 1 to individual town " << pair.first << endl;
            // cout << "*** send message to town id " << pair.first << " with size " << ((AgentListMessage*)(pair.second))->get_content().size() << endl;
            this->all_towns[pair.first]->send_message_to_town(pair.second);
        }
    }
    if (msg_index == 2) {
        for (auto& pair : this->msg2_to_town){
            // cout << "send towns message 2 to individual town " << pair.first << endl;
            this->all_towns[pair.first]->send_message_to_town(pair.second);
        }
    }
}

void Global::wait_ready_message_from_road() {
    while(!this->ready_msg) {
        wait_one_ready_message();
    }
    // cout << "Global Get READY messgae from Road" << endl;
}

void Global::wait_message_from_all_roads(){
    while (!this->msg1_from_road || !this->msg2_from_road) {
        // cout << "Global Before wait road" << endl;
        wait_one_road_message();
        // cout << "Global after wait road" << endl;
    }
    // cout << "Global Get 2 messages from Road" << endl;
}

void Global::wait_message_from_all_towns(){
    bool stop = false;
    int town_size = this->all_towns.size();
    while (!stop) {
        wait_one_town_message();
        // cout << "global get message from one town" << endl;
        // cout << ".......msg1_from_town size: " << msg1_from_town.size() << endl;
        // cout << ".......msg2_from_town size: " << msg2_from_town.size() << endl;
        // cout << ".......msg3_from_town size: " << msg3_from_town.size() << endl;
        if (this->msg1_from_town.size() == town_size && this->msg2_from_town.size() == town_size && this->msg3_from_town.size() == town_size) {
            stop = true;
        }
    }
    // cout << "global FINISH wait_message_from_all_towns" << endl;
}

void Global::wait_one_town_message(){
    // cout << "global start to wait one town message" << endl;
    unique_lock<mutex> l(*(this->m));  // 这里面干的事越少越好
    while(this->temp_msg_from_town.empty()){
        // cout << "===global enter while" << endl;
        this->cv->wait(l);
    }
    // cout << "global after while " << temp_msg_from_town.size() << endl;  // not reach
    pair<int, Message*> msg = this->temp_msg_from_town.front();
    // msg.second->to_string();
    int msg_type = msg.second->get_message_type();
    if (msg_type == 0) {  // NEW_FREE_AGENTS_TTG
        // cout << "=======global get NEW_FREE_AGENTS_TTG" << endl;
        this->msg3_from_town.push(msg);
    }
    if (msg_type == 6) {  // AGENTS_IN_LOCK_OUT_TTG
        // cout << "=======global get AGENTS_IN_LOCK_OUT_TTG" << endl;
        this->msg1_from_town.push(msg);
    }
    if (msg_type == 7) {  // AGENTS_STATUS_IN_LOCK_INT_TTG
        // cout << "=======global get AGENTS_STATUS_IN_LOCK_INT_TTG" << endl;
        this->msg2_from_town.push(msg);
    }
    this->temp_msg_from_town.pop();
    // cout << "after reading... town to global queue size " <<  temp_msg_from_town.size() << endl;
}

void Global::wait_one_road_message(){
    unique_lock<mutex> l(*(this->m));  // 这里面干的事越少越好
    while(this->temp_msg_from_road.empty()){
        this->cv->wait(l);
    }
    Message* msg = this->temp_msg_from_road.front();
    int msg_type = msg->get_message_type();
    this->temp_msg_from_road.pop();
    if (msg_type == 1) {  // AGENTS_IN_LOCK_INT_RTG
        // cout << "Receive msg2 from Road" << endl;
        this->msg2_from_road = msg;
    }
    if (msg_type == 10) {  // AGENTS_LEAVE_LOCK_OUTT_RTG
        // cout << "Receive msg1 from Road"<< endl;
        this->msg1_from_road = msg;
    }
}

void Global::wait_one_ready_message(){
    unique_lock<mutex> l(*(this->m));  // 这里面干的事越少越好
    while(this->temp_road_ready_queue.empty()){
        this->cv->wait(l);
    }
    Message* msg = this->temp_road_ready_queue.front();
    int msg_type = msg->get_message_type();
    this->temp_road_ready_queue.pop();
    if (msg_type == 11) {   // R_READY
        // cout << "Receive ready_msg from Road" << endl;
        this->ready_msg = true;
    }
}

void Global::town_to_global(int town_id, Message* msg){
    this->m->lock();
    // cout << "town id" << town_id << " sending global message" << endl;
    this->temp_msg_from_town.push(make_pair(town_id, msg));
    if(msg->get_message_type()==AGENTS_IN_LOCK_OUT_TTG){
        // cout << "town sends lock out message with size " << ((AgentListMessage*) msg)->get_content().size() <<endl;
    }
    // cout << "sending... " << town_id<<" town to global queue size " <<  temp_msg_from_town.size() << endl;
    this->cv->notify_all();
    this->m->unlock();
}

void Global::road_to_global(Message* msg){
    this->m->lock();
    if(msg->get_message_type() == 11){
        temp_road_ready_queue.push(msg);
    }
    else{
        this->temp_msg_from_road.push(msg);
    }
    
    this->cv->notify_all();
    this->m->unlock();
}

// setters
void Global::set_tasks(unordered_map<int, deque<pair<int, int>>> t){
    this->tasks = t;
}

void Global::set_road(std::shared_ptr<Road> r){
    this->road = r;
}

void Global::set_all_towns(unordered_map<int, std::shared_ptr<Town>> t){
    this->all_towns = t;
}

void Global::set_all_agents(unordered_map<int, std::shared_ptr<Agent>> a){
    this->all_agents = a;
}

void Global::set_all_locks(unordered_map<int, Lock*> l){
    this->all_locks = l;
}

void Global::set_town_locks(unordered_map<int, vector<Lock*>> tl){
    this->town_locks = tl;
}

void Global::set_gridid_to_townid(unordered_map<int, int> gtt){
    this->gridid_to_townid = gtt;
}

void Global::set_town_grids(unordered_map<int, vector<int>> tg){
    this->town_grids = tg;
}

void Global::set_lock_reachability(unordered_map<int, unordered_set<int>> lr){
    this->lock_reachability = lr;
}

void Global::print_agent_path(unordered_set<int> agent_id_set) {
    if (agent_id_set.empty()) {
    m->lock();
    cout << "called print_agent_path" << endl;
    for (auto& agent : this->all_agents) {
        if(agent_id_set.find(agent.first) != agent_id_set.end()){
            cout << "===== timestep " << this->timestep << " agent id " << agent.second->get_agent_id() << " path =====" << endl;
            for (auto& pair : agent.second->get_past_path()) {
                cout << "(" << pair.first << "," << pair.second << ") -> ";
            }
            cout << endl;
            cout << "with path size " << agent.second->get_past_path().size() << endl;
            cout << "==========" << endl;
        }
        
    }
    m->unlock(); 
    } else {
    m->lock();
    cout << "called print_agent_path" << endl;
    for (auto& agent : this->all_agents) {
    if(agent_id_set.find(agent.first) != agent_id_set.end()){  // not working???
        cout << "===== timestep " << this->timestep << " agent id " << agent.second->get_agent_id() << " path =====" << endl;
        for (auto& pair : agent.second->get_past_path()) {
            cout << "(" << pair.first << "," << pair.second << ") -> ";
        }
        cout << endl;
        cout << "with path size " << agent.second->get_past_path().size() << endl;
        cout << "==========" << endl;
    }
    }
    m->unlock(); 
    }
}

void Global::print_agent_path_single() {
    m->lock();
    ofstream sven_path;
    sven_path.open("sven_path.txt");   
    for (auto& agent : this->all_agents) {
        for (auto& pair : agent.second->get_past_path()) {
            int row = 4 - pair.second - 1 + 1;
            int col = pair.first + 1;
            int index = row * 14 + col;
            sven_path << index << " ";
        }
        sven_path << endl;
    }
    sven_path.close();
    m->unlock(); 
}

void Global::add_gridid_to_townid(int gridID, int townID) {
    this->gridid_to_townid[gridID] = townID;
}

unordered_map<int, int> Global::get_gridid_to_townid() {
    return this->gridid_to_townid;
}

int Global::get_col_dim(){
    return col_dim;
}

void Global::vis_to_json() {
    std::string file_path = "vis_export128-9/tmp/vis_agents.json";
    json j;
    this->m->lock();
    j["v"] = {};
    j["ggoal"] = {};
    j["lockin"] = {};
    j["lockout"] = {};
    j["locked"] = {};
    for (auto& a : this->all_agents) {
        int aid = a.second->get_agent_id();
        // current location
        pair<int, int> curr = a.second->get_current_grid();
        std::string s_curr = std::to_string(curr.first) + ',' + std::to_string(curr.second);
        if (curr.first >= 0 && curr.second >= 0) {
            if (j["v"].contains(s_curr)) {
                j["v"][s_curr].emplace_back(aid);
            } else {
                j["v"][s_curr] = std::vector<int>();
                j["v"][s_curr].emplace_back(aid);
            }
        }
        // global goal
        pair<int, int> goal = a.second->get_goal();
        if (goal.first >= 0 && goal.second >= 0) {
            std::string s_goal = std::to_string(goal.first) + ',' + std::to_string(goal.second);
            j["ggoal"][s_goal] = std::vector<int>();
            j["ggoal"][s_goal].emplace_back(aid);
        }

        // lock in
        if (a.second->get_lock_inT()) {
            pair<int, int> locki = a.second->get_lock_inT()->get_lock_location();
            std::string s_locki = std::to_string(locki.first) + ',' + std::to_string(locki.second);
            j["lockin"][s_locki] = std::vector<int>();
            j["lockin"][s_locki].emplace_back(aid);
        }

        // lock out
        if (a.second->get_lock_outT()) {
            pair<int, int> locko = a.second->get_lock_outT()->get_lock_location();
            std::string s_locko = std::to_string(locko.first) + ',' + std::to_string(locko.second);
            j["lockout"][s_locko] = std::vector<int>();
            j["lockout"][s_locko].emplace_back(aid);
        }
    }   
    // conflicts
    j["collision"] = {};
    for (auto& [k, v] : j["v"].items()) {
        if (v.size() > 1) {  // >= 2 agents at 1 location
            j["collision"][k] = v;
        }
    }
    j["pivots"] = {};
    j["col_v"] = std::vector<int>();

    // all locks NOT currently available
    for (auto& l : all_locks) {
        if (l.second->get_is_locked()) {
            pair<int, int> curr = l.second->get_lock_location();
            std::string s_curr = std::to_string(curr.first) + ',' + std::to_string(curr.second);
            j["locked"][s_curr] = "locked";
        }
    }

    this->m->unlock();
    std::ofstream jout(file_path);
    jout << std::setw(2) << j << std::endl;
}