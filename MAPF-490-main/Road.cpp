#include "Road.h"
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <map>
using namespace std;

struct compare
{
public:
    const bool operator()(pair<MapVertex *, int> &coord1, pair<MapVertex *, int> &coord2)
    {
        return coord1.second < coord2.second;
    }
};

struct compare1
{
public:
    bool operator()(pair<pair<int, int>, int> p1, pair<pair<int, int>, int> p2)
    {
        return p1.second < p2.second;
    }
};

//Manhattan Distance for A* Search
int MDistance(int x1, int y1, int x2, int y2)
{
    return (abs(x1 - x2) + abs(y1 - y2));
}

MapVertex *convertToMapVertex(warehouse_map wm, int x, int y)
{
    return wm.getmap()[wm.getnumofRows() - 1 - y][x];
}

pair<pair<int, int>, int> FindF(std::map<pair<int, int>, int> mp, int **f_score)
{
    pair<pair<int, int>, int> result;
    std::map<pair<int, int>, int>::iterator it;
    int f_min = INT_MAX;
    for (it = mp.begin(); it != mp.end(); it++)
    {
        if (it->second < f_min)
        {
            f_min = it->second;
            result = *it;
        }
    }
    return result;
}

Road::Road(ifstream &map)
{
    this->mywarehouse.readFile(map);
    this->m = make_shared<mutex>();
    this->cv = make_shared<condition_variable>();
    int num_row = mywarehouse.getnumofRows();
    int num_col = mywarehouse.getnumofCols();
    agent_on_road = 0;
    weight_score = new int *[num_row];
    for (int i = 0; i < num_row; i++)
    {
        weight_score[i] = new int[num_col];
    }
    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_col; j++)
        {
            //Modify to #agents - heuristic initial score?
            weight_score[i][j] = 0;
        }
    }
}

std::shared_ptr<Global> Road::get_global()
{
    return global;
}

unordered_map<int, std::shared_ptr<Agent>> Road::get_all_agents_on_road()
{
    return all_agents_on_road;
}

vector<std::shared_ptr<Agent>> Road::get_new_arrived_agents_at_road()
{
    return new_arrived_agents_at_road;
}

vector<std::shared_ptr<Agent>> Road::get_agents_leaving_lock_outT()
{
    return agents_leaving_lock_outT;
}

unordered_map<int, std::shared_ptr<Agent>> Road::get_agents_at_lock_inT()
{
    return agents_at_lock_inT;
}

warehouse_map Road::get_warehouse_map()
{
    return mywarehouse;
}
void Road::set_global(std::shared_ptr<Global> g)
{
    global = g;
}

void Road::add_all_agents_on_road(std::shared_ptr<Agent> a)
{
    all_agents_on_road.insert(make_pair(a->get_agent_id(), a));
}

void Road::set_new_arrived_agents_at_road(vector<std::shared_ptr<Agent>> v)
{
    new_arrived_agents_at_road = v;
}

void Road::set_agents_leaving_lock_outT(vector<std::shared_ptr<Agent>> v)
{
    agents_leaving_lock_outT = v;
}

void Road::add_agents_at_lock_inT(std::shared_ptr<Agent> a)
{
    agents_at_lock_inT.insert(make_pair(a->get_agent_id(), a));
}

double Road::get_road_load() {
    return all_agents_on_road.size();
}

unordered_map<int, vector<pair<int, int>>> Road::get_agents_remaining_path()
{
    return agents_remaining_path;
}

Message *Road::wait_message1_from_global()
{
    unique_lock<mutex> l(*(this->m));
    while (this->msg1_from_global_queue.empty())
    {
        // cout << "while time" << endl;
        cv->wait(l);
    }
    Message* temp = msg1_from_global_queue.front();
    msg1_from_global_queue.pop();
    return temp;
}

Message *Road::wait_message2_from_global()
{
    unique_lock<mutex> l(*(this->m));
    while (this->msg2_from_global_queue.empty())
    {
        // cout << "while time" << endl;
        cv->wait(l);
    }
    Message* temp = msg2_from_global_queue.front();
    msg2_from_global_queue.pop();
    return temp;
}

void Road::send_msg1_to_road(Message *message)
{
    m->lock();
    this->msg1_from_global_queue.push(message);
    cv->notify_all();
    m->unlock();
}

void Road::send_msg2_to_road(Message *message)
{
    m->lock();
    this->msg2_from_global_queue.push(message);
    cv->notify_all();
    m->unlock();
}

int Road::decide_priority_2(vector<int> conflict_agent)
{
    int wait_time1 = waiting_time[conflict_agent[0]];
    int wait_time2 = waiting_time[conflict_agent[1]];
    if (wait_time1 > wait_time2)
    {
        return conflict_agent[0];
    }
    else if (wait_time1 < wait_time2)
    {
        return conflict_agent[1];
    }
    else
    {
        int rad = rand() % 2;
        if (rad == 0)
        {
            return conflict_agent[0];
        }
        else
        {
            return conflict_agent[1];
        }
    }
}

int Road::decide_priority_3(vector<int> conflict_agent)
{
    int wait_time1 = waiting_time[conflict_agent[0]];
    int wait_time2 = waiting_time[conflict_agent[1]];
    int wait_time3 = waiting_time[conflict_agent[2]];
    if (wait_time1 > wait_time2)
    {
        if (wait_time1 > wait_time3)
        {
            return conflict_agent[0];
        }
        else if (wait_time1 < wait_time3)
        {
            return conflict_agent[2];
        }
        else
        {
            int rad = rand() % 2;
            if (rad == 0)
            {
                return conflict_agent[0];
            }
            else
            {
                return conflict_agent[2];
            }
        }
    }
    else if (wait_time2 > wait_time1)
    {
        if (wait_time2 > wait_time3)
        {
            return conflict_agent[1];
        }
        else if (wait_time2 < wait_time3)
        {
            return conflict_agent[2];
        }
        else
        {
            int rad = rand() % 2;
            if (rad == 0)
            {
                return conflict_agent[1];
            }
            else
            {
                return conflict_agent[2];
            }
        }
    }
    else
    {
        if (wait_time1 < wait_time3)
        {
            return conflict_agent[2];
        }
        else if (wait_time1 > wait_time3)
        {
            int rad = rand() % 2;
            if (rad == 0)
            {
                return conflict_agent[0];
            }
            else
            {
                return conflict_agent[1];
            }
        }
        else
        {
            int rad = rand() % 3;
            if (rad == 0)
            {
                return conflict_agent[0];
            }
            else if (rad == 1)
            {
                return conflict_agent[1];
            }
            else
            {
                return conflict_agent[2];
            }
        }
    }
}

void Road::StarSearch(std::shared_ptr<Agent> a)
{
    // cout << "!!!!!!!!!!!!!!!!!Ricardo finds path for agent ID " << a->get_agent_id() << endl;
    vector<vector<MapVertex *>> inputmap = mywarehouse.getmap();
    vector<MapVertex *> result_path;
    vector<pair<int, int>> path;
    bool flag = false;
    int start_x = a->get_lock_outT()->get_lock_location().first;
    int start_y = a->get_lock_outT()->get_lock_location().second;
    int end_x = a->get_lock_inT()->get_lock_location().first;
    int end_y = a->get_lock_inT()->get_lock_location().second;
    // cout << "Agent start from " << start_x << ", " << start_y << ", and its goal is " << end_x << ", " << end_y;
    int agent_id = a->get_agent_id();
    int num_row = mywarehouse.getnumofRows();
    int num_col = mywarehouse.getnumofCols();
    std::map<pair<int, int>, int> openQueue;
    vector<vector<MapVertex *>> prevlist(num_row);
    int **g_score = new int *[num_row];
    ;
    int **f_score = new int *[num_row];
    int **h_score = new int *[num_row];
    bool **inQueue = new bool *[num_row];
    //int **h_score = new int *[num_row];
    //bool **inQueue = new bool*[num_row];
    int x1 = num_row - 1 - start_y;
    int y1 = start_x;
    int goalx = num_row - 1 - end_y;
    int goaly = end_x;
    MapVertex *goal = inputmap[goalx][goaly];
    for (int i = 0; i < num_row; i++)
    {
        h_score[i] = new int[num_col];
        inQueue[i] = new bool[num_col];
        prevlist[i] = vector<MapVertex *>(num_col);
        g_score[i] = new int[num_col];
        f_score[i] = new int[num_col];
    }
    for (int i = 0; i < num_row; i++)
    {
        for (int j = 0; j < num_col; j++)
        {
            prevlist[i][j] = nullptr;
            g_score[i][j] = INT_MAX;
            f_score[i][j] = INT_MAX;
            //Modified: Initialize the heuristic
            h_score[i][j] = MDistance(i, j, goalx, goaly);
            inQueue[i][j] = false;
        }
    }
    g_score[x1][y1] = 0;
    f_score[x1][y1] = h_score[x1][y1] + 5*weight_score[x1][y1];
    inQueue[x1][y1] = true;
    openQueue.insert(pair<pair<int, int>, int>(make_pair(x1, y1), h_score[x1][y1] + 5*weight_score[x1][y1]));
    while (!openQueue.empty())
    {
        pair<pair<int, int>, int> temppair = FindF(openQueue, f_score);
        int realX = temppair.first.first;
        int realY = temppair.first.second;
        if (inputmap[realX][realY] == goal)
        {
            MapVertex *target = goal;
            int temp_row = goalx;
            int temp_col = goaly;

            //Using the vectors saving previous vertex to get the final path.
            //The path is stored in reversed order(from end point to start point)
            if (prevlist[temp_row][temp_col] != nullptr || ((temp_row == x1) && (temp_col == y1)))
            {
                while (target != nullptr)

                {
                    result_path.push_back(target);
                    path.push_back(make_pair(target->get_x(), target->get_y()));
                    temp_row = num_row - 1 - target->get_y();
                    temp_col = target->get_x();
                    target = prevlist[temp_row][temp_col];
                }
            } 
            flag = true;
            agents_remaining_path[agent_id] = path;
            m->lock();
            for (auto p : path)
            {
                weight_score[num_row - 1 - p.second][p.first] += 50;
            }
            m->unlock();
            break;
        }
        openQueue.erase(temppair.first);
        inQueue[realX][realY] = false;
        MapVertex *tempVertex = inputmap[realX][realY];
        if (tempVertex->goRight() == true && (realY < (num_col - 1)) && (inputmap[realX][realY + 1]->isTown() == false) && (inputmap[realX][realY + 1]->getLock() == false || inputmap[realX][realY + 1] == goal))
        {
            //MapVertex *neighborRight = inputmap[realX][realY + 1];
            int tempG = g_score[realX][realY] + 1;
            int fNew = tempG + h_score[realX][realY + 1] + 5*weight_score[realX][realY + 1];
            if (tempG < g_score[realX][realY + 1])
            {
                prevlist[realX][realY + 1] = tempVertex;
                g_score[realX][realY + 1] = tempG;
                f_score[realX][realY + 1] = fNew;
                if (inQueue[realX][realY + 1] == false)
                {
                    openQueue.insert(make_pair(pair<int, int>(realX, realY + 1), f_score[realX][realY + 1]));
                    inQueue[realX][realY + 1] = true;
                }
                else
                {
                    openQueue[make_pair(realX, realY + 1)] = fNew;
                }
            }
        }

        if (tempVertex->goLeft() == true && (realY > 0) && (inputmap[realX][realY - 1]->isTown() == false) && (inputmap[realX][realY - 1]->getLock() != true || inputmap[realX][realY - 1] == goal))
        {
            //MapVertex *neighborLeft = inputmap[realX][realY - 1];
            int tempG = g_score[realX][realY] + 1;
            int fNew = tempG + h_score[realX][realY - 1] + 5*weight_score[realX][realY - 1];
            if (tempG < g_score[realX][realY - 1])
            {
                prevlist[realX][realY - 1] = tempVertex;
                g_score[realX][realY - 1] = tempG;
                f_score[realX][realY - 1] = fNew;
                if (inQueue[realX][realY - 1] == false)
                {
                    openQueue.insert(make_pair(pair<int, int>(realX, realY - 1), f_score[realX][realY - 1]));
                    inQueue[realX][realY - 1] = true;
                }
                else
                {
                    openQueue[make_pair(realX, realY - 1)] = fNew;
                }
            }
        }

        if (tempVertex->goUp() == true && (realX > 0) && (inputmap[realX - 1][realY]->isTown() == false) && (inputmap[realX - 1][realY]->getLock() != true || inputmap[realX - 1][realY] == goal))
        {
            //MapVertex *neighborUp = inputmap[realX - 1][realY];
            int tempG = g_score[realX][realY] + 1;
            int fNew = tempG + h_score[realX - 1][realY] + 5*weight_score[realX - 1][realY];
            if (tempG < g_score[realX - 1][realY])
            {
                prevlist[realX - 1][realY] = tempVertex;
                g_score[realX - 1][realY] = tempG;
                f_score[realX - 1][realY] = fNew;
                //问题：根据pseudocode即便已经存在这个node了，但是他的fvalue是需要更新的
                if (inQueue[realX - 1][realY] == false)
                {
                    openQueue.insert(make_pair(pair<int, int>(realX - 1, realY), f_score[realX - 1][realY]));
                    inQueue[realX - 1][realY] = true;
                }
                else
                {
                    openQueue[make_pair(realX - 1, realY)] = fNew;
                }
            }
        }

        if (tempVertex->goDown() == true && (realX < (num_row - 1)) && (inputmap[realX + 1][realY]->isTown() == false) && (inputmap[realX + 1][realY]->getLock() != true || inputmap[realX + 1][realY] == goal))
        {
            //MapVertex *neighborDown = inputmap[realX + 1][realY];
            int tempG = g_score[realX][realY] + 1;
            int fNew = tempG + h_score[realX + 1][realY] + 5*weight_score[realX + 1][realY];
            if (tempG < g_score[realX + 1][realY])
            {
                prevlist[realX + 1][realY] = tempVertex;
                g_score[realX + 1][realY] = tempG;
                f_score[realX + 1][realY] = fNew;
                if (inQueue[realX + 1][realY] == false)
                {
                    openQueue.insert(make_pair(pair<int, int>(realX + 1, realY), f_score[realX + 1][realY]));
                    inQueue[realX + 1][realY] = true;
                }
                else
                {
                    openQueue[make_pair(realX + 1, realY)] = fNew;
                }
            }
        }
    }
    if(!flag){
        cout << "Road cannot find path for agent ID " << a->get_agent_id() << endl;
        cout << "Road cannot find path for agent ID " << a->get_agent_id() << endl;
        cout << "Road cannot find path for agent ID " << a->get_agent_id() << endl;
        cout << "Road cannot find path for agent ID " << a->get_agent_id() << endl;
        cout << "Road cannot find path for agent ID " << a->get_agent_id() << endl;
        agents_remaining_path[agent_id] = path;
    }
    for(int i= 0 ; i < path.size();i++){
        // cout << "(" << path[i].first << ", " << path[i].second << ") ";
    }
}

void Road::set_moved(unordered_map<int, MapVertex *> &thisStep, vector<pair<int, int>> &occupied, vector<int> &occupied_num, unordered_map<int, bool> &havetowait, int num1, int num2, int num3)
{
    if (num3 != 0)
    {
        pair<int, int> two = all_agents_on_road[num2]->get_current_grid();
        pair<int, int> three = all_agents_on_road[num3]->get_current_grid();
        thisStep[num2] = convertToMapVertex(mywarehouse, two.first, two.second);
        occupied.push_back(two);
        havetowait[num2] = true;
        occupied_num.push_back(num2);
        thisStep[num3] = convertToMapVertex(mywarehouse, three.first, three.second);
        occupied.push_back(three);
        havetowait[num3] = true;
        occupied_num.push_back(num3);
    }
    else
    {
        pair<int, int> two = all_agents_on_road[num2]->get_current_grid();
        thisStep[num2] = convertToMapVertex(mywarehouse, two.first, two.second);
        occupied.push_back(two);
        havetowait[num2] = true;
        occupied_num.push_back(num2);
    }
}

unordered_map<int, MapVertex *> Road::singleStepSolver(Message* msg, vector<std::shared_ptr<Agent>>& at_lock_in)
{
    srand(time(NULL));
    unordered_map<int, MapVertex *> LastStep;
    unordered_map<int, MapVertex *> thisStep;
    int numrow = mywarehouse.getnumofRows();
    // int numcol = mywarehouse.getnumofCols();
    //Dummy case: no conflict between any agent, just for test cases.
    for (auto it = all_agents_on_road.begin(); it != all_agents_on_road.end(); it++)
    {
        LastStep.insert(make_pair(it->first, mywarehouse.getmap()[numrow - 1 - it->second->get_current_grid().second][it->second->get_current_grid().first]));
    }
    for (auto it = agents_remaining_path.begin(); it != agents_remaining_path.end(); it++)
    {
        // cout << "Road Print remaining path." << endl;
        // cout << "Road Agent ID " << it->first << " has a remaining path of length " << it->second.size() << endl;

    }
    // cout << "Ricardo Road Timesstep " << timestep << " ThisStep: ";
    for (auto it = agents_remaining_path.begin(); it != agents_remaining_path.end(); it++)
    {
        // cout << "Road enter this fking loop." << endl;
        if(it->second.size()> 0){
            thisStep.insert(make_pair(it->first, convertToMapVertex(mywarehouse, it->second.back().first, it->second.back().second)));
        }   
        // cout << "(" << convertToMapVertex(mywarehouse, it->second.back().first, it->second.back().second)->get_x() << ", " << convertToMapVertex(mywarehouse, it->second.back().first, it->second.back().second)->get_y() << ") ";
    }

    //TODO: conflict solving
    unordered_map<int, bool> havetowait;
    vector<int> occupied_num;
    vector<pair<int, int>> occupied;
    std::map<pair<int, int>, vector<int>> conflicts;
    std::map<pair<int, int>, vector<int>>::iterator conflict_iter;
    for (auto it = agents_remaining_path.begin(); it != agents_remaining_path.end(); it++)
    {
        MapVertex *tempvertex;
        tempvertex = convertToMapVertex(mywarehouse, it->second.back().first, it->second.back().second);
        int x = tempvertex->get_x();
        int y = tempvertex->get_y();
        conflict_iter = conflicts.find(std::pair<int, int>(x, y));
        //if the agent is going to a coordinate that another agent is going to
        if (conflict_iter != conflicts.end())
        {
            conflicts[std::pair<int, int>(x, y)].push_back(it->first);
        }
        //an agent is going to a new coordinate
        else
        {
            vector<int> tempvec;
            tempvec.push_back(it->first);
            conflicts.insert(std::pair<pair<int, int>, vector<int>>(std::pair<int, int>(x, y), tempvec));
        }
        havetowait.insert(make_pair(it->first, false));
    }
    // if (agents_remaining_path.find(19) != agents_remaining_path.end() && agents_remaining_path.find(46) != agents_remaining_path.end())
    // {
    //     cout << "Road At Timestep " << timestep << endl;
    //     cout << "Agent 19 has moved  " << all_agents_on_road[19]->get_past_path().size() << ", and its next step is " << agents_remaining_path[19].back().first << ", " << agents_remaining_path[19].back().second << endl;
    //     cout << "Agent 46 has moved  " << all_agents_on_road[46]->get_past_path().size() << ", and its next step is " << agents_remaining_path[46].back().first << ", " << agents_remaining_path[46].back().second << endl;
    // }

    for (auto it = conflicts.begin(); it != conflicts.end(); it++)
    {
        if (it->second.size() <= 1)
        {
            continue;
        }
        // cout << "Conflict on " << it->first.first << ", " << it->first.second << " between Agents ";
        // for (int i = 0; i < it->second.size(); i++)
        // {
        //     cout << it->second[i] << " ";
        // }
        // cout << endl;
        vector<int> from_locks;
        vector<int> from_roads;
        int num1;
        int num2;
        int num3 = 0;
        for (int i = 0; i < it->second.size(); i++)
        {

            int x = all_agents_on_road[it->second[i]]->get_current_grid().first;
            int y = all_agents_on_road[it->second[i]]->get_current_grid().second;
            if (convertToMapVertex(mywarehouse, x, y)->getLock() == true)
            {
                from_locks.push_back(it->second[i]);
            }
            else if (convertToMapVertex(mywarehouse, x, y)->isRoadCell() == true)
            {
                from_roads.push_back(it->second[i]);
            }
        }
        if (from_locks.size() == 0)
        {
            if (from_roads.size() == 2)
            {
                int moving_id = decide_priority_2(from_roads);
                num1 = moving_id;
                for (int i = 0; i < 2; i++)
                {
                    if (from_roads[i] != moving_id)
                    {
                        num2 = from_roads[i];
                    }
                }
                set_moved(thisStep, occupied, occupied_num, havetowait, num1, num2, num3);
            }
            else if (from_roads.size() == 3)
            {
                int moving_id = decide_priority_3(from_roads);
                num1 = moving_id;
                int count = 0;
                for (int i = 0; i < 3; i++)
                {
                    if (from_roads[i] != moving_id)
                    {
                        if (count == 0)
                        {
                            count++;
                            num2 = from_roads[i];
                        }
                        else
                        {
                            num3 = from_roads[i];
                        }
                    }
                }
                set_moved(thisStep, occupied, occupied_num, havetowait, num1, num2, num3);
            }
        }
        else if (from_locks.size() == 1)
        {
            if (from_roads.size() == 1)
            {
                set_moved(thisStep, occupied, occupied_num, havetowait, from_roads[0], from_locks[0], 0);
            }
            else if (from_roads.size() == 2)
            {
                int moving_id = decide_priority_2(from_roads);
                num1 = moving_id;
                for (int i = 0; i < 2; i++)
                {
                    if (from_roads[i] != moving_id)
                    {
                        num2 = from_roads[i];
                    }
                }
                set_moved(thisStep, occupied, occupied_num, havetowait, num1, num2, from_locks[0]);
            }
        }
        else if (from_locks.size() == 2)
        {
            if (from_roads.size() == 1)
            {
                set_moved(thisStep, occupied, occupied_num, havetowait, from_roads[0], from_locks[0], from_locks[1]);
            }
            else if (from_roads.size() == 0)
            {
                int moving_id = decide_priority_2(from_locks);
                num1 = moving_id;
                for (int i = 0; i < 2; i++)
                {
                    if (from_locks[i] != moving_id)
                    {
                        num2 = from_locks[i];
                    }
                }
                set_moved(thisStep, occupied, occupied_num, havetowait, num1, num2, 0);
            }
        }
        else if (from_locks.size() == 3)
        {
            int moving_id = decide_priority_3(from_locks);
            num1 = moving_id;
            int count = 0;
            for (int i = 0; i < 3; i++)
            {
                if (from_locks[i] != moving_id)
                {
                    if (count == 0)
                    {
                        count++;
                        num2 = from_locks[i];
                    }
                    else
                    {
                        num3 = from_locks[i];
                    }
                }
            }
            set_moved(thisStep, occupied, occupied_num, havetowait, num1, num2, num3);
        }
    }
    // int times = 0;
    //TODO: check all agents that might be further stopped because of the agent in front of them cannot move because of conflict
    while (occupied.size() > 0)
    {
        //cout << "Occupied Size is: " << occupied.size() << endl;
        pair<int, int> coord = occupied[0];
        //considered_cell.insert(currocu);
        std::map<std::pair<int, int>, vector<int>>::iterator iter;
        //find if there is any other agent that needs to move to the occupied coordinate
        iter = conflicts.find(coord);
        int ind = occupied_num[0];
        //isvisited[ind-1] = true;
        //if there is a such agent, that agent must stop and remain at its place
        if (iter != conflicts.end())
        {
            for (int j = 0; j < iter->second.size(); j++)
            {
                if (iter->second[j] != ind)
                {
                    int index = iter->second[j];
                    havetowait[index] = true;
                    thisStep[index] = LastStep[index];
                    //if(isvisited[index-1] == false){
                    occupied.push_back(make_pair(LastStep[index]->get_x(), LastStep[index]->get_y()));
                    occupied_num.push_back(index);
                }
            }
        }
        occupied.erase(occupied.begin());
        occupied_num.erase(occupied_num.begin());
    }
    // for (auto it = all_agents_on_road.begin(); it != all_agents_on_road.end(); it++)
    // {
    //     cout << "Ricardo Timestep After conflict solving" << timestep << ": Agent ID: " << it->first << " (" <<thisStep[it->first]->get_x() << ", " << thisStep[it->first]->get_y() << ") ";
    // }

    //Post_processing:
    //Firstly, waiting for the towns to return the agents that can leave the road(from last step). If the agent can leave the road
    //the town will update its step at this time step, and the road just deletes it from agents_in_lockin.
    //Otherwise, if the agent remains in the road, the road update its position(still the same lock_in cell) at this timestep.
    //It remains in the road. Together with other newly arrived lock_in agents, send them to global.

    //Step* TODO: Receive message from global, remove the agents that leave the road. For remaining agents at lock_in, update their location
    if(!msg){
        // cout << "Road has nullptr" << endl;
    }
    Message* agent_leave_road = msg;
    vector<std::shared_ptr<Agent>> leaving = ((AgentListMessage *)(agent_leave_road))->get_content();
    if(agents_at_lock_inT.size()!=0){
        // cout << "Timestep " << timestep << ": ROAD Before at lock_inT size is " <<  agents_at_lock_inT.size() << endl;
    }
    // m->lock();
    // for(int i = 0; i < leaving.size();i++){
    //     cout << "Timestep " << timestep << " :ROAD ERASE Agents ID " << leaving[i]->get_agent_id() << " HERE Ricardo" << endl;
    //     agents_at_lock_inT.erase(leaving[i]->get_agent_id());
    // }
    // m->unlock();
    for(auto&  pair: agents_at_lock_inT){  // TODO SDM, RICARDO
        // pair.second->add_past_path(pair.second->get_current_grid());
    }
    //Add newly arrived agents to agents_at_lock_inT
    for (auto it = all_agents_on_road.begin(); it != all_agents_on_road.end(); it++)
    {
        //Agents newly arriving their lock_in, gonna send to global and then to towns. At next step, town check if they can enter the town
        if (thisStep[it->first]->get_x() == it->second->get_lock_inT()->get_lock_location().first && thisStep[it->first]->get_y() == it->second->get_lock_inT()->get_lock_location().second)
        {
            if (agents_at_lock_inT.find(it->first) == agents_at_lock_inT.end())
            {
                agent_on_road--;
                agents_at_lock_inT.insert(make_pair(it->first, it->second));
            }
        }
    }

     //Agents leaving lock_out, gonna send to global
    // for(auto it = thisStep.begin(); it!=thisStep.end();it++){
    //     cout << "Thisstep Agent ID " << it->first << " (" << it->second->get_x() << ", " << it->second->get_y() << ")" << endl;
    //     cout << "Current Grid Agent ID" << it->first << " (" << all_agents_on_road[it->first]->get_current_grid().first << ", " << all_agents_on_road[it->first]->get_current_grid().second << ")" << endl;
    // }
    for (auto it = thisStep.begin(); it != thisStep.end(); it++) 
    {
        if (all_agents_on_road[it->first]->get_lock_outT() && LastStep[it->first]->get_x() == all_agents_on_road[it->first]->get_lock_outT()->get_lock_location().first && LastStep[it->first]->get_y() == all_agents_on_road[it->first]->get_lock_outT()->get_lock_location().second) {
            if (agent_on_road > 1000) {   // max number of agents allowed on road
                it->second = LastStep[it->first];
            }
        }

    }
    for (auto it = thisStep.begin(); it != thisStep.end(); it++)
    {
        all_agents_on_road[it->first]->add_past_path(make_pair(it->second->get_x(), it->second->get_y()));
        pair<int,int> temp(it->second->get_x(), it->second->get_y());
        // cout << "Ricardo Road thisstep for agent " << to_string(it->first)  < " ("  << to_string(temp.first) << ", " << to_string(temp.second) << ") and current grid (" << to_string(all_agents_on_road[it->first]->get_current_grid().first)  << ", " << to_string(all_agents_on_road[it->first]->get_current_grid().second) << ")" << std::endl;
        if (temp != all_agents_on_road[it->first]->get_current_grid())
        {
            // cout << "Road Different from Current grid" << endl;
            int tmx = mywarehouse.getnumofRows() - 1 - all_agents_on_road[it->first]->get_current_grid().second;
            int tmy = all_agents_on_road[it->first]->get_current_grid().first;
            all_agents_on_road[it->first]->set_current_grid(temp);
            m->lock();
            weight_score[tmx][tmy] -= 50;
            m->unlock();
            agents_remaining_path[it->first].pop_back();
            waiting_time[it->first] = 0;
            if (all_agents_on_road[it->first]->get_lock_outT() && LastStep[it->first]->get_x() == all_agents_on_road[it->first]->get_lock_outT()->get_lock_location().first && LastStep[it->first]->get_y() == all_agents_on_road[it->first]->get_lock_outT()->get_lock_location().second)
            {
                agents_leaving_lock_outT.push_back(all_agents_on_road[it->first]);
                agent_on_road++;
            }
        }
        else
        {
            waiting_time[it->first]++;
        }
    }

    //Delete all agent at lock_in from road system. We don't care about their conflicts solving at next step. At next step,
    //their moves are determined by the results of towns. (Towns find it can leave, town will upate its location. Otherwise,
    //the town update its location at the same lock_in cell(as the above step* does)
    if(agents_at_lock_inT.size()!=0){
        // cout << "Timestep " << timestep << ": ROAD After lock_inT size is " <<  agents_at_lock_inT.size() << endl;
    }
    for (auto it = agents_at_lock_inT.begin(); it != agents_at_lock_inT.end(); it++)
    {
        // cout << "ROAD agents_at_lock_inT iteration" << endl;
        at_lock_in.push_back(it->second);
        if (all_agents_on_road.find(it->first) != all_agents_on_road.end())
        {
            // cout << "Erase agent " << it->first << " from road" << endl;
            all_agents_on_road.erase(it->first);
            agents_remaining_path.erase(it->first);
            waiting_time.erase(it->first);
        }
    }
    
    return thisStep;
}

void Road::run()
{
    sleep(4);
    //Receive message:
    //1. successfully leaving lock_in, delete them from all_agents_on_roads
    //2. Newly arrive lock_out, add them to newly_arrive_agents, add them to all_agents_on_roads, run path search on them. Clear
    //newly_arrive_agents. Update their remaning paths.
    // get 2 msg from global
    Message *step0_new_agent_in_lock_out = nullptr;
    Message* step0_agent_leave_road = nullptr;
    while (!step0_new_agent_in_lock_out)
    {
        // cout << timestep<< " " << "Road wait new_agent_in_lock_out from global" << endl;
        step0_new_agent_in_lock_out = wait_message1_from_global();
    }
    while (!step0_agent_leave_road)
    {
        // cout << timestep<< " " << "Road wait new_agent_in_lock_out from global" << endl;
        step0_agent_leave_road = wait_message2_from_global();
    }
    // (skipped planning) send ready
    Message* ready = new ReadyMessage(11, true);
    this->global->road_to_global(ready);
    while (true)
    {
        // get 2 msg from global
        timestep++;
        Message *new_agent_in_lock_out = nullptr;
        while (!new_agent_in_lock_out)
        {
            // cout << timestep<< " " << "Road wait new_agent_in_lock_out from global" << endl;
            new_agent_in_lock_out = wait_message1_from_global();
        }
        // cout  << timestep<< " " << "Get message 1 from global" << endl;
        Message* new_agent_leave_road = nullptr;
        while(!new_agent_leave_road){
            new_agent_leave_road = wait_message2_from_global();
        }
        vector<std::shared_ptr<Agent>> leaving = ((AgentListMessage *)(new_agent_leave_road))->get_content();
        m->lock();
        for(int i = 0; i < leaving.size();i++){
            // cout << "Timestep " << timestep << " :ROAD ERASE Agents ID " << leaving[i]->get_agent_id() << " HERE Ricardo" << endl;
            agents_at_lock_inT.erase(leaving[i]->get_agent_id());
        }
        m->unlock();
        all_agents_at_lock_in.clear();
        for (auto it = agents_at_lock_inT.begin(); it != agents_at_lock_inT.end(); it++){
            all_agents_at_lock_in.push_back(it->second);
        }
        // cout  << timestep<< " " << "Get message 2 from global" << endl;
        // m->lock();
        // if(all_agents_at_lock_in.size() > 0){
        //     // cout << timestep <<  " !!!!!Road sending message of lock in to global" << endl;
        //     // cout << "ROAD AT TIMESTEP " << timestep << " send these agents ids at lock in to global: " ;
        //     for(int i = 0; i < all_agents_at_lock_in.size();i++){
        //         cout << all_agents_at_lock_in[i]->get_agent_id() << " ";
        //     }
        //     // cout << endl;
        // }
        // m->unlock();

        // send 2 msg to global
        Message* msg2 = new AgentListMessage(1, all_agents_at_lock_in);
        Message* msg1 = new AgentListMessage(10, agents_leaving_lock_outT);
        this->global->road_to_global(msg1);
        this->global->road_to_global(msg2);
        
        this->new_arrived_agents_at_road = ((AgentListMessage *)(new_agent_in_lock_out))->get_content();
        // cout << "Road Reach Here!" << endl;
        // path finding
        for(int i = 0; i < new_arrived_agents_at_road.size();i++){
            // cout << "New Arrived Agent ID " << new_arrived_agents_at_road[i]->get_agent_id() << "(" << new_arrived_agents_at_road[i]->get_current_grid().first << ", "<< new_arrived_agents_at_road[i]->get_current_grid().second << ")" << endl;
        }
        for (int i = 0; i < new_arrived_agents_at_road.size(); i++)
        {
            if(all_agents_on_road.find(new_arrived_agents_at_road[i]->get_agent_id()) == all_agents_on_road.end()){
                all_agents_on_road.insert(make_pair(new_arrived_agents_at_road[i]->get_agent_id(), new_arrived_agents_at_road[i]));
                waiting_time.insert(make_pair(new_arrived_agents_at_road[i]->get_agent_id(), 0));
            }
            //StarSearch(new_arrived_agents_at_road[i]);
            // m->lock();
            // if(new_arrived_agents_at_road[i]->get_agent_id() == 30){

            //     cout << "Agent 30's Past Path in " << timestep << ": ";
            //     for(int j = 0; j < new_arrived_agents_at_road[i]->get_past_path().size(); j++){
            //         cout <<"(" <<  new_arrived_agents_at_road[i]->get_past_path()[j].first << ", " <<  new_arrived_agents_at_road[i]->get_past_path()[j].second << ") -> ";
                    
            //     }
            //     cout << endl;
            //     cout << "Agent 30's Planned Path in " << timestep << ": ";
            //     for(int j =  (agents_remaining_path[30].size() -1); j >= 0; j--){
            //         cout <<"(" <<  agents_remaining_path[30][j].first << ", " << agents_remaining_path[30][j].second << ") -> ";
                    
            //     }
            //     cout << endl;
            // }
            // m->unlock();
        }
        int numagent = new_arrived_agents_at_road.size();
        const int NumberOfThreads = std::thread::hardware_concurrency();
        // while (numagent != 0)
        // {
        //     std::vector<thread> threadlist;
        //     for (int j = 0; j < NumberOfThreads && numagent != 0; j++)
        //     {
        //         // threadlist.push_back(std::thread(StarSearch, new_arrived_agents_at_road[numagent - 1]));
        //         // threadlist.push_back(std::thread(StarSearch, this,new_arrived_agents_at_road[numagent - 1]));
        //         // threadlist.push_back(std::thread(StarSearch,new_arrived_agents_at_road[numagent - 1],  this,));
        //         numagent--;
        //     }
        //     for (auto &th : threadlist)
        //     {
        //         th.join();
        //     }
        // }
        while (numagent != 0)
        {
            
                StarSearch(new_arrived_agents_at_road[numagent - 1]);
                numagent--;
           
        }
        new_arrived_agents_at_road.clear();
        agents_leaving_lock_outT.clear();
        all_agents_at_lock_in.clear();


        /*Single Step Solver:
        Get the next step of each agent's remaining path(if it's at lock_in, assume its next step is still the current step), 
        solve conflicts. Check all the agents' step. If it's in lock_in and the agent is not at lock_in now, 
        add it to agents_at_lock_inT.(send to global later) Check if agent moves, compared with last step. 
        If so, update its current grid and also its remaining path(pop_back). If it's diferent, 
        check if its last step is on lock_out, if so, add it to agents_leaving_lock_out, send to global. Clear agents_leaving_lock_out
        */
        // cout << "Road Before single step solver!" << endl;
        
        singleStepSolver(new_agent_leave_road, all_agents_at_lock_in);
        // if (agents_remaining_path.find(19) != agents_remaining_path.end() && agents_remaining_path.find(46) != agents_remaining_path.end())
        // {
        //     cout << "After Single Step Solver at Timestep " << timestep << endl;
        //     cout << "Agent 19's remaining path:  ";
        //     m->lock();
        //     for (int i = (agents_remaining_path[19].size() -1); i >=0; i--)
        //     {
        //         cout << "(" << agents_remaining_path[19][i].first << ", " << agents_remaining_path[19][i].second << ") -> ";
        //     }
        //     cout << endl;
        //     m->unlock();
        //     m->lock();
        //     cout << "Agent 46's remaining path:  ";
        //     for (int i = (agents_remaining_path[46].size() -1); i >=0; i--)
        //     {
        //         cout << "(" << agents_remaining_path[46][i].first << ", " << agents_remaining_path[46][i].second << ") -> ";
        //     }
        //     cout << endl;
        //     m->unlock();
        // }
        // cout << "Road After single step solver!" << endl;
        // send ready
        Message* ready = new ReadyMessage(11, true);
        this->global->road_to_global(ready);
        // cout  << timestep<< " " << "Road end iteration" << endl;
    }


}