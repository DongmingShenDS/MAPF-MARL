#include <iostream>
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <thread>
#include <mutex>
#include <vector>
#include<unistd.h>
#include <condition_variable>
#ifndef TEST_GLOBAL_H
#define TEST_GLOBAL_H
#include "test_town.h"

using namespace std;
class Town;
class Global{
    public:
        Global();
        void set_all_towns(unordered_map<int, Town*> t);
        unordered_map<int, Town*> get_all_towns();
        void send_alltowns_message(unordered_map<int, string> messages);  
        string wait_message_from_one_town();
        string wait_message_from_all_towns();
        void send_message_to_global(int, string);
    private:
        unordered_map<int, Town*> all_towns;
        queue<pair<int,string>> message_queue_from_town;
        shared_ptr<mutex> m;
        shared_ptr<condition_variable> cv;
};
#endif