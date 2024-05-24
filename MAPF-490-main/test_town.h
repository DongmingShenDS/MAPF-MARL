#include <iostream>
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <vector>
#include<unistd.h>
#include <condition_variable>
#ifndef TEST_TOWN_H
#define TEST_TOWN_H
#include "test_global.h"
using namespace std;
class Global;
class Town{
    public:
        Town(int town_id, Global* global);
        Global* get_global();
        void send_message_to_town(string message);
        // void send_message_to_global(string message);
        string wait_message_from_global();
    private:
        int town_id;
        Global* global;
        string message_from_global;
        string message_to_global;
        shared_ptr<mutex> m;
        shared_ptr<condition_variable> cv;
};
#endif