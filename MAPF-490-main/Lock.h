#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <unordered_map>

#ifndef LOCK_H
#define LOCK_H
using namespace std;

class Lock{
    public:
        Lock(int lock_id, pair<int, int> location, int town_id);  // global location in the whole map
        pair<int, int> get_lock_location();
        int get_lock_id();
        int get_town_id(); 
        bool get_is_locked();
        vector<Lock*> get_reachale_locks_by_town(int town_id); 
        void set_location(pair<int, int>);
        void set_town_id(int town_id);
        void set_reachable_town_locks(unordered_map<int, vector<Lock*>> r);
        void lock();
        void unlock();
    private:
        int lock_id;
        bool is_locked;
        pair<int, int> location;
        int town_id;  // which town the lock is adjacent to
        unordered_map<int, vector<Lock*>> reachable_town_locks;
};

#endif