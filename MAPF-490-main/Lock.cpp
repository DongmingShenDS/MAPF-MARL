#include "Lock.h"

Lock::Lock(int lock_id, pair<int, int> location, int town_id){
    this->lock_id = lock_id;
    this->location = location;
    this->town_id = town_id;
    this->is_locked = false;
}

pair<int, int> Lock::get_lock_location(){
    return location;
}

int Lock::get_lock_id(){
    return this->lock_id;
}

int Lock::get_town_id(){
    return town_id;
}

bool Lock::get_is_locked(){
    return is_locked;
}

vector<Lock*> Lock::get_reachale_locks_by_town(int town_id){
    return reachable_town_locks[town_id];
}

void Lock::set_location(pair<int, int> newloc){
    this->location = newloc;
}

void Lock::set_town_id(int tid){
    this->town_id = tid;
}

void Lock::set_reachable_town_locks(unordered_map<int, vector<Lock*>> r){
    this->reachable_town_locks = r;
}

void Lock::lock(){
    if (is_locked)
        throw std::invalid_argument("Cannot lock a Lock that is already locked");
    is_locked = true;
    return;
}

void Lock::unlock(){
    if (!is_locked)
        throw std::invalid_argument("Cannot free a Lock that is already freed");
    is_locked = false;
    return;
}
