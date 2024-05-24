#ifndef MESSAGE_H
#define MESSAGE_H
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include "Agent.h"
using namespace std;

enum message_type {NEW_FREE_AGENTS_TTG=0,AGENTS_IN_LOCK_INT_RTG=1,NEW_FREE_AGENTS_GTS=2,AGENTS_W_NEW_GOAL_STG=3,AGENTS_W_NEW_GOAL_LOCKS_GTT=4,AGENTS_IN_LOCK_INT_GTT=5,AGENTS_IN_LOCK_OUT_TTG=6,AGENTS_STATUS_IN_LOCK_INT_TTG=7, AGENTS_IN_LOCK_OUTT_GTR=8, AGENTS_STATUS_IN_LOCK_INT_GTR=9,AGENTS_LEAVE_LOCK_OUTT_RTG=10,R_READY=11};
// enum content_type {AGENT_LIST=0,LOCKIN_STATUS=1,NEW_TASK=2};

class Message{
    public:
        void set_message_type(int);
        int get_message_type();
        virtual string to_string();
    protected:
        int message_type;
};

class AgentListMessage: public Message{
    public:
        AgentListMessage(int message_type, vector<std::shared_ptr<Agent>> content);
        vector<std::shared_ptr<Agent>> get_content();
        void set_content(vector<std::shared_ptr<Agent>>);
        void add_content(std::shared_ptr<Agent>);
        string to_string();
    private:
        vector<std::shared_ptr<Agent>> content;  // a list of agent

};

class LockinStatusResponseMessage: public Message{
    public:
        LockinStatusResponseMessage(int message_type, unordered_map<int, bool> content);
        unordered_map<int, bool> get_content();
        void set_content(unordered_map<int, bool>);
        string to_string();
    private:
        unordered_map<int, bool> content;  // agent_id to if agent successfully taken by town in lock_inT
};

class NewTaskMessage: public Message{
    public:
        NewTaskMessage(int message_type, vector<std::shared_ptr<Agent>> content);
        vector<std::shared_ptr<Agent>> get_content();
        void set_content(vector<std::shared_ptr<Agent>>);
        void add_content(std::shared_ptr<Agent>);
        string to_string();
    private:
        vector<std::shared_ptr<Agent>> content;  // agent to new goal
};

class ReadyMessage: public Message{
    public:
        ReadyMessage(int message_type, bool content);
        bool get_content();
        void set_content(bool);
        string to_string();
    private:
        bool content;  // if road ready to accept global's message
};
#endif