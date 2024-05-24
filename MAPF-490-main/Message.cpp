#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <vector>
#include <unordered_map>
#include <memory>
#include "Message.h"
using namespace std;

void Message::set_message_type(int message_type) {
    this->message_type = message_type;
}

int Message::get_message_type(){
    return this->message_type;
}

string Message::to_string(){
    return "";
}

string AgentListMessage::to_string(){
    for (auto i : content) {
        cout << "AgentListMessage" << i->get_agent_id() << endl;
    }
    return "";
}

AgentListMessage::AgentListMessage(int message_type, vector<std::shared_ptr<Agent>> content) {
    this->message_type = message_type;
    this->content = content;
}

vector<std::shared_ptr<Agent>> AgentListMessage::get_content(){
    return this->content;
}

void AgentListMessage::set_content(vector<std::shared_ptr<Agent>> v){
    this->content = v;
}

void AgentListMessage::add_content(std::shared_ptr<Agent> a){
    this->content.push_back(a);
}

string LockinStatusResponseMessage::to_string(){
    for (auto pair : content) {
        cout << pair.first << pair.second << endl;
    }
    return "";
}

LockinStatusResponseMessage::LockinStatusResponseMessage(int message_type, unordered_map<int, bool> content) {
    this->message_type = message_type;
    this->content = content;
}

unordered_map<int, bool> LockinStatusResponseMessage::get_content(){
    return this->content;
}

void LockinStatusResponseMessage::set_content(unordered_map<int, bool> um){
    this->content = um;
}

string NewTaskMessage::to_string(){
    for (auto i : content) {
        cout << "NewTaskMessage" << i->get_agent_id() << endl;
    }
    return "";
}

NewTaskMessage::NewTaskMessage(int message_type, vector<std::shared_ptr<Agent>> content) {
    this->message_type = message_type;
    this->content = content;
}

vector<std::shared_ptr<Agent>> NewTaskMessage::get_content(){
    return this->content;
}

void NewTaskMessage::set_content(vector<std::shared_ptr<Agent>> v){
    this->content = v;
}

void NewTaskMessage::add_content(std::shared_ptr<Agent> a){
    this->content.push_back(a);
}

ReadyMessage::ReadyMessage(int message_type, bool content) {
    this->message_type = message_type;
    this->content = content;
}

bool ReadyMessage::get_content(){
    return this->content;
}

void ReadyMessage::set_content(bool r){
    this->content = r;
}

string ReadyMessage::to_string(){
    return "";
}
