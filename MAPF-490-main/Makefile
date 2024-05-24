CXX=g++
CXXFLAGS=-g -Wall -std=c++11 -pthread
# Uncomment for parser DEBUG
#DEFS=-DDEBUG

OBJS= main.o Agent.o Global.o Lock.o Message.o Road.o ScenarioLoader.o Town.o Vertex.o warehouse_map.o
#OBJS2= mapFinder.o map_generator.o

try: main

main: $(OBJS)
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ $(OBJS)

Agent.o: Agent.cpp Agent.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c Agent.cpp

Global.o: Global.cpp Global.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c Global.cpp

Lock.o: Lock.cpp Lock.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c Lock.cpp

Message.o: Message.cpp Message.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c Message.cpp

Road.o: Road.cpp Road.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c Road.cpp

ScenarioLoader.o: ScenarioLoader.cpp ScenarioLoader.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c ScenarioLoader.cpp

Town.o: Town.cpp Town.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c Town.cpp

Vertex.o: Vertex.cpp Vertex.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c Vertex.cpp

warehouse_map.o: warehouse_map.cpp warehouse_map.h
	$(CXX) $(CXXFLAGS) $(DEFS) -o $@ -c warehouse_map.cpp


clean:
	rm -f *.o main
