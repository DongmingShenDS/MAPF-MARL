//
//  Vertex.h
//  PathFinder_WareHouseMap
//
//  Created by Tiancheng Xu on 2021/10/17.
//

#ifndef Vertex_h
#define Vertex_h
#include <iomanip>
#include <sstream>

class MapVertex
{
public:
    MapVertex(int x, int y, char ch, int town_id);
    ~MapVertex();
    bool goLeft();
    bool goRight();
    bool goUp();
    bool goDown();
    bool getIntersect();
    bool getLock();
    bool isRoadCell();
    bool isTown();
    int get_town_id();
    int get_f();
    char getchar();
    int get_x();
    int get_y();
    void setLeft(bool b);
    void setRight(bool b);
    void setUp(bool b);
    void setDown(bool b);
    void setf(int f);
    void set_town_id(int town_id);

protected:
    int x_ax;
    int y_ax;
    int town_id;
    char cha;
    int f_score;
    bool isRoad = false;
    bool isLock = false;
    bool isIntersect = false;
    bool toLeft = false;
    bool toRight = false;
    bool toUp = false;
    bool toDown = false;
};

#endif /* Vertex_h */
