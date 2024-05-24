//
//  Vertex.cpp
//  PathFinder_WareHouseMap
//
//  Created by Tiancheng Xu on 2021/10/17.
//

#include <stdio.h>
#include "Vertex.h"

MapVertex::MapVertex(int x, int y, char ch, int town_id)
{
    this->x_ax = x;
    this->y_ax = y;
    this->cha = ch;
    this->town_id = town_id;
    if (ch == 'e' || ch == 'E')
    {
        toRight = true;
        isRoad = true;
    }
    else if (ch == 'w' || ch == 'W')
    {
        toLeft = true;
        isRoad = true;
    }
    else if (ch == 'n' || ch == 'N')
    {
        toUp = true;
        isRoad = true;
    }
    else if (ch == 's' || ch == 'S')
    {
        toDown = true;
        isRoad = true;
    }

    else if (ch == 'i' || ch == 'I')
    {
        isIntersect = true;
        isRoad = true;
    }
    else if (ch == 'L')
    {
        isLock = true;
    }
    //other chars are invalid(barriers)
    else
    {
        isRoad = false;
        this->town_id = 1;
    }
}
MapVertex::~MapVertex()
{
}

bool MapVertex::goUp()
{
    return toUp;
}

bool MapVertex::goDown()
{
    return toDown;
}

bool MapVertex::goLeft()
{
    return toLeft;
}

bool MapVertex::goRight()
{
    return toRight;
}

int MapVertex::get_x()
{
    return x_ax;
}

int MapVertex::get_y()
{
    return y_ax;
}

void MapVertex::setUp(bool b)
{
    toUp = b;
}

void MapVertex::setDown(bool b)
{
    toDown = b;
}

void MapVertex::setLeft(bool b)
{
    toLeft = b;
}

void MapVertex::setRight(bool b)
{
    toRight = b;
}

bool MapVertex::getIntersect()
{
    return isIntersect;
}
bool MapVertex::getLock()
{
    return isLock;
}

bool MapVertex::isRoadCell()
{
    return isRoad;
}

bool MapVertex::isTown()
{
    if (town_id == -1)
    {
        return false;
    }
    else
    {
        return true;
    }
}

int MapVertex::get_town_id()
{
    return town_id;
}

char MapVertex::getchar()
{
    return cha;
}
int MapVertex::get_f()
{
    return f_score;
}

void MapVertex::setf(int f)
{
    f_score = f;
}

void MapVertex::set_town_id(int town_id)
{
    this->town_id = town_id;
}