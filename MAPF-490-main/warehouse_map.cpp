//
//  warehouse_map.cpp
//  PathFinder_WareHouseMap
//
//  Created by Tiancheng Xu on 2021/10/17.
//

#include "warehouse_map.h"
#include <iostream>
using namespace std;

warehouse_map::warehouse_map()
{

    numcols = 0;
    numrows = 0;
}

warehouse_map::~warehouse_map()
{
}

void warehouse_map::readFile(ifstream &inputfile)
{
    int count_row = 0;
    int count_col = 0;
    string line;
    vector<string> stringlist;
    getline(inputfile,line);
    getline(inputfile,line);
    getline(inputfile,line);
    vector<int> col_size;
    int max_col = 0;
    while (getline(inputfile, line))
    {
        count_col = 0;
        count_row++;
        string temp_str = "";
        for (int i = 0; i < line.length(); i++)
        {
            if (line[i] != ' ')
            {
                temp_str += line[i];
                count_col++;
            }
        }
        line = temp_str;
        col_size.push_back(count_col);
        if (count_col > max_col)
        {
            max_col = count_col;
        }
        stringlist.push_back(line);
    }
    numcols = max_col;
    numrows = count_row;

    mymap = vector<vector<MapVertex *>>(numrows);
    for (int i = 0; i < numrows; i++)
    {
        mymap[i] = vector<MapVertex *>(numcols);
        string str = stringlist[i];
        for (int j = 0; j < numcols; j++)
        {
            char tempchar;
            int num_x = j;
            int num_y = count_row - i - 1;
            if (j < col_size[i])
            {
                tempchar = str[j];
                mymap[i][j] = new MapVertex(num_x, num_y, tempchar, -1);
            }
            else
            {
                //Automatically fulfill vertices to make the map a perfect rectangle
                //Make these missing vertices barriers
                tempchar = '@';
                mymap[i][j] = new MapVertex(num_x, num_y, tempchar, -1);
            }
        }
    }

    for (int i = 0; i < numrows; i++)
    {
        for (int j = 0; j < numcols; j++)
        {
            if (mymap[i][j]->getLock())
            {
                if (i > 0)
                {
                    if ((mymap[i - 1][j]->goDown() == false || mymap[i - 1][j]->getIntersect() == true) && !mymap[i - 1][j]->getLock() && !mymap[i - 1][j]->isTown())
                    {
                        mymap[i][j]->setUp(true);
                        mymap[i - 1][j]->setDown(true);
                    }
                    else
                    {
                        if (!mymap[i - 1][j]->getLock() && !mymap[i - 1][j]->isTown())
                        {
                            mymap[i - 1][j]->setDown(true);
                        }
                    }
                }
                if (i < (mymap.size() - 1))
                {
                    if ((mymap[i + 1][j]->goUp() == false || mymap[i + 1][j]->getIntersect() == true) && !mymap[i + 1][j]->getLock() && !mymap[i + 1][j]->isTown())
                    {
                        mymap[i][j]->setDown(true);
                        mymap[i + 1][j]->setUp(true);
                    }
                    else
                    {
                        if (!mymap[i + 1][j]->getLock() && !mymap[i + 1][j]->isTown())
                        {
                            mymap[i + 1][j]->setUp(true);
                        }
                    }
                }
                if (j > 0)
                {
                    if ((mymap[i][j - 1]->goRight() == false || mymap[i][j - 1]->getIntersect() == true) && !mymap[i][j - 1]->getLock() && !mymap[i][j - 1]->isTown())
                    {
                        mymap[i][j]->setLeft(true);
                        mymap[i][j - 1]->setRight(true);
                    }
                    else
                    {
                        if (!mymap[i][j - 1]->getLock() && !mymap[i][j - 1]->isTown())
                        {
                            mymap[i][j - 1]->setRight(true);
                        }
                    }
                }
                if (j < (mymap[0].size() - 1))
                {
                    if ((mymap[i][j + 1]->goLeft() == false || mymap[i][j + 1]->getIntersect() == true) && !mymap[i][j + 1]->getLock() && !mymap[i][j + 1]->isTown())
                    {
                        mymap[i][j]->setRight(true);
                        mymap[i][j + 1]->setLeft(true);
                    }
                    else
                    {
                        if (!mymap[i][j + 1]->getLock() && !mymap[i][j + 1]->isTown())
                        {
                            mymap[i][j + 1]->setLeft(true);
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < numrows; i++)
    {
        for (int j = 0; j < numcols; j++)
        {
            if (mymap[i][j]->getIntersect())
            {
                if(i > 0){
                    if(mymap[i-1][j]->goDown()==false){
                        if(mymap[i - 1][j]->getLock() == false && !mymap[i - 1][j]->isTown()){
                            mymap[i][j]->setUp(true);
                        }
                    }
                }
                if(i < (numrows-1)){
                    if(mymap[i+1][j]->goUp()==false){
                        if(mymap[i + 1][j]->getLock() == false && !mymap[i + 1][j]->isTown()){
                            mymap[i][j]->setDown(true);
                        }
                    }
                }
                if(j > 0){
                    if(mymap[i][j-1]->goRight()==false){
                        if(mymap[i][j-1]->getLock() == false && !mymap[i][j-1]->isTown()){
                            mymap[i][j]->setLeft(true);
                        }
                    }
                }
                if(j < (numcols-1)){
                    if(mymap[i][j+1]->goLeft()==false){
                        if(mymap[i][j+1]->getLock() == false && !mymap[i][j+1]->isTown()){
                            mymap[i][j]->setRight(true);
                        }
                    }
                }
                 
            }
        }
    }
}

vector<vector<MapVertex *>> warehouse_map::getmap()
{
    return mymap;
}

int warehouse_map::getnumofCols()
{
    return numcols;
}

int warehouse_map::getnumofRows()
{
    return numrows;
}
