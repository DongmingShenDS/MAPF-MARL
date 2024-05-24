import json
import math
import os.path
import time
from copy import deepcopy
from load_global_graph import load_G
import networkx as nx
import utilities as ut
from visualize import plot_unpartitioned_G
import itertools
from scipy.spatial.distance import cityblock

"""
v3: graph edge is unweighted 
    => highly possible to find path (2 direction) with overlapping edges
v4: graph with weighted edge design 
    => problem: edge cases where limiting edges does not resolve conflicts at intersection
    => simply removing edge/adding edge cost does not resolve node conflicts
v5: weighted & directed edge design
    => when creating the original graph, each undirected edge replaced by 2 directed opposite edge (conjugate)
    => when an edge is occupied by a hw, remove its conjugate => avoid actual conflict
    => when an edge is occupied by a hw, add cost c in [0, inf) => avoid too much traffic
    => in this way, each road can have at most 1 in/out in each direction, a max of 4 in/out
v6: add file output function
v7: add "lock" function
    set lock as L
v8: what "lock"s each "lock" can reach => lock reachability
    replace end of road with X
v9: try to fix previous "lock" two problems:
    1. lock not connected any town nodes => cannot reach from town
    2. lock disconnect town nodes => town become disconnected now
v10: refactor code & add documentation up to v9
v11: modifications on making the result T-connected
    - this makes the result more possible to be T-conncected when graph is square
    1. find outer highway endpoints
    2. try to make them connected clockwise, before layout other part of highways
    * this step can be done recursively, layer by layer, possible extension... *
    Definition of T-connected: 
        for each town1-town2 pair, there is a direct highway from town1 to town2, where: 
            adj 2 town nodes (within same town) are bi-direction connected 
            adj town node & lock node are bi-direction connected
            adj 2 lock nodes are NOT connected
            adj lock node & highway node are bi-direction connected
            adj 2 highway nodes are only connected based on highway direction
v12: build town classes, check town connectivity
v13: add more constraint on picking outer highway endpoints
    1. the end point picked is as close to the ideal as possible (max is max_search_around)
    2. the end point picked MUST be connected with 4 surrounding map nodes (except edges?)
    => result T-connected maps: 
        Boston_0_256 (0), ost003d (0), den312d (0)，random-64-64-20 (0), den520d (1) 
v14: improve inner highway layout algorithm:
    1. try all adjacent (s, t), same as before 
    2. if not found path between (s, t), try: (s-, t) & (s, t+) until find both or run out of s- & t+  
v15: output update for lock_reachability => no more json, now locks.txt
v16: highway connection output => rl_edges.txt
    - previous: from previous versions, highways.txt does not show direction at interaction
    - update: rl_edges.txt contains the direction of outputs for each highway node (4 dir binary)
v17: output for visualization => vis_gmap.json 
v18: fix ideal_end_pts problem (top right corner)
    - can't simply switch between floor & ceiling, example: (7.3 will have one 7 one 8)
    - fixed by using a double to store the summation, then always cast (trim) to int
v19: fix some bugs in a1v1
"""


def largest_connected_component(original_G):
    """
    Find & Return the largest connected component in original_G as the new graph to work on
    :param original_G: original_G read from the mapf-baseline map
    :return new_G: the new_G corresponds to the largest connected component
    """
    if nx.is_connected(original_G):
        return original_G
    new_G = original_G.subgraph(max(nx.connected_components(original_G), key=len)).copy()
    assert nx.is_connected(new_G)
    return new_G


def ideal_strips(nrows, ncols, max_town_size):
    """
    Calculate the ideal number of strips (both x & y directions) based on expected max town size
    :param nrows: number of rows in the mapf-map (corresponds to horizontal highways)
    :param ncols: number of cols in the mapf-map (corresponds to vertical highways)
    :param max_town_size: maximum town size expected, should be smaller than the desired
    :return ideal_num_strip: ideal number of strips in both directions
    """
    max_size = nrows * ncols  # max possible number of nodes in G
    ideal_num_grid = max_size // max_town_size  # idea number of grids (ideally each gird surrounded by 4 hw in 4 dir)
    ideal_num_strip = math.ceil(math.sqrt(ideal_num_grid))  # idea number of highways in each direction
    ideal_num_strip = ideal_num_strip + 1 if ideal_num_strip % 2 != 0 else ideal_num_strip  # divisible by 2
    return ideal_num_strip


def ideal_endpts_dict(nrows, ncols, n_highway_hor, n_highway_ver):
    """
    Get ideal endpoints of highways, ignoring obstacles, as dict representing 2D matrix
    :param nrows: number of rows in the mapf-map (corresponds to horizontal highways)
    :param ncols: number of cols in the mapf-map (corresponds to vertical highways)
    :param n_highway_hor: number of expected horizontal highways
    :param n_highway_ver: number of expected vertical highways
    :return endpts: 2D matrox, each index corresponds to the 'ideal' highway end point location
    :return x_width: expected width of separation of horizontal highways
    :return y_width: expected width of separation of vertical highways
    """
    assert n_highway_hor > 1 and n_highway_ver > 1  # make sure there are more than 1 lanes in each direction
    assert n_highway_hor % 2 == 0 and n_highway_ver % 2 == 0  # make sure each direction has multiple of 2 lanes
    # access element by calling endpts[row][col]
    endpts = dict()
    # init the empty 2D dict with None everywhere
    for i in range((n_highway_hor + 2)):
        endpts[i] = dict()
        for j in range((n_highway_ver + 2)):
            endpts[i][j] = None  # None means no end points there
    # x's: horizontal index
    x_width = ncols // (n_highway_ver + 1)
    xs = []
    prev_add = 0
    for i in range(n_highway_ver + 2):
        if i == 0:
            xs.append(0)
        elif i == n_highway_ver + 1:
            xs.append(ncols - 1)
        else:
            # switch between floor & ceil in assigning the actual width so that in average evenly distributed
            prev_add += ncols / (n_highway_ver + 1)
            xs.append(int(prev_add))
    # y's: vertical index
    y_width = nrows // (n_highway_hor + 1)
    ys = []
    prev_add = 0
    for i in range(n_highway_hor + 2):
        if i == 0:
            ys.append(0)
        elif i == n_highway_hor + 1:
            ys.append(nrows - 1)
        else:
            # switch between floor & ceil in assigning the actual width so that in average evenly distributed
            prev_add += nrows / (n_highway_hor + 1)
            ys.append(int(prev_add))
    # plug xs, ys into the endpts dict in corresponding index
    for i in range((n_highway_hor + 2)):
        for j in range((n_highway_ver + 2)):
            endpts[i][j] = ut.Coord(xs[j], ys[i], '')
            # print("endpts i={},j={} with coord x={},y={}".format(i, j, xs[j], ys[i]))
    assert x_width > 2 and y_width > 2  # make sure towns are not too tiny
    return endpts, x_width, y_width


def actual_endpts_dict(G, ideal_endpts, max_search_around):
    """
    Get actual endpoints of highways, considering obstacles and max_search_around, as dict for 2D matrix
    :param G: the undirected graph corresponds to the input map (returned from load_G)
    :param ideal_endpts: returned from ideal_endpts_dict storing the ideal endpoints (might be invalid)
    :param max_search_around: max manhattan distance searching around when finding actual endpoints
    :return actual: 2D matrox, each index corresponds to the 'actual' highway end point location (must exist in G)
    """
    # init the actual 2D dict with Nones
    actual = dict()
    for i in range(len(ideal_endpts)):
        actual[i] = dict()
        for j in range(len(ideal_endpts[i])):
            actual[i][j] = None  # None: did not find any actual by default
    # find actual highway endpts in G for each corresponding ideal
    for i in range(len(ideal_endpts)):
        for j in range(len(ideal_endpts[i])):
            assert ideal_endpts[i][j]  # ideal should never be None
            if ideal_endpts[i][j]:
                loc = ideal_endpts[i][j]
                # NOTE: top right corner might encounter a big max_search_around (~2n) than other nodes
                for l1_dist in range(max_search_around + 1):
                    # potential: all valid index (>=0) that are EXACTLY l1_dist (manhattan) from loc
                    potential = surrounding_loc(loc, l1_dist)
                    for p_loc in potential:
                        if p_loc in G.nodes:
                            # TODO: tie breaking can be optimized here, or maybe pass the entire array?
                            p_surr = surrounding_loc(p_loc, 1)
                            surrounding_count = 0  # TODO: what is this for
                            for p_s in p_surr:
                                if p_s in G.nodes:
                                    surrounding_count += 1
                            if surrounding_count < 0:  # if p_loc connected to <4 other map nodes, ignore TODO???
                                continue
                            actual[i][j] = p_loc  # only pick p_loc if connected to =4 other map nodes
                            # print("i={},j={},actual[i][j]={}".format(i, j, p_loc))
                            break
                    else:
                        continue
                    break
    return actual


def get_outer_hw_endpts(actual_endpts):
    """
    Get the outer highway endpoints
    1. select the potential endpoints out from actual_endpts
    2. sort them in the clockwise highway order
    :param actual_endpts: 2D matrix, each index corresponds to the 'actual' highway end point location (must exist in G)
    :return final_endpts_list: a list storing index of outer highway endpoints in clockwise order
    """
    # init data structure to store the endpoints
    endpts_set = set()  # potential endpoints set
    endpts_index_set = set()  # potential endpoints set's index
    # init important constants to use
    length = len(actual_endpts)
    half_len = int((length + 1) / 2)  # >= half means above, < half means below
    # upper row
    for i in range(length):
        for j in range(length):
            if actual_endpts[j][i] is not None:
                endpts_set.add(actual_endpts[j][i])
                endpts_index_set.add((j, i))
                break
    # right col
    for i in range(length):
        for j in range(length - 1, -1, -1):
            if actual_endpts[i][j] is not None:
                endpts_set.add(actual_endpts[i][j])
                endpts_index_set.add((i, j))
                break
    # lower row
    for i in range(length):
        for j in range(length - 1, -1, -1):
            if actual_endpts[j][i] is not None:
                endpts_set.add(actual_endpts[j][i])
                endpts_index_set.add((j, i))
                break
    # left col
    for i in range(length):
        for j in range(length):
            if actual_endpts[i][j] is not None:
                endpts_set.add(actual_endpts[i][j])
                endpts_index_set.add((i, j))
                break
    # start with top-left, start index = row, col
    row, col = -1, -1
    start, current = None, None
    while col < length:
        col += 1
        row = -1
        while row < length:
            row += 1
            if (row, col) in endpts_index_set:
                start = row, col, quartile(row, col, half_len)
                current = start
                break
        else:
            continue
        break
    # construct the end points in order
    assert start is not None
    final_endpts_list = []  # storing index of endpoints
    stop = False
    while not stop:
        row, col = current[0], current[1]
        final_endpts_list.append((row, col))  # add current to the ordered index list
        endpts_index_set.remove((row, col))  # remove current from the set (avoid picking again)
        # 1st quartile: try up first, then right, finally down
        if current[2] == 1:
            # try going up (right)
            for c in range(col, length):  # col -> length: go right
                for r in range(row, -1, -1):  # row-1 -> 0: go up
                    if (r, c) in endpts_index_set:
                        row, col = r, c
                        break
                else:
                    continue
                break
            if row != current[0] or col != current[1]:  # already find the next 'current'
                current = row, col, quartile(row, col, half_len)
                stop = True if current == start else False
                continue
            # try going down, larger col has larger priority => go as right as possible
            # try until get to half-row or +3
            for r in range(row + 1, int(max(half_len, row + 4))):  # row+1 -> +4: go down
                for c in range(col, length):  # col -> len: go right (b.c. left is better)
                    if (r, c) in endpts_index_set:
                        row, col = r, c
                        break
                else:
                    continue
                break
            if row != current[0] or col != current[1]:  # already find the next 'current'
                current = row, col, quartile(row, col, half_len)
                stop = True if current == start else False
                continue
        # 2nd quartile: try left first, then up, finally right
        if current[2] == 2:
            # try going left (up)
            for r in range(row, -1, -1):  # row -> 0: go up
                for c in range(col, -1, -1):  # col-1 -> 0: go left
                    if (r, c) in endpts_index_set:
                        row, col = r, c
                        break
                else:
                    continue
                break
            if row != current[0] or col != current[1]:  # already find the next 'current'
                current = row, col, quartile(row, col, half_len)
                stop = True if current == start else False
                continue
            # try going right, smaller row has larger priority => go as up as possible
            # try until get to half-col or +3
            for c in range(col + 1, int(max(half_len, col + 4))):  # col+1 -> +4: go right
                for r in range(row, -1, -1):  # row -> 0: go up (b.c. down is better)
                    if (r, c) in endpts_index_set:
                        row, col = r, c
                        break
                else:
                    continue
                break
            if row != current[0] or col != current[1]:  # already find the next 'current'
                current = row, col, quartile(row, col, half_len)
                stop = True if current == start else False
                continue
        # 3rd quartile: try down first, then left, finally up
        if current[2] == 3:
            # try going down (left)
            for c in range(col, -1, -1):  # col -> 0: go left
                for r in range(row, length):  # row+1 -> length: go down
                    if (r, c) in endpts_index_set:
                        row, col = r, c
                        break
                else:
                    continue
                break
            if row != current[0] or col != current[1]:  # already find the next 'current'
                current = row, col, quartile(row, col, half_len)
                stop = True if current == start else False
                continue
            # try going up, smaller col has larger priority => go as left as possible
            # try until get to half-row or +3
            for r in range(row - 1, int(min(half_len - 1, row - 4)), -1):  # row-1 -> -4: go up
                for c in range(col, -1, -1):  # col -> 0: go left (b.c. right is better)
                    if (r, c) in endpts_index_set:
                        row, col = r, c
                        break
                else:
                    continue
                break
            if row != current[0] or col != current[1]:  # already find the next 'current'
                current = row, col, quartile(row, col, half_len)
                stop = True if current == start else False
                continue
        # 4th quartile: try right first, then down, finally left
        if current[2] == 4:
            # try going right (down)
            for r in range(row, length):  # row -> length: go down
                for c in range(col, length):  # col+1 -> length: go right
                    if (r, c) in endpts_index_set:
                        row, col = r, c
                        break
                else:
                    continue
                break
            if row != current[0] or col != current[1]:  # already find the next 'current'
                current = row, col, quartile(row, col, half_len)
                stop = True if current == start else False
                continue
            # try going left, larger row has larger priority => go as down as possible
            # try until get to half-col or -3
            for c in range(col - 1, int(min(half_len - 1, col - 4)), -1):  # col-1 -> -4: go left
                for r in range(row, length):  # row -> length: go down (b.c. up is better)
                    if (r, c) in endpts_index_set:
                        row, col = r, c
                        break
                else:
                    continue
                break
            if row != current[0] or col != current[1]:  # already find the next 'current'
                current = row, col, quartile(row, col, half_len)
                stop = True if current == start else False
                continue
        # still cannot find any, then pick the nearest as the next index
        min_dist = 10000
        target = (-1, -1)
        for (x, y) in endpts_index_set:
            i_dist = cityblock((x, y), (row, col))
            if i_dist < min_dist:
                min_dist = i_dist
                target = (x, y)
        row, col = target if target != (-1, -1) else (row, col)
        if row != current[0] or col != current[1]:  # already find the next 'current'
            current = row, col, quartile(row, col, half_len)
            stop = True if current == start else False
            continue
        # post-processing, determining if stop
        if len(endpts_index_set) == 0:
            print("Success")
            stop = True
            continue
        if row == current[0] and col == current[1]:
            print("Not Success")
            stop = True
            continue
    return final_endpts_list


def quartile(row, col, half):
    """
    Return quartile index of row, col based on half
    :param row: row index
    :param col: col index
    :param half: half of the length pf actual_endpts
    :return index: quartile index (same as the quartile order in math 2D grid)
    """
    if row < half and col < half:
        return 2  # left-top
    if row < half and col >= half:
        return 1  # right-top
    if row >= half and col < half:
        return 3  # left-bottom
    if row >= half and col >= half:
        return 4  # right-bottom


def actual_endpts_graph(actual_endpts):
    """
    Return graph storing prospective highways endpoints' corresponding location in actual_endpts index
    :param actual_endpts: returned from actual_endpts_dict storing the actual endpoints (valid)
    :return G_endpts_index: graph storing highways endpoints' corresponding location in actual_endpts index
    """
    # TODO: considering the new Greedy, this function might be useless
    # TODO: handle outer most highway differently?
    G_endpts_index = nx.Graph()
    for i in range(len(actual_endpts)):
        for j in range(len(actual_endpts[i])):
            if actual_endpts[i][j]:
                # only add edge between two adjacent, not none endpoints
                G_endpts_index.add_node(ut.Coord(i, j, ''), pos=(i, j))
    for u in G_endpts_index.nodes:
        for direction in ut.directions:
            v = ut.coord_add(u, ut.dir_to_vec[direction])
            if v in G_endpts_index.nodes:  # add adjacent nodes (1 away from current node)
                G_endpts_index.add_edge(u, v)
    G_endpts_index.remove_edges_from(nx.selfloop_edges(G_endpts_index))  # remove self-loops...
    return G_endpts_index


def surrounding_loc(loc, l1_dist):
    """
    Get all valid index (>=0) that are EXACTLY l1_dist from loc, as an array, in O(l1_dist)
    :param loc: the center location, the target of l1_dist
    :param l1_dist: the manhattan distance
    :return surrounding locations as an array, shape should be diamond with loc at center
    """
    new_loc = []
    if l1_dist == 0:
        new_loc = [loc]
        return new_loc
    for i in range(1, l1_dist + 1):
        j = l1_dist - i
        if i == 0 or j == 0:
            new_loc.append((loc[0] + i, loc[1] + j))
            new_loc.append((loc[0] - i, loc[1] + j))
            new_loc.append((loc[0] + j, loc[1] + i))
            new_loc.append((loc[0] + j, loc[1] - i))
        else:
            new_loc.append((loc[0] + i, loc[1] + j))
            new_loc.append((loc[0] + i, loc[1] - j))
            new_loc.append((loc[0] - i, loc[1] + j))
            new_loc.append((loc[0] - i, loc[1] - j))
    new_loc = [ut.Coord(x, y, '') for (x, y) in new_loc if x >= 0 and y >= 0]
    return new_loc


def highway_span(highway):
    """
    Get the max difference in x (horizontal span) & y (vertical span) in a highway
    :param highway: the highway to work on
    :return: x_span and y_span
    """
    max_x, min_x, max_y, min_y = -1, 100000, -1, 100000
    for node in highway:
        # simply keep track of max_x, min_x, max_y, min_y of nodes in the highway
        x, y = node[0], node[1]
        max_x = max(x, max_x)
        max_y = max(y, max_y)
        min_x = min(x, min_x)
        min_y = min(y, min_y)
    return (max_x - min_x), (max_y - min_y)


def plan_outer_highways(G_directed, final_endpts_list, actual_endpts):
    """
    Plan the outer-most highway in clockwise direction, this highway MUST be connected
    :param G_directed: the directed version of G, returned from build_directed_from_undirected
    :param final_endpts_list: the outer highway endpoints list in the correct order storing index of endpoints
    :param actual_endpts: actual endpoints of highways, considering obstacles and max_search_around, as 2D matrix
    :return all_highways: list storing all highway segments (outer)
    """
    # all_highways in format ([hw as list], hor_ver, direction) tuple
    all_highways = []
    endpts_size = len(final_endpts_list)
    has_path_from = dict()
    has_path_to = dict()
    # init a graph to keep track if highway connectivity
    outer_hw_dG = nx.DiGraph()
    for i in range(endpts_size):
        outer_hw_dG.add_node(final_endpts_list[i])
    for i in range(endpts_size):  # iterate through final_endpts_list in order to get index
        if i + 1 < endpts_size:
            i_curr, i_next = final_endpts_list[i], final_endpts_list[i + 1]
        else:
            i_curr, i_next = final_endpts_list[i], final_endpts_list[0]
        has_path_from[i_curr] = False
        has_path_to[i_next] = False
        # get actual highway endpoints for s and t for the current highway
        actual_s = actual_endpts[i_curr[0]][i_curr[1]]
        actual_t = actual_endpts[i_next[0]][i_next[1]]
        try:
            # get dijkstra_path of the highway.
            path = nx.dijkstra_path(G_directed, actual_s, actual_t)
            # append path & path info to all_highways in format ([hw as list], hor_ver, direction) tuple
            all_highways.append((path, -1, -1))  # -1, -1 indicating it is special outer highway
            # update edge properties in G_directed so later plans have access to the updated version
            for i in range(1, len(path)):
                curr_node = path[i]  # current highway node
                prev_node = path[i - 1]  # previous highway node
                highway_edge = (prev_node, curr_node)  # highway_edge (in the direction of highway)
                conj_edge = (curr_node, prev_node)  # conj_edge (in the opposite direction of highway)
                assert highway_edge in G_directed.edges  # highway_edge should be in G_directed for sure
                # remove conjugate edge in the path => avoid conflict!!!
                if conj_edge in G_directed.edges:
                    G_directed.remove_edge(conj_edge[0], conj_edge[1])
            has_path_from[i_curr] = True
            has_path_to[i_next] = True
            outer_hw_dG.add_edge(i_curr, i_next)
        except nx.NetworkXNoPath:
            _ = None
    # now deal with has_path_to[i] == False endpoints
    for i in range(endpts_size):
        if not has_path_to[final_endpts_list[i]]:  # has_path_from[i_curr] == False
            before = range(i - 2, -1, -1)
            after = range(endpts_size - 1, i + 1, -1)
            for i_p in list(before) + list(after):  # try i_next
                i_curr, i_prev = final_endpts_list[i], final_endpts_list[i_p]
                actual_s = actual_endpts[i_prev[0]][i_prev[1]]
                actual_t = actual_endpts[i_curr[0]][i_curr[1]]
                try:
                    # get dijkstra_path of the highway.
                    path = nx.dijkstra_path(G_directed, actual_s, actual_t)
                    # append path & path info to all_highways in format ([hw as list], hor_ver, direction) tuple
                    all_highways.append((path, -1, -1))  # -1, -1 indicating it is special outer highway
                    # update edge properties in G_directed so later plans have access to the updated version
                    for i in range(1, len(path)):
                        curr_node = path[i]  # current highway node
                        prev_node = path[i - 1]  # previous highway node
                        highway_edge = (prev_node, curr_node)  # highway_edge (in the direction of highway)
                        conj_edge = (curr_node, prev_node)  # conj_edge (in the opposite direction of highway)
                        assert highway_edge in G_directed.edges  # highway_edge should be in G_directed for sure
                        # remove conjugate edge in the path => avoid conflict!!!
                        if conj_edge in G_directed.edges:
                            G_directed.remove_edge(conj_edge[0], conj_edge[1])
                    has_path_from[i_prev] = True
                    has_path_to[i_curr] = True
                    outer_hw_dG.add_edge(i_prev, i_curr)
                    break
                except nx.NetworkXNoPath:
                    _ = None
    # now deal with has_path_from[i] == False endpoints
    for i in range(endpts_size):
        if not has_path_from[final_endpts_list[i]]:  # has_path_from[i_curr] == False
            after = range(i + 2, endpts_size)  # start trying from i+2 since i+1 failed
            before = range(0, i - 1)
            for i_n in list(after) + list(before):  # try i_next
                i_curr, i_next = final_endpts_list[i], final_endpts_list[i_n]
                actual_s = actual_endpts[i_curr[0]][i_curr[1]]
                actual_t = actual_endpts[i_next[0]][i_next[1]]
                try:
                    # get dijkstra_path of the highway.
                    path = nx.dijkstra_path(G_directed, actual_s, actual_t)
                    # append path & path info to all_highways in format ([hw as list], hor_ver, direction) tuple
                    all_highways.append((path, -1, -1))  # -1, -1 indicating it is special outer highway
                    # update edge properties in G_directed so later plans have access to the updated version
                    for i in range(1, len(path)):
                        curr_node = path[i]  # current highway node
                        prev_node = path[i - 1]  # previous highway node
                        highway_edge = (prev_node, curr_node)  # highway_edge (in the direction of highway)
                        conj_edge = (curr_node, prev_node)  # conj_edge (in the opposite direction of highway)
                        assert highway_edge in G_directed.edges  # highway_edge should be in G_directed for sure
                        # remove conjugate edge in the path => avoid conflict!!!
                        if conj_edge in G_directed.edges:
                            G_directed.remove_edge(conj_edge[0], conj_edge[1])
                    has_path_from[i_curr] = True
                    has_path_to[i_next] = True
                    outer_hw_dG.add_edge(i_curr, i_next)
                    break
                except nx.NetworkXNoPath:
                    _ = None
    return all_highways


def plan_highways(
        G_directed, G_endpts_index, actual_endpts, visited_add_cost, max_allowed_cost, x_width, y_width,
        restricted_box=True, r_box_factor=10
):
    """
    Main Plan Highway Layout Function
    => plan the highway on G, return a list of a bunch of list (each element: a highway segment)
    :param G_directed: the directed version of G, returned from build_directed_from_undirected
    :param G_endpts_index: graph storing prospective highways endpoints' corresponding location in actual_endpts index
    :param actual_endpts: actual endpoints of highways, considering obstacles and max_search_around, as 2D matrix
    :param visited_add_cost: cost added to an edge when a highway used it, avoid heavy traffic jam
    :param max_allowed_cost: maximum cost allowed on an edge before removing it, avoid heavy traffic jam
    :param x_width: expected width of separation of horizontal highways
    :param y_width: expected width of separation of vertical highways
    :param restricted_box: True if we want to limit the range a highway can span (non-straight)
    :param r_box_factor: how wide can a highway span in regard to its 'width' (how non-straight)
    :return all_highways: list storing all highway segments (inner)
    """
    all_highways = []
    # horizontal
    hor_ver = 1
    for sy in range(len(actual_endpts)):  # number of rows
        ty = sy
        direction = (sy + 1) % 2  # direction: alternating 0 and 1
        for sx in range(len(actual_endpts[sy]) - 1):  # number of cols - 1
            tx = sx + 1
            # get actual highway endpoints for s and t for the current highway, based on direction, order matters
            if not direction:
                temp_x, temp_y = sx, sy
                sx, sy = tx, ty
                tx, ty = temp_x, temp_y
            actual_s = actual_endpts[sy][sx]
            actual_t = actual_endpts[ty][tx]
            # only lay highway between two adjacent, not None highway endpoints
            if actual_s is None or actual_t is None:
                continue  # TODO: what about non-adjacent?
            from_s, to_t = False, False
            try:
                # get dijkstra_path of the highway. if direction=1: from s to t; if direction=0: from t to s
                path = nx.dijkstra_path(G_directed, actual_s, actual_t)
                # check if the highway is in the required box range, if not, the highway is improper, remove it
                if restricted_box and r_box_factor != 0:
                    # get the max difference in x (horizontal span) & y (vertical span) in a highway
                    max_dx, max_dy = highway_span(path)
                    # if any (x or y) span is larger than the box, remove the highway by 'continue' without adding
                    if (hor_ver == 1 and max_dx > x_width * r_box_factor) \
                            or (hor_ver == 0 and max_dy > y_width * r_box_factor):
                        print("\tno path between {} and {} in the restricted box".format(actual_s, actual_t))
                        raise nx.NetworkXNoPath('box error')
                # append path & path info to all_highways in format ([hw as list], hor_ver, direction) tuple
                all_highways.append((path, hor_ver, direction))
                # update edge properties in G_directed so later plans have access to the updated version
                for i in range(1, len(path)):
                    curr_node = path[i]  # current highway node
                    prev_node = path[i - 1]  # previous highway node
                    highway_edge = (prev_node, curr_node)  # highway_edge (in the direction of highway)
                    conj_edge = (curr_node, prev_node)  # conj_edge (in the opposite direction of highway)
                    assert highway_edge in G_directed.edges  # highway_edge should be in G_directed for sure
                    # remove conjugate edge in the path => avoid conflict!!!
                    if conj_edge in G_directed.edges:
                        G_directed.remove_edge(conj_edge[0], conj_edge[1])
                    # update edge cost in the path by add visited_add_cost => avoid traffic jam!!!
                    if visited_add_cost > 0:  # update weight
                        new_weight = G_directed.edges[highway_edge[0], highway_edge[1]]['weight'] + visited_add_cost
                        if new_weight > max_allowed_cost != 0:  # if too much traffic, remove
                            G_directed.remove_edge(highway_edge[0], highway_edge[1])
                        else:  # otherwise, add cost to weight
                            G_directed.edges[highway_edge[0], highway_edge[1]].update({'weight': new_weight})
                from_s, to_t = True, True  # if path found, should set flags True's
            except nx.NetworkXNoPath:
                print("\tno path between {} and {}".format(actual_s, actual_t))
            if not from_s and not to_t:
                if sx < tx:  # to the right
                    sx_prev = [i for i in range(sx - 1, -1, -1)]
                    tx_after = [i for i in range(tx + 1, len(actual_endpts[sy]))]
                else:  # to the left
                    sx_prev = [i for i in range(sx + 1, len(actual_endpts[sy]))]
                    tx_after = [i for i in range(tx - 1, -1, -1)]
                for sx_new in sx_prev:  # fix to_t
                    actual_s = actual_endpts[sy][sx_new]
                    actual_t = actual_endpts[ty][tx]
                    if actual_s is None or actual_t is None:
                        continue
                    try:  # same as the try above
                        path = nx.dijkstra_path(G_directed, actual_s, actual_t)
                        if restricted_box and r_box_factor != 0:
                            max_dx, max_dy = highway_span(path)
                            if (hor_ver == 1 and max_dx > x_width * r_box_factor) \
                                    or (hor_ver == 0 and max_dy > y_width * r_box_factor):
                                print("\tno path between {} and {} in the restricted box".format(actual_s, actual_t))
                                continue
                        all_highways.append((path, hor_ver, direction))
                        for i in range(1, len(path)):
                            curr_node, prev_node = path[i], path[i - 1]
                            highway_edge, conj_edge = (prev_node, curr_node), (curr_node, prev_node)
                            assert highway_edge in G_directed.edges
                            if conj_edge in G_directed.edges:
                                G_directed.remove_edge(conj_edge[0], conj_edge[1])
                            if visited_add_cost > 0:
                                new_weight = G_directed.edges[highway_edge[0], highway_edge[1]]['weight'] \
                                             + visited_add_cost
                                if new_weight > max_allowed_cost != 0:
                                    G_directed.remove_edge(highway_edge[0], highway_edge[1])
                                else:
                                    G_directed.edges[highway_edge[0], highway_edge[1]].update({'weight': new_weight})
                        print("to_t solved")
                        break  # if found path from sx_new to t, break from the loop to stop
                    except nx.NetworkXNoPath:
                        continue
                for tx_new in tx_after:  # fix from_s
                    actual_s = actual_endpts[sy][sx]
                    actual_t = actual_endpts[ty][tx_new]
                    if actual_s is None or actual_t is None:
                        continue
                    try:  # same as the try above
                        path = nx.dijkstra_path(G_directed, actual_s, actual_t)
                        if restricted_box and r_box_factor != 0:
                            max_dx, max_dy = highway_span(path)
                            if (hor_ver == 1 and max_dx > x_width * r_box_factor) \
                                    or (hor_ver == 0 and max_dy > y_width * r_box_factor):
                                print("\tno path between {} and {} in the restricted box".format(actual_s, actual_t))
                                continue
                        all_highways.append((path, hor_ver, direction))
                        for i in range(1, len(path)):
                            curr_node, prev_node = path[i], path[i - 1]
                            highway_edge, conj_edge = (prev_node, curr_node), (curr_node, prev_node)
                            assert highway_edge in G_directed.edges
                            if conj_edge in G_directed.edges:
                                G_directed.remove_edge(conj_edge[0], conj_edge[1])
                            if visited_add_cost > 0:
                                new_weight = G_directed.edges[highway_edge[0], highway_edge[1]]['weight'] \
                                             + visited_add_cost
                                if new_weight > max_allowed_cost != 0:
                                    G_directed.remove_edge(highway_edge[0], highway_edge[1])
                                else:
                                    G_directed.edges[highway_edge[0], highway_edge[1]].update({'weight': new_weight})
                        print("from_s solved")
                        break  # if found path from sx_new to t, break from the loop to stop
                    except nx.NetworkXNoPath:
                        continue
    # vertical
    hor_ver = 0
    for sx in range(len(actual_endpts[0])):  # number of cols
        tx = sx
        direction = sx % 2  # direction: alternating 0 and 1
        for sy in range(len(actual_endpts) - 1):  # number of rows - 1
            ty = sy + 1
            # get actual highway endpoints for s and t for the current highway, based on direction, order matters
            if not direction:
                temp_x, temp_y = sx, sy
                sx, sy = tx, ty
                tx, ty = temp_x, temp_y
            actual_s = actual_endpts[sy][sx]
            actual_t = actual_endpts[ty][tx]
            # only lay highway between two adjacent, not None highway endpoints
            if actual_s is None or actual_t is None:
                continue  # TODO: what about non-adjacent?
            from_s, to_t = False, False
            try:
                # get dijkstra_path of the highway. if direction=1: from s to t; if direction=0: from t to s
                path = nx.dijkstra_path(G_directed, actual_s, actual_t)
                # check if the highway is in the required box range, if not, the highway is improper, remove it
                if restricted_box and r_box_factor != 0:
                    # get the max difference in x (horizontal span) & y (vertical span) in a highway
                    max_dx, max_dy = highway_span(path)
                    # if any (x or y) span is larger than the box, remove the highway by 'continue' without adding
                    if (hor_ver == 1 and max_dx > x_width * r_box_factor) \
                            or (hor_ver == 0 and max_dy > y_width * r_box_factor):
                        print("\tno path between {} and {} in the restricted box".format(actual_s, actual_t))
                        raise nx.NetworkXNoPath('box error')
                # append path & path info to all_highways in format ([hw as list], hor_ver, direction) tuple
                all_highways.append((path, hor_ver, direction))
                # update edge properties in G_directed so later plans have access to the updated version
                for i in range(1, len(path)):
                    curr_node = path[i]  # current highway node
                    prev_node = path[i - 1]  # previous highway node
                    highway_edge = (prev_node, curr_node)  # highway_edge (in the direction of highway)
                    conj_edge = (curr_node, prev_node)  # conj_edge (in the opposite direction of highway)
                    assert highway_edge in G_directed.edges  # highway_edge should be in G_directed for sure
                    # remove conjugate edge in the path => avoid conflict!!!
                    if conj_edge in G_directed.edges:
                        G_directed.remove_edge(conj_edge[0], conj_edge[1])
                    # update edge cost in the path by add visited_add_cost => avoid traffic jam!!!
                    if visited_add_cost > 0:  # update weight
                        new_weight = G_directed.edges[highway_edge[0], highway_edge[1]]['weight'] + visited_add_cost
                        if new_weight > max_allowed_cost != 0:  # if too much traffic, remove
                            G_directed.remove_edge(highway_edge[0], highway_edge[1])
                        else:  # otherwise, add cost to weight
                            G_directed.edges[highway_edge[0], highway_edge[1]].update({'weight': new_weight})
                from_s, to_t = True, True  # if path found, should set flags True's
            except nx.NetworkXNoPath:
                print("\tno path between {} and {}".format(actual_s, actual_t))
            if not from_s and not to_t:
                if sx < tx:  # to the down
                    sx_prev = [i for i in range(sx - 1, -1, -1)]
                    tx_after = [i for i in range(tx + 1, len(actual_endpts[sy]))]
                else:  # to the up
                    sx_prev = [i for i in range(sx + 1, len(actual_endpts[sy]))]
                    tx_after = [i for i in range(tx - 1, -1, -1)]
                for sx_new in sx_prev:  # fix to_t
                    actual_s = actual_endpts[sy][sx_new]
                    actual_t = actual_endpts[ty][tx]
                    if actual_s is None or actual_t is None:
                        continue
                    try:  # same as the try above
                        path = nx.dijkstra_path(G_directed, actual_s, actual_t)
                        if restricted_box and r_box_factor != 0:
                            max_dx, max_dy = highway_span(path)
                            if (hor_ver == 1 and max_dx > x_width * r_box_factor) \
                                    or (hor_ver == 0 and max_dy > y_width * r_box_factor):
                                print("\tno path between {} and {} in the restricted box".format(actual_s, actual_t))
                                continue
                        all_highways.append((path, hor_ver, direction))
                        for i in range(1, len(path)):
                            curr_node, prev_node = path[i], path[i - 1]
                            highway_edge, conj_edge = (prev_node, curr_node), (curr_node, prev_node)
                            assert highway_edge in G_directed.edges
                            if conj_edge in G_directed.edges:
                                G_directed.remove_edge(conj_edge[0], conj_edge[1])
                            if visited_add_cost > 0:
                                new_weight = G_directed.edges[highway_edge[0], highway_edge[1]]['weight'] \
                                             + visited_add_cost
                                if new_weight > max_allowed_cost != 0:
                                    G_directed.remove_edge(highway_edge[0], highway_edge[1])
                                else:
                                    G_directed.edges[highway_edge[0], highway_edge[1]].update({'weight': new_weight})
                        print("to_t solved")
                        break  # if found path from sx_new to t, break from the loop to stop
                    except nx.NetworkXNoPath:
                        continue
                for tx_new in tx_after:  # fix from_s
                    actual_s = actual_endpts[sy][sx]
                    actual_t = actual_endpts[ty][tx_new]
                    if actual_s is None or actual_t is None:
                        continue
                    try:  # same as the try above
                        path = nx.dijkstra_path(G_directed, actual_s, actual_t)
                        if restricted_box and r_box_factor != 0:
                            max_dx, max_dy = highway_span(path)
                            if (hor_ver == 1 and max_dx > x_width * r_box_factor) \
                                    or (hor_ver == 0 and max_dy > y_width * r_box_factor):
                                print("\tno path between {} and {} in the restricted box".format(actual_s, actual_t))
                                continue
                        all_highways.append((path, hor_ver, direction))
                        for i in range(1, len(path)):
                            curr_node, prev_node = path[i], path[i - 1]
                            highway_edge, conj_edge = (prev_node, curr_node), (curr_node, prev_node)
                            assert highway_edge in G_directed.edges
                            if conj_edge in G_directed.edges:
                                G_directed.remove_edge(conj_edge[0], conj_edge[1])
                            if visited_add_cost > 0:
                                new_weight = G_directed.edges[highway_edge[0], highway_edge[1]]['weight'] \
                                             + visited_add_cost
                                if new_weight > max_allowed_cost != 0:
                                    G_directed.remove_edge(highway_edge[0], highway_edge[1])
                                else:
                                    G_directed.edges[highway_edge[0], highway_edge[1]].update({'weight': new_weight})
                        print("from_s solved")
                        break  # if found path from sx_new to t, break from the loop to stop
                    except nx.NetworkXNoPath:
                        continue
    return all_highways


def build_highway_on_empty(all_highways):
    """
    Build highway from all_highways on a new empty graph, only storing highway information (node, direction, etc.)
    :param all_highways: all highways in a list returned from plan_highways
    :return G_hw: new graph with ONLY these highways, node in format ut.HighwayCoord
    """
    G_hw = nx.Graph()
    # each highway segment in all_highways is in format ([hw as list], hor_ver, direction)
    for highway, hor_ver, direction in all_highways:
        assert len(highway) > 1
        prev_dir = ''  # store direction of the previous node (use for when reach the end)
        # add highway nodes to G_hw
        for i in range(len(highway)):
            # get direction depending on current node & next node on the highway (4 cases)
            index, index_next = i, i + 1
            node = highway[index]
            x_curr, y_curr = node[0], node[1]
            next_node = highway[index_next] if -len(highway) < index_next < len(highway) else None
            if next_node:  # when next node exists: get direction of node based on next_node
                dir = ''
                x_next, y_next = next_node[0], next_node[1]
                # 4 kinds of relation between 2 connected grid node
                if x_curr == x_next and y_next == y_curr + 1:  # up/north
                    dir = 'u'
                elif x_curr == x_next and y_next == y_curr - 1:  # down/south
                    dir = 'd'
                elif y_next == y_curr and x_next == x_curr + 1:  # right/east
                    dir = 'r'
                elif y_next == y_curr and x_next == x_curr - 1:  # left/west
                    dir = 'l'
                else:
                    assert dir != ''  # next node exist, direction must be one of the four, must not be ''
            else:  # next node does not exist => end of hw => put direction as X for placeholder
                dir = 'X'.upper()
            prev_dir = dir
            # init a temp highway node with x, y index, only to check if G_hw contain this node already
            temp_hw_node = ut.HighwayCoord(x_curr, y_curr, '')
            if temp_hw_node in G_hw:
                # if the node is already in the G_hw, get new_label = append dir to the original dir
                original_node = G_hw.nodes[temp_hw_node]
                original_label = original_node['label']
                new_label = original_label + dir
                G_hw.remove_node(temp_hw_node)  # remove the original node
            else:
                # if the node not in G_hw yet, simply new_label = dir
                new_label = dir
            # add the new node to G_hw, with new_label as the direction label (possibly contain multiple directions)
            new_node = ut.HighwayCoord(x_curr, y_curr, new_label)
            G_hw.add_node(new_node, pos=(x_curr, y_curr), label=new_label)
        # add highway edges to G_hw (undirected)
        for i in range(len(highway)):
            index, index_next = i, i + 1
            node = highway[index]
            x_curr, y_curr = node[0], node[1]
            next_node = highway[index_next] if -len(highway) < index_next < len(highway) else None
            if next_node:  # when next node exists
                x_next, y_next = next_node[0], next_node[1]
                n_hw_curr = ut.HighwayCoord(x_curr, y_curr, '')
                n_hw_next = ut.HighwayCoord(x_next, y_next, '')
                G_hw.add_edge(n_hw_curr, n_hw_next)
            # TODO: when next node does not exist, should still add edge based on direction, otherwise miss end edge
    return G_hw


def highway_post_processing(G_original, G_highway_only, all_highways):
    """
    Return several graphs storing different graphs for later use
    :param G_original: (ut.Coord) the undirected original graph returned from load_G
    :param G_highway_only: (ut.HighwayCoord) highway graph G_hw returned from build_highway_on_empty
    :param all_highways: all highways as list returned from plan_highways => [([hw as list], hor_ver, direction)]
    :return G_together: (ut.HighwayCoord) containing highway nodes, town nodes, temp lock nodes
    :return G_town_only: (ut.Coord) containing only non-highway nodes (possibly town & locks, no distinction)
    :return G_highway_only: (ut.HighwayCoord) same G_highway_only as passed in, no change from build_highway_on_empty
    :return G_temp_locks: (ut.HighwayCoord) containing only temp lock nodes with no edges
    :return G_final_town: (ut.HighwayCoord) containing all non-highway nodes (possibly town & locks, no distinction)
    """
    all_actual_highways = [hw[0] for hw in all_highways]  # actual highway locations list
    all_hw_nodes = [*set(itertools.chain(*all_actual_highways))]  # all highway nodes, no repeat
    # remove all hw_nodes from G_original => G_original now becomes town nodes only
    for hw_node in all_hw_nodes:
        G_original.remove_node(hw_node)
    # init G_town_only, containing only town nodes (ut.Coord)
    G_town_only = deepcopy(G_original)
    # init G_final_town as empty graph (later should contain towns & temporary locks)
    G_final_town = nx.Graph()
    # init G_together from G_highway_only, containing only highway nodes (ut.HighwayCoord)
    G_together = deepcopy(G_highway_only)
    # init G_temp_locks as empty graph (later should contain temporary locks only)
    G_temp_locks = nx.Graph()
    for town_node in G_town_only.nodes:  # iterate each town node (ut.Coord) in G_town_only
        tx, ty = town_node[0], town_node[1]  # town x, y index
        # surroundings: 4 surrounding nodes of town_node (ut.HighwayCoord no label)
        surrounding1 = ut.HighwayCoord(tx + 1, ty, '')
        surrounding2 = ut.HighwayCoord(tx, ty + 1, '')
        surrounding3 = ut.HighwayCoord(tx - 1, ty, '')
        surrounding4 = ut.HighwayCoord(tx, ty - 1, '')
        # if any surroundings is also in G_highway_only (a highway nodes) => mark town_node as temp 'lock'
        if surrounding1 in G_highway_only or surrounding2 in G_highway_only \
                or surrounding3 in G_highway_only or surrounding4 in G_highway_only:
            # add it to G_together & G_temp_locks as a temp lock
            G_together.add_node(ut.HighwayCoord(tx, ty, 'lock'), pos=(tx, ty), label='lock')
            G_temp_locks.add_node(ut.HighwayCoord(tx, ty, 'lock'), pos=(tx, ty), label='lock')
        # otherwise, keep town_node as a normal town node with '.'
        else:
            # add it to G_together as a town
            G_together.add_node(ut.HighwayCoord(tx, ty, '.'), pos=(tx, ty), label='.')
        # either way, add town_node to G_final_town (ut.HighwayCoord label '.')
        G_final_town.add_node(ut.HighwayCoord(tx, ty, '.'), pos=(tx, ty), label='.')
    # add all edges between towns in G_together and G_final_town
    for tn_u, tn_v in G_town_only.edges:
        tn_u_x, tn_u_y = tn_u[0], tn_u[1]
        tn_v_x, tn_v_y = tn_v[0], tn_v[1]
        u_node, v_node = ut.HighwayCoord(tn_u_x, tn_u_y, '.'), ut.HighwayCoord(tn_v_x, tn_v_y, '.')
        G_together.add_edge(u_node, v_node)
        G_final_town.add_edge(u_node, v_node)
    return G_together, G_town_only, G_highway_only, G_temp_locks, G_final_town


def get_valid_locks(G_together, G_temp_locks, G_final_town):
    """
    Process temp_locks and get the actual valid locks (since many are invalid and leads to possible errors)
    :param G_together: (ut.HighwayCoord) containing highway nodes, town nodes, temp lock nodes
    :param G_temp_locks: (ut.HighwayCoord) containing only temp lock nodes with no edges
    :param G_final_town: (ut.HighwayCoord) containing all non-highway nodes (possibly town & locks, no distinction)
    :return G_locks: (ut.HighwayCoord) contain the finalized, valid, true locks (next to town & highway)
    :return G_together_new: (ut.HighwayCoord): updated, containing highway nodes, town nodes, valid lock nodes
    """
    # init G_locks as empty graph (later should contain valid locks only)
    G_locks = nx.Graph()
    # init G_together_new from G_together (containing all highway nodes, town nodes, temp lock nodes)
    G_together_new = deepcopy(G_together)
    # separate original towns as a list (towns are connected components in G_final_town, should include temp locks)
    towns = [G_final_town.subgraph(c).copy() for c in nx.connected_components(G_final_town)]
    for town_and_lock in towns:  # iterate each town in all towns
        # town_temp_locks: temp locks (from G_temp_locks) in the town (town_and_lock) as a list
        town_temp_locks = [t_node for t_node in town_and_lock.nodes if t_node in G_temp_locks.nodes]
        town_temp_locks_copy = deepcopy(town_temp_locks)  # init town_temp_locks_copy (to keep a copy of locks)
        # town_without_lock: from town_and_lock remove temp locks => town nodes only!
        town_without_lock = deepcopy(town_and_lock)
        for l in town_temp_locks:
            town_without_lock.remove_node(l)
        # 1. Greedy fix for problem type 1: lock not connected any town nodes => cannot reach from town
        for l in town_temp_locks_copy:  # iterate each temp lock l (ut.HighwayCoord)
            if l not in town_temp_locks:  # make sure this temp lock is still in town_temp_locks (keep updating)
                continue  # otherwise, l is already not a lock (so a normal town), skip
            x, y, label = l.x, l.y, l.label
            # get 4 surrounding nodes of l (ut.HighwayCoord no label), combine them along with l as locks_surr list
            l_sur1 = ut.HighwayCoord(x + 1, y, '')
            l_sur2 = ut.HighwayCoord(x, y + 1, '')
            l_sur3 = ut.HighwayCoord(x - 1, y, '')
            l_sur4 = ut.HighwayCoord(x, y - 1, '')
            locks_surr = [l, l_sur1, l_sur2, l_sur3, l_sur4]
            # if l is NOT next to any town node (aka none of the 4 surrounding nodes are in town_without_lock)
            if not (l_sur1 in town_without_lock or l_sur2 in town_without_lock
                    or l_sur3 in town_without_lock or l_sur4 in town_without_lock):
                # then, there is no 1-step move from any town node to l, this is not allowed
                for ls in locks_surr:
                    if ls in town_temp_locks:
                        town_temp_locks.remove(ls)  # remove all temp locks (in locks_surr) from town_temp_locks
                        town_without_lock.add_node(ls)  # add them back to town_without_lock, become normal town node
        # update town_temp_locks_copy to current town_temp_locks, now problem 1 is fixed already
        town_temp_locks_copy = deepcopy(town_temp_locks)
        # 2. Greedy fix for problem type 2: lock disconnect town nodes => town become disconnected by locks
        for l in town_temp_locks_copy:  # iterate each temp lock l
            if l not in town_temp_locks:  # make sure this temp lock is still in town_temp_locks (keep updating)
                continue  # otherwise, l is already not a lock (so a normal town), skip
            # tmp_town_and_lock: all town & lock nodes in the connected component of this original town
            tmp_town_and_lock = deepcopy(town_and_lock)
            assert nx.is_connected(tmp_town_and_lock)  # should always be connected
            tmp_town_and_lock.remove_node(l)  # remove temp lock l from tmp_town_and_lock (try and see)
            if not nx.is_connected(tmp_town_and_lock):
                # if tmp_town_and_lock become disconnected, l is not valid, remove l from town_temp_locks
                town_temp_locks.remove(l)
            else:  # otherwise, update town_and_lock by actually removing l
                town_and_lock = deepcopy(tmp_town_and_lock)
        # town_temp_locks now contains locks that are valid (does not bring problems)
        for l in town_temp_locks:
            x, y, label = l.x, l.y, l.label
            G_together_new.remove_node(l)
            # add the finalized valid lock (as 'LOCK') to G_locks, G_together, G_together_new
            G_together_new.add_node(ut.HighwayCoord(x, y, 'LOCK'), pos=(x, y), label='LOCK')
            # TODO: why update G_together here? why cannot simply use G_together_new later?
            G_together.add_node(ut.HighwayCoord(x, y, 'LOCK'), pos=(x, y), label='LOCK')
            G_locks.add_node(ut.HighwayCoord(x, y, 'LOCK'), pos=(x, y), label='LOCK')
    return G_locks, G_together_new


def color_graph(G):
    """
    Get a color_map used in visualization corresponds to G
    :param G: final graph G with node labeled by (town, u, d, l, r)
    :return c_map: color_map for G
    """
    c_map = ['' for _ in G]
    label_map = dict()
    for i, node in enumerate(G):
        # label: len-1 string returned from parsing G_together_new node label
        label = deal_with_hw_label(node.label)
        # assign color based on node type
        if label == 'l':  # blue => left/west
            label_map[i] = 'Highway-west'
            c_map[i] = 'blue'
        elif label == 'r':  # green => right/east
            label_map[i] = 'Highway-east'
            c_map[i] = 'green'
        elif label == 'u':  # orange => up/north
            label_map[i] = 'Highway-north'
            c_map[i] = 'orange'
        elif label == 'd':  # purple => down/south
            label_map[i] = 'Highway-south'
            c_map[i] = 'red'
        elif label == '.':  # lightgrey => town
            label_map[i] = 'Town'
            c_map[i] = 'lightgrey'
        elif len(label) > 3:  # lightcoral => lock
            label_map[i] = 'Lock'
            c_map[i] = 'lightcoral'
        else:  # black => intersection out (a highway node with multiple directions out)
            label_map[i] = 'Highway-cross'
            c_map[i] = 'black'
    return c_map, label_map


def node_size_graph(G, town_size=0.5, road_size=10, lock_size=5):
    """
    Get a size_map used in visualization corresponds to G
    :param G: final graph G with node labeled by (town, u, d, l, r)
    :param town_size: town node size
    :param road_size: highway & buffer node size
    :return size_map: size_map for G
    """
    # if town then town_size, otherwise (highway & buffer) road_size
    size_map = [town_size for _ in G]
    for i, node in enumerate(G):
        if node.label == '.':
            size_map[i] = town_size
        elif node.label.lower() == 'lock':
            size_map[i] = lock_size
        else:
            size_map[i] = road_size
    return size_map


def build_directed_from_undirected(G_undirected):
    """
    Return a directed version of G, used for highway planning
    :param G_undirected: the undirected graph corresponds to the input map (returned from load_G)
    :return G_directed: a directed version of G_undirected, each edge replaced by TWO edges in opposite direction
    """
    G_directed = nx.DiGraph()  # directed graph
    for n in G_undirected.nodes:
        G_directed.add_node(n)
    for (u, v) in G_undirected.edges:  # for each add, add two opposite direction edges
        cost = G_undirected.edges[u, v]['weight']
        G_directed.add_edge(u, v, weight=cost)
        G_directed.add_edge(v, u, weight=cost)
    return G_directed


def bbox_graph(G, nrows, ncols):
    """
    Get bounding box of graph G (a segment of G, usually town)
    :param G: a town graph
    :param nrows: number of total rows in the original graph
    :param ncols: number of total cols in the original graph
    :return min_x, min_y, max_x, max_y: bounding box of G
    """
    # return bbox 4 number (x_min, y_min, x_max, y_max)
    max_x, min_x, max_y, min_y = -1, ncols + 1, -1, nrows + 1
    for node in G:
        x, y = node[0], node[1]
        max_x = max(x, max_x)
        max_y = max(y, max_y)
        min_x = min(x, min_x)
        min_y = min(y, min_y)
    return min_x, min_y, max_x, max_y


def deal_with_hw_label(label):
    """
    Parse highway label to remove redundant information and return a shorter and simpler version containing same info
    :param label: original label from highway nodes
    :return new_label: new_label from the original label
    """
    if label == 'LOCK':  # if 'LOCK', it is valid lock, keep the label 'LOCK' and return
        return label
    if label == 'lock':  # if 'lock', it is temp_lock that are discarded, make it back to town '.' and return
        return '.'
    label = ''.join(set(label))  # remove duplicates (mainly about duplicating directions)
    # TODO: this is useless after v8 I think?
    # contain highway ends: currently end direction equals prev grid direction, capitalized
    if 'L' in label or 'R' in label or 'U' in label or 'D' in label:
        # if contain any non-end elements & end element: non-end direction dominates end direction
        if len(label) > 1 and ('l' in label or 'r' in label or 'u' in label or 'd' in label):
            label = label.replace('L', '').replace('R', '').replace('U', '').replace('D', '')  # remove all ends
        else:  # otherwise, just use the original end direction, cast to lower
            label = label.lower()
    # remove duplicates again
    return ''.join(set(label))


def get_output_hw_label(label):
    """
    Parse node label to return a single char containing same information, used for highway map output
    :param label: original label from (highway, town, lock) nodes
    :return new_label: new_label from the original label
    """
    if label == 'LOCK':  # if LOCK, use L
        return 'L'
    elif label == 'l':  # if l (left), use w (west)
        return 'w'
    elif label == 'r':  # if r (right), use e (east)
        return 'e'
    elif label == 'u':  # if u (up), use n (north)
        return 'n'
    elif label == 'd':  # if d (down), use s (south)
        return 's'
    elif label == '.':  # if ., use @ => highway map treats town nodes as obstacles
        return '@'
    elif label == 'X':  # if X, keep X => representing highway ends
        return 'X'
    else:  # otherwise, should be intersection (with len>1 since multiple out directions), use I (intersection)
        assert len(label) > 1
        return 'I'


def get_output_global_label(label):
    """
    Parse node label to return a single char containing same information, used for global map json output
    :param label: original label from (highway, town, lock) nodes
    :return new_label: new_label from the original label
    """
    if label == 'LOCK':  # if LOCK, use L
        return 'L'
    elif label == 'l':  # if l (left), use w (west)
        return 'w'
    elif label == 'r':  # if r (right), use e (east)
        return 'e'
    elif label == 'u':  # if u (up), use n (north)
        return 'n'
    elif label == 'd':  # if d (down), use s (south)
        return 's'
    elif label == '.':  # if ., use . => global map treats town nodes as .
        return '`'
    elif label == 'X':  # if X, keep X => representing highway ends
        return 'X'
    else:  # otherwise, should be intersection (with len>1 since multiple out directions), use I (intersection)
        assert len(label) > 1
        return 'I'


def highway_lock_digraph(G_highway_only, G_valid_locks):
    """
    Build a directed graph with only highway nodes and lock nodes. Used to check which locks each lock can get to
    :param G_highway_only: (ut.HighwayCoord) highway graph G_hw returned from build_highway_on_empty
    :param G_valid_locks: (ut.HighwayCoord) contain the finalized, valid, true locks (next to town & highway)
    :return G_lock_highway: (ut.HighwayCoord) directed graph with only highway nodes and lock nodes, has edges
    """
    G_lock_highway = nx.DiGraph()  # init G_lock_highway as an empty directed graph
    # iterate each (highway) node in G_highway_only, add edges between highway nodes based on directions
    for node in G_highway_only.nodes:
        x, y, label = node.x, node.y, node.label.lower()
        # use 4 parallel if instead of if-else because one highway node can have multiple out directions (intersection)
        if 'u' in label:  # up/north
            # n: next node (should be in G_highway_only also) on the north (x, y+1) of current node (x, y)
            n = ut.HighwayCoord(x, y + 1, G_highway_only.nodes[ut.HighwayCoord(x, y + 1, '')]['label'])
            assert n in G_highway_only.nodes
            G_lock_highway.add_edge(node, n)  # add edge node->n
        if 'd' in label:  # down/south
            # n: next node (should be in G_highway_only also) on the south (x, y-1) of current node (x, y)
            n = ut.HighwayCoord(x, y - 1, G_highway_only.nodes[ut.HighwayCoord(x, y - 1, '')]['label'])
            assert n in G_highway_only.nodes
            G_lock_highway.add_edge(node, n)  # add edge node->n
        if 'r' in label:  # right/east
            # n: next node (should be in G_highway_only also) on the east (x+1, y) of current node (x, y)
            n = ut.HighwayCoord(x + 1, y, G_highway_only.nodes[ut.HighwayCoord(x + 1, y, '')]['label'])
            assert n in G_highway_only.nodes
            G_lock_highway.add_edge(node, n)  # add edge node->n
        if 'l' in label:  # left/west
            # n: next node (should be in G_highway_only also) on the west (x-1, y) of current node (x, y)
            n = ut.HighwayCoord(x - 1, y, G_highway_only.nodes[ut.HighwayCoord(x - 1, y, '')]['label'])
            assert n in G_highway_only.nodes
            G_lock_highway.add_edge(node, n)  # add edge node->n
    # iterate each (valid lock) node in G_valid_locks, add bi-direction edges between lock & highway nodes (in & out)
    for node in G_valid_locks.nodes:
        x, y, label = node.x, node.y, node.label.lower()
        # init 4 surrounding nodes of the valid lock
        surrounding1 = ut.HighwayCoord(x + 1, y, '')
        surrounding2 = ut.HighwayCoord(x, y + 1, '')
        surrounding3 = ut.HighwayCoord(x - 1, y, '')
        surrounding4 = ut.HighwayCoord(x, y - 1, '')
        # for each surrounding, if it is a highway node, add edges node->n and n->node
        if surrounding1 in G_highway_only.nodes:
            n = ut.HighwayCoord(x + 1, y, G_highway_only.nodes[surrounding1]['label'])
            G_lock_highway.add_edge(node, n)
            G_lock_highway.add_edge(n, node)
        if surrounding2 in G_highway_only.nodes:
            n = ut.HighwayCoord(x, y + 1, G_highway_only.nodes[surrounding2]['label'])
            G_lock_highway.add_edge(node, n)
            G_lock_highway.add_edge(n, node)
        if surrounding3 in G_highway_only.nodes:
            n = ut.HighwayCoord(x - 1, y, G_highway_only.nodes[surrounding3]['label'])
            G_lock_highway.add_edge(node, n)
            G_lock_highway.add_edge(n, node)
        if surrounding4 in G_highway_only.nodes:
            n = ut.HighwayCoord(x, y - 1, G_highway_only.nodes[surrounding4]['label'])
            G_lock_highway.add_edge(node, n)
            G_lock_highway.add_edge(n, node)
    return G_lock_highway


# def get_lock_reachability(G_lock_highway, G_valid_locks):
#     """
#     Return which locks each lock can get using only highway as a dictionary
#     :param G_lock_highway: (ut.HighwayCoord) directed graph with only highway nodes and lock nodes, has edges
#     :param G_valid_locks: (ut.HighwayCoord) contain the finalized, valid, true locks (next to town & highway)
#     :return lock_dict: dict{lock s: {all locks reachable from lock s in G_lock_highway}}
#     """
#     lock_dict = dict()  # init lock_dict as empty dict
#     i = 0
#     out = 100
#     for s in G_valid_locks.nodes:
#         i += 1
#         print("\tstep {}".format(i)) if i % out == 0 else None  # output the progress periodically
#         # s_descendants: all descendants (reachable) nodes in G_lock_highway starting from s
#         s_descendants = set(nx.descendants(G_lock_highway, s))
#         # lock_dict[s]: all locks in the s_descendants == all reachable lock nodes from s
#         lock_dict[s] = {lock for lock in s_descendants if lock in G_valid_locks.nodes}
#     return lock_dict


def get_lock_reachability(G_lock_highway, G_valid_locks):
    """
    Return which locks each lock can get using only highway as a dictionary
    :param G_lock_highway: (ut.HighwayCoord) directed graph with only highway nodes and lock nodes, has edges
    :param G_valid_locks: (ut.HighwayCoord) contain the finalized, valid, true locks (next to town & highway)
    :return lock_dict: dict{lock s: {all locks reachable from lock s in G_lock_highway}}
    """
    # TODO (NOTE): can override descendent so that the flow stops at any 2nd lock (instead of the end)
    lock_dict = dict()      # init lock_dict as empty dict
    i = 0
    out = 100
    for s1 in G_valid_locks.nodes:
        lock_dict[s1] = set()
        i += 1
        print("\tstep {} done".format(i)) if i % out == 0 else None  # output the progress periodically
        for s2 in G_valid_locks.nodes:
            if s1 == s2:
                continue
            try:
                path = nx.dijkstra_path(G_lock_highway, s1, s2)
                illegal_flag = False
                for tmp in path:
                    if tmp == s1 or tmp == s2:
                        continue
                    if tmp in G_valid_locks.nodes:
                        illegal_flag = True
                        break
                if not illegal_flag:
                    lock_dict[s1].add(s2)
            except nx.NetworkXNoPath:
                _ = None
    return lock_dict


def get_town_reachability(DiG_everything, towns):
    """
    Return which towns each town can reach using only highway as a dictionary
    :param DiG_everything: (ut.HighwayCoord) DiGraph with ALL NODES (hws, locks, towns) connected accordingly
    :param towns: list of town classes, each represent a town
    :return town_cannot_reach: dict{town i: {all town ids NOT reachable from town i}}
    """
    town_cannot_reach = dict()  # init town_dict as empty dict
    valid_towns = []
    invalid_towns = []          # either no lock / town node OR not connected with other towns
    # get all valid towns
    for t in towns:
        if t.num_lock_nodes > 0 and t.num_town_nodes > 0:
            # TODO: how to define valid towns, what if some big towns are invalid?
            valid_towns.append(t)  # for now, valid towns: must have >0 lock nodes & >0 town nodes
        else:
            print("Town #{} invalid due to size".format(t.town_index))
            invalid_towns.append(t)
    print(f"There are {len(valid_towns)} valid towns in {len(towns)} total towns.")
    # for each valid town, get a random town node; recall towns are strongly connected
    valid_towns_nodes = dict()
    for t in valid_towns:
        for n in t.G_town_lock.nodes:
            if n.label == '.':  # only add town nodes, not lock nodes
                valid_towns_nodes[t.town_index] = n  # dict: town_index -> town node
                break
    # determine reachability (only store those that cannot reach each other)
    for t1 in valid_towns:
        town_cannot_reach[t1.town_index] = set()  # the set stores town index not reachable from t1
        for t2 in valid_towns:
            if t1 == t2:  # make sure the two towns are not the same
                continue
            n1, n2 = valid_towns_nodes[t1.town_index], valid_towns_nodes[t2.town_index]
            if not nx.has_path(DiG_everything, n1, n2):  # no path between n1, n2 => t1, t2 not connected by highways
                town_cannot_reach[t1.town_index].add(t2.town_index)
    # replace empty set with some information
    not_connected = 0
    for t in valid_towns:
        if len(town_cannot_reach[t.town_index]) == 0:
            town_cannot_reach[t.town_index] = 'None'
        else:
            print("Town #{} invalid due to connectivity".format(t.town_index))
            invalid_towns.append(t)
            not_connected += 1
    # print information
    if not_connected == 0:
        print("ALL valid towns are pairwise reachable from highways.")
    else:
        print(f"{not_connected} valid towns are NOT connected within {len(valid_towns)} total valid towns.")
        for k, v in town_cannot_reach.items():
            if v:
                print("\t", towns[k], v)
    return town_cannot_reach, valid_towns, invalid_towns


def create_town_classes(G_town_only, G_valid_locks):
    """
    Create each town as a class holding important information about the town&lock
    :param G_town_only: (ut.Coord) containing only non-highway nodes (possibly town & locks, no distinction)
    :param G_valid_locks: (ut.HighwayCoord) contain the finalized, valid, true locks (next to town & highway)
    :return towns: list of town classes, each represent a town
    """
    # connected_comp = towns (including locks) = connected components from the original G_town_only (ut.Coord)
    connected_comp = [G_town_only.subgraph(c).copy() for c in nx.connected_components(G_town_only)]
    # create town classes
    max_size_town = 0
    towns = []
    for i, cc in enumerate(connected_comp):
        T = ut.Town(cc, G_valid_locks, i)
        towns.append(T)
        size = T.num_town_nodes + T.num_lock_nodes
        if size > max_size_town:
            max_size_town = size
    return towns, max_size_town


def highway_lock_town_digraph(G_lock_highway, towns):
    """
    Build a directed graph with ALL NODES (hws, locks, towns) connected accordingly, to check connectivity of map
    :param G_lock_highway: (ut.HighwayCoord) DiGraph with only highway nodes and lock nodes, has di-edges
    :param towns: list of town classes with G_town_lock (ut.HighwayCoord) DiGraph of town & lock
    :return DiG_result: (ut.HighwayCoord) DiGraph with ALL NODES (hws, locks, towns) connected accordingly
    """
    DiG_result = nx.DiGraph()
    # 1. add all pure TOWN nodes to DiG_result from towns
    for town in towns:
        for t_node in town.G_town_lock.nodes:  # iterate all ut.HighwayCoord in town.G_town_lock.nodes
            if t_node not in G_lock_highway.nodes:  # if in G_lock_highway, must be valid locks; otherwise must town
                assert t_node.label == '.'
                DiG_result.add_node(t_node)  # only add town nodes
    # 2. add all LOCK & HIGHWAY nodes to DiG_result from G_lock_highway
    for node in G_lock_highway.nodes:  # iterate all ut.HighwayCoord in G_lock_highway.nodes
        DiG_result.add_node(node)
    # 3. add TOWN-TOWN & TOWN-LOCK edges to DiG_result from towns
    for town in towns:
        for edge in town.G_town_lock.edges:
            DiG_result.add_edge(edge[0], edge[1])
    # 4. add LOCK-HW & HW-HW edges to DiG_result from G_lock_highway
    for edge in G_lock_highway.edges:
        DiG_result.add_edge(edge[0], edge[1])
    return DiG_result


def to_file(
        G_together_new, G_town_only, G_valid_locks, G_lock_highway, lock_dict, mapf_map_name, mapf_map,
        nrows, ncols, valid_towns_map, invalid_towns_map
):
    """
    Create & Write to the output files
    :param G_together_new: (ut.HighwayCoord): a complete graph containing highway nodes, town nodes, valid lock nodes
    :param G_town_only: (ut.Coord) containing only non-highway nodes (possibly town & locks, no distinction)
    :param G_valid_locks: (ut.HighwayCoord) contain the finalized, valid, true locks (next to town & highway)
    :param G_lock_highway: (ut.HighwayCoord) directed graph with only highway nodes and lock nodes, has edges
    :param lock_dict: dict{lock s: {all locks reachable from lock s in G_lock_highway}} from get_lock_reachability
    :param mapf_map_name: name of the mapf-baseline-map
    :param mapf_map: mapf_map = config[mapf_map_name] the config dictionary
    :param nrows: number of rows in the mapf-map (corresponds to horizontal highways)
    :param ncols: number of cols in the mapf-map (corresponds to vertical highways)
    :return: None
    """
    # set up result paths
    result_path = f"results/{mapf_map_name}"  # result folder, if not exist, create one
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    hw_path = f"{result_path}/highways.txt"  # highway output
    cfg_path = f"{result_path}/map_config_used.json"  # config used
    town_path = f"{result_path}/towns.json"  # towns output
    lock_dict_path = f"{result_path}/locks.txt"  # lock reachability output
    road_lock_edge_path = f"{result_path}/rl_edge.txt"  # G_lock_highway all nodes' directions (u, d, l, r)
    vis_gmap_path = f"{result_path}/vis_gmap.json"  # vis_gmap.json for visualize = global map
    # create highway map
    temp_hw_map = ["" for _ in range(0, nrows)]  # init temp_hw_map as list of empty strings with nrows rows
    for y in range(nrows - 1, -1, -1):  # y: row index, from nrows-1 to 0 decreasing
        for x in range(0, ncols):  # x: col index, from 0 to ncols-1 increasing
            temp_node = ut.HighwayCoord(x, y, '')
            if temp_node in G_together_new:  # (x, y) is not obstacle (can be highway, town, lock)
                # original_node: node corresponds to the (x, y) location in G_together_new
                original_node = G_together_new.nodes[temp_node]
                # label it with dir_out: len-1 string returned from parsing G_together_new node label
                label = deal_with_hw_label(original_node['label'])
                dir_out = get_output_hw_label(label)
                temp_hw_map[nrows - y - 1] += dir_out
            else:  # (x, y) is obstacle, label it with '@'
                temp_hw_map[nrows - y - 1] += '@'
    # create visualize global map
    g_map = ["" for _ in range(0, nrows)]
    for y in range(nrows - 1, -1, -1):  # y: row index, from nrows-1 to 0 decreasing
        for x in range(0, ncols):  # x: col index, from 0 to ncols-1 increasing
            temp_node = ut.HighwayCoord(x, y, '')
            if temp_node in G_together_new:  # (x, y) is not obstacle (can be highway, town, lock)
                # original_node: node corresponds to the (x, y) location in G_together_new
                original_node = G_together_new.nodes[temp_node]
                # label it with dir_out: len-1 string returned from parsing G_together_new node label
                label = deal_with_hw_label(original_node['label'])
                dir_out = get_output_global_label(label)
                g_map[nrows - y - 1] += dir_out
            else:  # (x, y) is obstacle, label it with '@'
                g_map[nrows - y - 1] += '@'
    annotation = g_map
    vtype_to_p_id = {
        '`': "LightSlateGray",
        "s": "LightBlue",
        "n": "LightGreen",
        "e": "mistyrose",
        "w": "PapayaWhip",
        "I": "Salmon",
        "L": "Khaki"
    }
    vis_gmap_js = {
        "g_map": g_map,
        "vtype_to_p_id": vtype_to_p_id,
        "annotation": annotation
    }
    # connected_comp = towns (including locks) = connected components from the original G_town_only (ut.Coord)
    connected_comp = [G_town_only.subgraph(c).copy() for c in nx.connected_components(G_town_only)]
    cp_index = -1  # cp_index keeps track of town ID
    result_towns_js = dict()  # result_towns_js: store result towns as a dict (later output with json)
    lockid_to_townid = dict()
    for cp in connected_comp:  # iterate each town (cp) in connected_comp
        cp_index += 1
        if cp_index in invalid_towns_map:
            curr_town = None
            print("\ttown id:", cp_index, "not in valid_towns_map")
        else:
            curr_town = valid_towns_map[cp_index]
        min_x, min_y, max_x, max_y = bbox_graph(cp, nrows, ncols)  # get bbox of the town
        temp_origin = min_x, min_y  # origin of them town: left-bottom corner of the town
        temp_map = ["" for _ in range(min_y, max_y + 1)]  # init temp_map as list of empty strings with ... rows
        for y in range(max_y, min_y - 1, -1):  # y: row index
            for x in range(min_x, max_x + 1):  # x: col index
                temp_node = ut.Coord(x, y, '')
                temp_node_L = ut.HighwayCoord(x, y, '')
                # if invalid town, all nodes is '@'
                if curr_town is None:
                    temp_map[max_y - y] += '@'
                # if (x, y) is valid lock and in the current town, use 'L' for lock
                elif temp_node_L in G_valid_locks.nodes and temp_node_L in curr_town.T_valid_locks:
                    temp_map[max_y - y] += 'L'
                    new_key = x + y * ncols
                    lockid_to_townid[new_key] = cp_index
                # else if (x, y) is in cp.nodes, use '.' for town
                elif temp_node in cp.nodes and temp_node_L in curr_town.G_town_lock:
                    temp_map[max_y - y] += '.'
                else:  # otherwise (x, y) is obstacle, use '@'
                    temp_map[max_y - y] += '@'
        # update result_towns_js at cp_index (a specific town)
        result_towns_js[cp_index] = {
            'town id': cp_index,
            'map': temp_map,
            'origin': temp_origin
        }
    # work with dict formatting for lock_dict
    new_lock_dict = {}
    for k in lock_dict.keys():  # encoding: i = Y * n_cols + X, decode by X = i % n_cols, Y = i // n_cols
        new_key = k.x + k.y * ncols
        vals = [(j.x + j.y * ncols) for j in lock_dict[k]]
        new_lock_dict[new_key] = vals
    # write to files
    with open(hw_path, 'w') as f:
        f.write('height (n_rows): {} \n'.format(nrows))
        f.write('width (n_cols): {} \n'.format(ncols))
        f.write('Highway Map: \n')
        f.write('\n'.join(temp_hw_map))
    with open(cfg_path, "w") as f:
        json.dump([mapf_map], f, indent=4)
    with open(town_path, "w") as f:
        json.dump(result_towns_js, f, indent=4)
    with open(vis_gmap_path, "w") as f:
        json.dump(vis_gmap_js, f, indent=4)
    with open(lock_dict_path, "w") as f:
        for k in lockid_to_townid.keys():
            f.write('{} {} '.format(k, lockid_to_townid[k]))
            reach = ["{} ".format(i) for i in new_lock_dict[k]]
            f.writelines(reach)
            f.write('\n')
    with open(road_lock_edge_path, "w") as f:  # u,d,l,r = binary indicates availability
        for node in G_lock_highway.nodes:  # node.x, node.y, node.label
            # node_idx = node.x + node.y * ncols  # no use now
            u, d, l, r = 0, 0, 0, 0
            for edge in G_lock_highway.edges:
                if edge[0].x == node.x and edge[0].y == node.y:
                    if edge[1].x == edge[0].x + 1 and edge[1].y == edge[0].y:
                        r = 1
                    elif edge[1].x == edge[0].x - 1 and edge[1].y == edge[0].y:
                        l = 1
                    elif edge[1].x == edge[0].x and edge[1].y == edge[0].y + 1:
                        u = 1
                    elif edge[1].x == edge[0].x and edge[1].y == edge[0].y - 1:
                        d = 1
                    else:
                        print("error")
                        exit(1)
            f.write('{} {} {} {} {} {} '.format(node.x, node.y, u, d, l, r))
            f.write('\n')


if __name__ == '__main__':
    "Load Config, store everything needed"
    json_path = "../json_configs/config_v2.json"
    config = json.load(open(json_path))  # config: the json dict
    mapf_map_name = "Shanghai_0_256"  # change this name for different mapf-file
    mapf_map = config[mapf_map_name]
    f_path = mapf_map["f_path"]  # mapf-map file path
    cell_types = mapf_map["cell_types"]  # tells what kind of fild are space / obstacles (space is . usually)
    max_town_size = mapf_map["max_town_size"]  # maximum town size expected, should be smaller than the desired
    max_search_around = mapf_map["max_search_around"]  # max distance searching around when find actual endpoints
    visited_add_cost = mapf_map["visited_add_cost"]  # when planning highway, add cost when path taken (same direction)
    max_allowed_cost = mapf_map["max_allowed_cost"]  # when planning highway, remove edge if cost over this threshold
    "Get original graph G, apply Config, get directed version also"
    nrows, ncols, G = load_G(f_path, cell_types)  # get original graph
    G = largest_connected_component(G)  # only keep the subgraph of the largest connected component
    plot_unpartitioned_G(G)
    G_d = build_directed_from_undirected(G)  # get the directed version of G
    ideal_num_strip = ideal_strips(nrows, ncols, max_town_size)
    "Prepare for layout highway"
    # get ideal endpoints of highways, ignoring obstacles, as 2D matrix (dict)
    ideal_endpts, x_width, y_width = ideal_endpts_dict(
        nrows, ncols, ideal_num_strip, ideal_num_strip
    )
    # get actual endpoints of highways, considering obstacles, as 2D matrix (dict)
    actual_endpts = actual_endpts_dict(G, ideal_endpts, max_search_around)
    # get outer_endpts_list storing index of outer highway endpoints in clockwise order
    outer_endpts_list = get_outer_hw_endpts(actual_endpts)  # index list
    # get the graph storing prospective highways endpoints' corresponding location in actual_endpts index
    G_endpts_index = actual_endpts_graph(actual_endpts)  # index graph
    "Plan Highway Layout"
    # plan outer highways
    outer_highways = plan_outer_highways(G_d, outer_endpts_list, actual_endpts)
    # plan inner highways
    inner_highways = plan_highways(
        G_d, G_endpts_index, actual_endpts, visited_add_cost, max_allowed_cost, x_width, y_width
    )
    all_highways = outer_highways + inner_highways
    # build highway on a new, empty graph, only storing highway information (node, direction, etc.)
    G_hw = build_highway_on_empty(all_highways)
    "Post-Processing Final Maps"
    # get different graphs containing different information using the highway information, also build temp locks
    G_together, G_town_only, G_highway_only, G_temp_locks, G_final_town = highway_post_processing(G, G_hw, all_highways)
    # get valid locks from temp locks and put everything in G_together_new
    G_valid_locks, G_together_new = get_valid_locks(G_together, G_temp_locks, G_final_town)
    # merge lock & highways => G_lock_highway (DiGraph of ut.HighwayCoord)
    G_lock_highway = highway_lock_digraph(G_highway_only, G_valid_locks)
    # get towns as a list of Town Class
    towns, max_size_town = create_town_classes(G_town_only, G_valid_locks)
    print(f"=== max size town has {max_size_town} nodes ===")
    # merge town & lock & highways => DiG_everything (DiGraph of ut.HighwayCoord)
    DiG_everything = highway_lock_town_digraph(G_lock_highway, towns)
    # get town reachability and store in town_dict => which town each town can reach
    town_dict, valid_towns, invalid_towns = get_town_reachability(DiG_everything, towns)
    invalid_towns_map = {t.town_index: t for t in invalid_towns}
    valid_towns_map = {t.town_index: t for t in valid_towns}
    print(f"=== get_town_reachability complete ===")
    # get lock reachability and store in lock_dict => which lock each lock and get using only highway
    lock_dict = get_lock_reachability(G_lock_highway, G_valid_locks)
    print(f"=== get_lock_reachability complete ===")
    "Create & Write to Output Files"
    to_file(
        G_together_new, G_town_only, G_valid_locks, G_lock_highway, lock_dict, mapf_map_name,
        mapf_map, nrows, ncols, valid_towns_map, invalid_towns_map
    )
    print(f"=== to_file complete ===")
    "Display Images"
    color_map, label_map = color_graph(G_together_new)  # get colors
    node_size_map = node_size_graph(G_together_new)  # get sizes
    plot_unpartitioned_G(G_together_new, color_map, node_size_map)  # plot the final graph
