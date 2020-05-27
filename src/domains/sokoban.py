import numpy as np
from domains.environment import Environment

class Sokoban(Environment):
    
    def __init__(self, string_state):
        
        self._number_channels = 4
        self._width = len(string_state[0])
        self._height = len(string_state)
        
        self._channel_walls = 0
        self._channel_boxes = 1
        self._channel_man = 2
        self._channel_goals = 3
        
        self._goal = '.'
        self._man = '@'
        self._wall = '#'
        self._box = '$'
        
        self._puzzle = np.zeros((self._height, self._width, self._number_channels))
        
        for i in range(self._height):
            for j in range(self._width):
                if string_state[i][j] == self._goal:
                    self._puzzle[i][j][self._channel_goals] = 1
                    
                if string_state[i][j] == self._man:
                    self._puzzle[i][j][self._channel_man] = 1
                    self._y_man = i
                    self._x_man = j
                    
                if string_state[i][j] == self._wall:
                    self._puzzle[i][j][self._channel_walls] = 1
                if string_state[i][j] == self._box:
                    self._puzzle[i][j][self._channel_boxes] = 1
                    
        self._E = 0
        self._W = 1
        self._N = 2
        self._S = 3
    
    def __hash__(self):
        return hash(str(self._puzzle))

    def __eq__(self, other):        
        return np.array_equal(self._puzzle, other._puzzle)
    
    def successors(self):
        actions = []
        
        if self._x_man + 1 < self._width:
            if (self._puzzle[self._y_man][self._x_man + 1][self._channel_walls] == 0 and
                self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 0):
                
                actions.append(self._E)
            elif (self._puzzle[self._y_man][self._x_man + 1][self._channel_walls] == 0 and 
                self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 1 and 
                self._x_man + 2 < self._width and 
                self._puzzle[self._y_man][self._x_man + 2][self._channel_walls] == 0 and  
                self._puzzle[self._y_man][self._x_man + 2][self._channel_boxes] == 0):
                
                actions.append(self._E)
            
        if self._x_man - 1 > 0:
            if (self._puzzle[self._y_man][self._x_man - 1][self._channel_walls] == 0 and  
                self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 0):
                
                actions.append(self._W)
            elif (self._puzzle[self._y_man][self._x_man - 1][self._channel_walls] == 0 and  
                self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 1 and
                self._x_man - 2 > 0 and 
                self._puzzle[self._y_man][self._x_man - 2][self._channel_walls] == 0 and  
                self._puzzle[self._y_man][self._x_man - 2][self._channel_boxes] == 0):
                
                actions.append(self._W)
                
        if self._y_man + 1 < self._height:
            if (self._puzzle[self._y_man + 1][self._x_man][self._channel_walls] == 0 and  
                self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 0):
                
                actions.append(self._S)
            elif (self._puzzle[self._y_man + 1][self._x_man][self._channel_walls] == 0 and 
                self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 1 and
                self._y_man + 2 < self._height and 
                self._puzzle[self._y_man + 2][self._x_man][self._channel_walls] == 0 and 
                self._puzzle[self._y_man + 2][self._x_man][self._channel_boxes] == 0):
                
                actions.append(self._S)
                
        if self._y_man - 1 > 0:
            if (self._puzzle[self._y_man - 1][self._x_man][self._channel_walls] == 0 and  
                self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 0):
                
                actions.append(self._N)
            elif (self._puzzle[self._y_man - 1][self._x_man][self._channel_walls] == 0 and  
                self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 1 and
                self._y_man - 2 > 0 and 
                self._puzzle[self._y_man - 2][self._x_man][self._channel_walls] == 0 and  
                self._puzzle[self._y_man - 2][self._x_man][self._channel_boxes] == 0):
                
                actions.append(self._N)
    
        return actions;
    
    def successors_parent_pruning(self, op):
        actions = []
        
        return self.successors()
        
#         if self._x_man + 1 < self._width and not (op == self._W and self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 0):
#             if (self._puzzle[self._y_man][self._x_man + 1][self._channel_walls] == 0 and
#                 self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 0):
#                 
#                 actions.append(self._E)
#             elif (self._puzzle[self._y_man][self._x_man + 1][self._channel_walls] == 0 and 
#                 self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 1 and 
#                 self._x_man + 2 < self._width and 
#                 self._puzzle[self._y_man][self._x_man + 2][self._channel_walls] == 0 and  
#                 self._puzzle[self._y_man][self._x_man + 2][self._channel_boxes] == 0):
#                 
#                 actions.append(self._E)
#             
#         if self._x_man - 1 > 0 and not (op == self._E and self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 0):
#             if (self._puzzle[self._y_man][self._x_man - 1][self._channel_walls] == 0 and  
#                 self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 0):
#                 
#                 actions.append(self._W)
#             elif (self._puzzle[self._y_man][self._x_man - 1][self._channel_walls] == 0 and  
#                 self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 1 and
#                 self._x_man - 2 > 0 and 
#                 self._puzzle[self._y_man][self._x_man - 2][self._channel_walls] == 0 and  
#                 self._puzzle[self._y_man][self._x_man - 2][self._channel_boxes] == 0):
#                 
#                 actions.append(self._W)
#                 
#         if self._y_man + 1 < self._height and not (op == self._N and self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 0):
#             if (self._puzzle[self._y_man + 1][self._x_man][self._channel_walls] == 0 and  
#                 self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 0):
#                 
#                 actions.append(self._S)
#             elif (self._puzzle[self._y_man + 1][self._x_man][self._channel_walls] == 0 and 
#                 self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 1 and
#                 self._y_man + 2 < self._height and 
#                 self._puzzle[self._y_man + 2][self._x_man][self._channel_walls] == 0 and 
#                 self._puzzle[self._y_man + 2][self._x_man][self._channel_boxes] == 0):
#                 
#                 actions.append(self._S)
#                 
#         if self._y_man - 1 > 0 and not (op == self._S and self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 0):
#             if (self._puzzle[self._y_man - 1][self._x_man][self._channel_walls] == 0 and  
#                 self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 0):
#                 
#                 actions.append(self._N)
#             elif (self._puzzle[self._y_man - 1][self._x_man][self._channel_walls] == 0 and  
#                 self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 1 and
#                 self._y_man - 2 > 0 and 
#                 self._puzzle[self._y_man - 2][self._x_man][self._channel_walls] == 0 and  
#                 self._puzzle[self._y_man - 2][self._x_man][self._channel_boxes] == 0):
#                 
#                 actions.append(self._N)
    
        return actions;
    
    def apply_action(self, action):
        
        if action == self._N: 
            if self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] == 1:
                self._puzzle[self._y_man - 1][self._x_man][self._channel_boxes] = 0
                self._puzzle[self._y_man - 2][self._x_man][self._channel_boxes] = 1
            self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            self._puzzle[self._y_man - 1][self._x_man][self._channel_man] = 1
            self._y_man = self._y_man - 1 
    
        if action == self._S:
            if self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] == 1:
                self._puzzle[self._y_man + 1][self._x_man][self._channel_boxes] = 0
                self._puzzle[self._y_man + 2][self._x_man][self._channel_boxes] = 1
            self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            self._puzzle[self._y_man + 1][self._x_man][self._channel_man] = 1
            self._y_man = self._y_man + 1 
    
        if action == self._E:
            if self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] == 1:
                self._puzzle[self._y_man][self._x_man + 1][self._channel_boxes] = 0
                self._puzzle[self._y_man][self._x_man + 2][self._channel_boxes] = 1
            self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            self._puzzle[self._y_man][self._x_man + 1][self._channel_man] = 1
            self._x_man = self._x_man + 1 
    
        if action == self._W:
            if self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] == 1:
                self._puzzle[self._y_man][self._x_man - 1][self._channel_boxes] = 0
                self._puzzle[self._y_man][self._x_man - 2][self._channel_boxes] = 1
            self._puzzle[self._y_man][self._x_man][self._channel_man] = 0
            self._puzzle[self._y_man][self._x_man - 1][self._channel_man] = 1
            self._x_man = self._x_man - 1 
    
    def is_solution(self):
        for i in range(self._height):
            for j in range(self._width):
                if self._puzzle[i][j][self._channel_boxes] == 1 and self._puzzle[i][j][self._channel_goals] == 0:
                    return False 
        return True
    
    def get_image_representation(self):
        return self._puzzle
    
    def heuristic_value(self):
        h = 0
        h_man = self._width + self._height
        
        for i in range(self._height):
            for j in range(self._width):        
                if self._puzzle[i][j][self._channel_boxes] == 1 and self._puzzle[i][j][self._channel_goals] == 0:
                    h_box = self._width + self._height
                    for l in range(self._height):
                        for m in range(self._width):
                            if self._puzzle[l][m][self._channel_goals] == 1:
                                dist_to_goal = abs(l - i) + abs(m - j)
                                if dist_to_goal < h_box:
                                    h_box = dist_to_goal
                    h += h_box
                    
                if self._puzzle[i][j][self._channel_boxes] == 1:
                    dist_to_man = abs(self._y_man - i) + abs(self._x_man - j) - 1
                    if dist_to_man < h_man:
                        h_man = dist_to_man
        h += h_man
        return h
        
    def print(self):
        for i in range(self._height):
            for j in range(self._width):
                if self._puzzle[i][j][self._channel_goals] == 1 and self._puzzle[i][j][self._channel_boxes] == 1:
                    print('*', end='')
                elif self._puzzle[i][j][self._channel_man] == 1:
                    print(self._man, end='')
                elif self._puzzle[i][j][self._channel_goals] == 1:
                    print(self._goal, end='')
                elif self._puzzle[i][j][self._channel_walls] == 1:
                    print(self._wall, end='')
                elif self._puzzle[i][j][self._channel_boxes] == 1:
                    print(self._box, end='')
                else:
                    print(' ', end='')
            print()