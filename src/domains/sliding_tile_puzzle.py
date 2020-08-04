import numpy as np
import math
from domains.environment import Environment
import copy

class SlidingTilePuzzle(Environment):
    
    def __init__(self, tiles):
        if isinstance(tiles, str):
            tiles = tiles.replace('\n', '').split(' ')
            self._tiles = []
            
            for tile in tiles:
                if tile == '':
                    continue
                if tile == 'B':
                    self._tiles.append(0)
                else:
                    self._tiles.append(int(tile))
        else:
            self._tiles = tiles
        
        self._size = len(self._tiles)
        self._width = int(math.sqrt(self._size))
        
        self._pos = np.zeros(self._size)
        self._op = 0
        self._op = -1
        
        for i in range(self._size):
            self._pos[self._tiles[i]] = i
            
            if self._tiles[i] == 0:
                self._blank = i
        self._E = 0
        self._W = 1
        self._N = 2
        self._S = 3
        
    def is_valid(self):
        t = 0
        
        if not(self._width & 1) > 0:
            t = self._pos[0] // self._width;
            
        for i in range(2, self._size):
            for l in range(1, i):
                if self._pos[i] < self._pos[l]:
                    t += 1
        
        return (int(t) & 1) ^ 1 == 1
        
    def copy(self):
        return copy.deepcopy(self)
        
    def getSize(self):
        return self._size
    
    def getWidth(self):
        return self._width
    
    def getValueTile(self, i):
        return self._tiles[i]
    
    def __hash__(self):
        return hash(str(self._tiles))

    def __eq__(self, other):        
        for i in range(self._size):
            if other._tiles[i] != self._tiles[i]:
                return False
        return True
    
    def save_state(self, filename):
        file = open(filename, 'a+')
        
        for i in range(self._size):
            file.write(str(self._tiles[i]) + ' ')
        file.write('\n')
            
        file.close()
    
    def successors(self):
        actions = []
        
        if not ((self._blank + 1) % self._width == 0): # and op != self._W:
            actions.append(self._E);
        
        if self._blank > self._width - 1: # and op != self._S: 
            actions.append(self._N);
        
        if not((self._blank) % self._width == 0): # and op != self._E: 
            actions.append(self._W);
    
        if self._blank < self._size - self._width: # and op != self._N: 
            actions.append(self._S);
    
        return actions;
    
    def successors_parent_pruning(self, op):
        actions = []
        
        if not ((self._blank + 1) % self._width == 0) and op != self._W:
            actions.append(self._E);
        
        if self._blank > self._width - 1 and op != self._S: 
            actions.append(self._N);
        
        if not((self._blank) % self._width == 0) and op != self._E: 
            actions.append(self._W);
    
        if self._blank < self._size - self._width and op != self._N: 
            actions.append(self._S);
    
        return actions;
    
    def apply_action(self, action):
        
        if action == self._N: 
            self._tiles[self._blank] = self._tiles[self._blank - self._width];
            self._pos[self._tiles[self._blank - self._width]] = self._blank;
            self._tiles[self._blank - self._width] = 0;
            self._pos[0] = self._blank - self._width;
            self._blank = self._blank - self._width;
    
        if action == self._S:
            self._tiles[self._blank] = self._tiles[self._blank + self._width];
            self._pos[self._tiles[self._blank + self._width]] = self._blank;
            self._tiles[self._blank + self._width] = 0;
            self._pos[0] = self._blank + self._width;
            self._blank = self._blank + self._width;
    
        if action == self._E:
            self._tiles[self._blank] = self._tiles[self._blank + 1];
            self._pos[self._tiles[self._blank + 1]] = self._blank;
            self._tiles[self._blank + 1] = 0;
            self._pos[0] = self._blank + 1;
            self._blank = self._blank + 1;
    
        if action == self._W:
            self._tiles[self._blank] = self._tiles[self._blank - 1];
            self._pos[self._tiles[self._blank - 1]] = self._blank;
            self._tiles[self._blank - 1] = 0;
            self._pos[0] = self._blank - 1;
            self._blank = self._blank - 1;
    
    def is_solution(self):
        for i in range(self._size):
            if self._tiles[i] != i:
                return False
        return True
    
    def get_image_representation(self):
         
        image = np.zeros((self._width, self._width, self._size))
        
        for i in range(self._size):
            l = int(self._pos[i] / self._width)
            c = int(self._pos[i] % self._width)
            
            image[l][c][i] = 1
            
        return image
    
    def heuristic_value(self):
        h = 0
        
        for i in range(0, self._size):
            if self._tiles[i] == 0:
                continue     
            h = h + abs((self._tiles[i] % self._width) - (i % self._width)) + abs(int((self._tiles[i] / self._width)) - int((i / self._width)));

        return h
        
    def print(self):
        for i in range(len(self._tiles)):
            print(self._tiles[i], end=' ')
            if (i + 1) % self._width == 0:
                print()