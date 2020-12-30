import numpy as np
import matplotlib
from domains.environment import Environment
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from collections import deque
import copy

class InvalidPuzzlePositionException(Exception):
    pass

class InvalidColorException(Exception):
    pass

class WitnessState(Environment):
    """
    This class reprensents a state of a puzzle inspired in "separable-colors puzzle" from The Witness
    
    The state is represented by several matrices:
        (1) _cells stores the colors in each cell, where 0 represents the neutral color
        (i.e., the color that doesn't have to be separated from the others).
        (2) _dots stores the junctions of the cells, where the "snake" can navigate. If
        _cells contains L lines and C columns, then _dots has L+1 lines and C+1 columns. 
        (3) _v_seg stores the vertical segments of the puzzle that is occupied by the snake. 
        If _v_seg[i][j] equals 1, then the snake is going through that vertical segment. 
        _v_seg[i][j] equals 0 otherwise.
        (4) _h_seg is defined analogously for the horizontal segments. 
        
    In addition to the matrices, GameState also stores the line and column of the tip of the snake
    (see _line_tip and _column_tip). The class also contains the starting line and column (_line_init
    and _column_init).
    
    Currently the state supports 6 colors (see attribute _colors)
    """
    # Possible colors for separable bullets
    _colors = ['b', 'r', 'g', 'c', 'y', 'm']
    
    def __init__(self, lines=1, columns=1, line_init=0, column_init=0, line_goal=1, column_goal=1, max_lines=11, max_columns=11):
        """
        GameState's constructor. The constructor receives as input the variable lines and columns,
         which specify the number of lines and columns of the puzzle; line_init and column_init speficy
         the entrance position of the snake in the puzzle's grid; line_goal and column_goal speficy 
         the exit position of the snake on the grid; max_lines and max_columns are used to embed the puzzle
         into an image of fixed size (see method get_image_representation). The default value of max_lines 
         and max_columns is 4. This value has to be increased if working with puzzles larger than 4x4 grid cells.    
         
         The constructor has a step of default values for the variable in case one wants to read a puzzle 
         from a text file through method read_state.
        """
        self._v_seg = np.zeros((lines, columns+1))
        self._h_seg = np.zeros((lines+1, columns))
        self._dots = np.zeros((lines+1, columns+1))
        self._cells = np.zeros((lines, columns))

        self._column_init = column_init
        self._line_init = line_init
        self._column_goal = column_goal
        self._line_goal = line_goal
        self._lines = lines
        self._columns = columns
        self._line_tip = line_init
        self._column_tip = column_init
        self._max_lines = max_lines
        self._max_columns = max_columns
        
        # Raises an exception if the initial position of the snake equals is goal position
        if self._column_init == self._column_goal and self._line_init == self._line_goal:
            raise InvalidPuzzlePositionException('Initial postion of the snake cannot be equal its goal position', 'Initial: ' + 
                                                 str(self._line_init) + ', ' +  str(self._column_init) + 
                                                 ' Goal: ' + str(self._line_goal) + ', ' +  str(self._column_goal))
        
        # Raises an exception if the initial position of the snake is invalid
        if self._column_init < 0 or self._line_init < 0 or self._column_init > self._columns or self._line_init > self._lines:
            raise InvalidPuzzlePositionException('Initial position of the snake is invalid', 
                                                 str(self._line_init) + ', ' +  str(self._column_init))
        
        # Raises an exception if the goal position of the snake is invalid    
        if self._column_goal < 0 or self._line_goal < 0 or self._column_goal > self._columns or self._line_goal > self._lines:
            raise InvalidPuzzlePositionException('Goal position of the snake is invalid', 
                                                 str(self._line_goal) + ', ' +  str(self._column_goal))
        
        # Initializing the tip of the snake
        self._dots[self._line_tip][self._column_tip] = 1
        
        self._solution_depth = -1
        
    def __repr__(self):
        state_str = 'Cells: \n'
        state_str += '\n'.join('\t'.join('%d' %x for x in y) for y in self._cells)
        state_str += '\nDots: \n'
        state_str += '\n'.join('\t'.join('%d' %x for x in y) for y in self._dots)
        return state_str
    
    def __rotated90_position(self, x, y):
        matrix = np.zeros((self._lines+1, self._columns+1))
        matrix[x][y] = 1
        matrix = np.rot90(matrix)
        pos = np.argwhere(matrix==1)
        return pos[0][0], pos[0][1]
    
    def __flip_up_down_position(self, x, y):
        matrix = np.zeros((self._lines+1, self._columns+1))
        matrix[x][y] = 1
        matrix = np.flipud(matrix)
        pos = np.argwhere(matrix==1)
        return pos[0][0], pos[0][1]
    
    def clear_path(self):
        """
        This method resets a path that has be written to the state. This is achieved by reseting the following structures:
        (1) self._v_seg
        (2) self._h_seg
        (3) self._dots
        The tip of the snake is also reset to the init variables. Finally, the only dot filled with one is the tip of the snake.
        """
        self._v_seg = np.zeros((self._lines, self._columns+1))
        self._h_seg = np.zeros((self._lines+1, self._columns))
        self._dots = np.zeros((self._lines+1, self._columns+1))
        
        self._line_tip = self._line_init
        self._column_tip = self._column_init
        
        self._dots[self._line_tip][self._column_tip] = 1
    
    def get_rotate90_action(self, action):
        #up = 0, down = 1, right = 2, left = 3
        #mapping: up -> right (0 -> 2); right -> down (2 -> 1); down -> left (1 -> 3); left -> up (3 -> 0)
        mapping_actions = {0:2, 2:1, 1:3, 3:0}
        return mapping_actions[action]
    
    def get_flip_up_down_action(self, action):
        #up = 0, down = 1, right = 2, left = 3
        #mapping: up -> down (0 -> 1); right -> right (2 -> 2); down -> up (1 -> 0); left -> left (3 -> 3)
        mapping_actions = {0:1, 2:2, 1:0, 3:3}
        return mapping_actions[action]
    
    def rotate90(self):
        self._line_init, self._column_init = self.__rotated90_position(self._line_init, self._column_init)
        self._line_goal, self._column_goal = self.__rotated90_position(self._line_goal, self._column_goal)
        self._line_tip, self._column_tip = self.__rotated90_position(self._line_tip, self._column_tip)
        
        v_seg = self._v_seg
        self._v_seg = self._h_seg
        self._h_seg = v_seg
        
        self._v_seg = np.rot90(self._v_seg, 1)
        self._h_seg = np.rot90(self._h_seg, 1)
        self._dots = np.rot90(self._dots, 1)
        self._cells = np.rot90(self._cells, 1)
        columns = self._columns
        self._columns = self._lines
        self._lines = columns
    
    def flip_up_down(self):
        self._line_init, self._column_init = self.__flip_up_down_position(self._line_init, self._column_init)
        self._line_goal, self._column_goal = self.__flip_up_down_position(self._line_goal, self._column_goal)
        self._line_tip, self._column_tip = self.__flip_up_down_position(self._line_tip, self._column_tip)
        
        self._v_seg = np.flipud(self._v_seg)
        self._h_seg = np.flipud(self._h_seg)
        self._dots = np.flipud(self._dots)
        self._cells = np.flipud(self._cells)
    
    
    def get_image_representation(self):
        """
        Generates an image representation for the puzzle. Currently the method supports 4 colors and includes
        the following channels (third dimension of image): one channel for each color; one channel with 1's 
        where is "open" in the grid (this allows learning systems to work with a fixed image size defined
        by max_lines and max_columns); one channel for the current path (cells occupied by the snake);
        one channel for the tip of the snake; one channel for the exit of the puzzle; one channel for the
        entrance of the snake. In total there are 9 different channels. 
        
        Each channel is a matrix with zeros and ones. The image returned is a 3-dimensional numpy array. 
        """
                
        number_of_colors = 4
        channels = 9
        
        #defining the 3-dimnesional array that will be filled with the puzzle's information 
        image = np.zeros((2 * self._max_lines, 2 * self._max_columns, channels))

        #create one channel for each color i        
        for i in range(0, number_of_colors):
            for j in range(0, self._cells.shape[0]):
                for k in range(0, self._cells.shape[1]):
                    if self._cells[j][k] == i:
                        image[2*j+1][2*k+1][i] = 1
        channel_number = number_of_colors
        
        #the number_of_colors-th channel specifies the open spaces in the grid
        for j in range(0, 2 * self._lines + 1):
            for k in range(0, 2 * self._columns + 1):
                image[j][k][channel_number] = 1
        
        #channel for the current path
        channel_number += 1
        for i in range(0, self._v_seg.shape[0]):
            for j in range(0, self._v_seg.shape[1]):
                if self._v_seg[i][j] == 1:
                    image[2*i][2*j][channel_number] = 1
                    image[2*i+1][2*j][channel_number] = 1
                    image[2*i+2][2*j][channel_number] = 1
                    
        for i in range(0, self._h_seg.shape[0]):
            for j in range(0, self._h_seg.shape[1]):
                if self._h_seg[i][j] == 1:
                    image[2*i][2*j][channel_number] = 1
                    image[2*i][2*j+1][channel_number] = 1
                    image[2*i][2*j+2][channel_number] = 1
                    
        #channel with the tip of the snake
        channel_number += 1
        image[2*self._line_tip][2*self._column_tip][channel_number] = 1
        
        #channel for the exit of the puzzle
        channel_number += 1
        image[2*self._line_goal][2*self._column_goal][channel_number] = 1
        
        #channel for the entrance of the puzzle
        channel_number += 1
        image[2*self._line_init][2*self._column_init][channel_number] = 1
        
        return image
    
    def distance_images(self, other):
        distance = 0
        if self._column_goal != other._column_goal or self._line_goal != other._line_goal:
            distance += 1 
        if self._column_init != other._column_init or self._line_init != other._line_init:
            distance += 1
        
        for i in range(0, max(self._lines, other._lines)):
            for j in range(0, max(self._columns, other._columns)):
                if i >= self._lines or i >= other._lines:
                    distance += 100
                    continue
                if j >= self._columns or j >= other._columns:
                    distance += 100
                    continue
                if self._cells[i][j] != other._cells[i][j]:
                    distance += 1
        #print(self._filename, other._filename, distance)
        return distance
    
    def __print_image(self, image):
        for i in range(0, image.shape[2]):
            for j in range(0, image.shape[0]):
                for k in range(0, image.shape[0]):
                    print(image[j][k][i], end=' ')
                print()
            print('\n\n')
            
    def swap_colors(self):
        """
        Swaps the colors of the bullets. Method works only for puzzles with two colors. 
        """
        for i in range(0, self._cells.shape[0]):
            for j in range(0, self._cells.shape[1]):                
                if self._cells[i][j] == 1:
                    self._cells[i][j] = 2
                elif self._cells[i][j] == 2:
                    self._cells[i][j] = 1
                    
    def __canonical_colored_cells(self, cells):
        """
        Returns the canonical representation of the colors of the cells. In the canonical representation
        the first colored cell from left to right and top to bottom is mapped to color 1, the second 
        to 2 and so on. Two puzzles with the exact color structure have the same canonical representation.
        
        The canonical representation allows us to avoid duplicated puzzles that only differ on the colors used.  
        """
        counter = 1
        map_self = {}
        
        for i in range(0, cells.shape[0]):
            for j in range(0, cells.shape[1]):                
                if cells[i][j] != 0 and cells[i][j] not in map_self:
                    map_self[cells[i][j]] = counter
                    counter += 1
                    
        c_self = np.zeros((self._lines, self._columns))
        
        for i in range(0, cells.shape[0]):
            for j in range(0, cells.shape[1]):
                if cells[i][j] != 0:
                    c_self[i][j] = map_self[cells[i][j]]
        return c_self
    
    def __cell_color_invariant_eq__(self, other):
        """
        This method compares the _cells variable of two puzzles. But instead of using _cells directly,
        they use the canonical representation of each puzzle.
        """
        if self._cells.shape != other._cells.shape:
            return False
        
        for i in range(0, self._cells.shape[0]):
            for j in range(0, self._cells.shape[1]):
                if (self._cells[i][j] == 0 and other._cells[i][j] != 0) or (self._cells[i][j] != 0 and other._cells[i][j] == 0):
                    return False
                    
        c_self = self.__canonical_colored_cells(self._cells)
        c_other = self.__canonical_colored_cells(other._cells)

        return np.array_equal(c_self, c_other)
    
    def __hash__(self):
        #canonical_cells = self.__canonical_colored_cells(self._cells)
        #return hash((str(self._v_seg), str(self._h_seg), str(canonical_cells)))
        return hash((str(self._v_seg), str(self._h_seg), str(self._cells)))

    def __eq__(self, other):
        return (
                    np.array_equal(self._v_seg, other._v_seg) and np.array_equal(self._h_seg, other._h_seg) and 
                    np.array_equal(self._cells, other._cells) and self._column_init == other._column_init and
                    self._line_init == other._line_init and self._column_goal == other._column_goal and
                    self._line_goal == other._line_goal and self._line_tip == other._line_tip and 
                    self._column_tip == other._column_tip and np.array_equal(self._dots, other._dots)
                )
        #return np.array_equal(self._v_seg, other._v_seg) and np.array_equal(self._h_seg, other._h_seg) and self.__cell_color_invariant_eq__(other) 
        
    def color_invariant_eq(self, other):
        return np.array_equal(self._v_seg, other._v_seg) and np.array_equal(self._h_seg, other._h_seg) and self.__cell_color_invariant_eq__(other)
    
    def __init_structures(self):
        """
        This method initializes the puzzle's structures. We assume that the following attributes were set elsewhere: 
        (1) self._lines 
        (2) self._columns 
        (3) self._line_init
        (4) self._column_init
        (5) self._line_goal
        (6) self._column_goal
        """
        self._v_seg = np.zeros((self._lines, self._columns+1))
        self._h_seg = np.zeros((self._lines+1, self._columns))
        self._dots = np.zeros((self._lines+1, self._columns+1))
        self._cells = np.zeros((self._lines, self._columns))
        
        self._line_tip = self._line_init
        self._column_tip = self._column_init
        
        self._dots[self._line_tip][self._column_tip] = 1
        
    def reset(self):
        """
        This method resets a path that has be written to the state. This is achieved by reseting the following structures:
        (1) self._v_seg
        (2) self._h_seg
        (3) self._dots
        The tip of the snake is also reset to the init variables. Finally, the only dot filled with one is the tip of the snake.
        """
        self._v_seg = np.zeros((self._lines, self._columns+1))
        self._h_seg = np.zeros((self._lines+1, self._columns))
        self._dots = np.zeros((self._lines+1, self._columns+1))
        
        self._line_tip = self._line_init
        self._column_tip = self._column_init
        
        self._dots[self._line_tip][self._column_tip] = 1
        
    def add_color(self, line, column, color):
        """
        Sets a colored bullet into the puzzle. Variables line and column speficy the position in the puzzle
        where a bullet of color 'color' will be added. Variable color must be an integer. 
        """
        if color > len(self._colors):
            raise InvalidColorException('This is an invalid color', 'Color passed: ' + str(color) + 
                                        ' maximum color value: ' + str(len(self._colors)))
        self._cells[line][column] = color
    
    def plot(self):
        self.generate_image(False, '')
        
    def save_figure(self, filename):
        self.generate_image(True, filename)
                
    def generate_image(self, save_file, filename):
        """
        This method plots the state. Several features in this method are hard-coded and might
        need adjustment as one changes the size of the puzzle. For example, the size of the figure is set to be fixed
        to [5, 5] (see below).
        """
        fig = plt.figure(figsize=[5,5])
        fig.patch.set_facecolor((1,1,1))
        
        ax = fig.add_subplot(111)
        
        # draw vertical lines of the grid
        for y in range(self._dots.shape[1]):
            ax.plot([y, y], [0, self._cells.shape[0]], 'k')
        # draw horizontal lines of the grid
        for x in range(self._dots.shape[0]):
            ax.plot([0, self._cells.shape[1]], [x,x], 'k')
        
        # scale the axis area to fill the whole figure
        ax.set_position([0,0,1,1])
        
        ax.set_axis_off()
        
        ax.set_xlim(-1, np.max(self._dots.shape))
        ax.set_ylim(-1, np.max(self._dots.shape))
        
        # Draw the vertical segments of the path
        for i in range(self._v_seg.shape[0]):
            for j in range(self._v_seg.shape[1]):
                if self._v_seg[i][j] == 1:
                    ax.plot([j, j], [i, i+1], 'r', linewidth=5)
        
        # Draw the horizontal segments of the path            
        for i in range(self._h_seg.shape[0]):
            for j in range(self._h_seg.shape[1]):
                if self._h_seg[i][j] == 1:
                    ax.plot([j, j+1], [i, i], 'r', linewidth=5)
                
        # Draw the separable bullets according to the values in self._cells and self._colors
        offset = 0.5
        for i in range(self._cells.shape[0]):
            for j in range(self._cells.shape[1]):               
                if self._cells[i][j] != 0:
                    ax.plot(j+offset,i+offset,'o',markersize=15, markeredgecolor=(0,0,0), markerfacecolor=self._colors[int(self._cells[i][j]-1)], markeredgewidth=2)
        
        # Draw the intersection of lines: red for an intersection that belongs to a path and black otherwise
        for i in range(self._dots.shape[0]):
            for j in range(self._dots.shape[1]):               
                if self._dots[i][j] != 0:
                    ax.plot(j,i,'o',markersize=10, markeredgecolor=(0,0,0), markerfacecolor='r', markeredgewidth=0)
                else:
                    ax.plot(j,i,'o',markersize=10, markeredgecolor=(0,0,0), markerfacecolor='k', markeredgewidth=0)
        
        # Draw the entrance of the puzzle in red as it is always on the state's path
        ax.plot(self._column_init-0.15, self._line_init,'>',markersize=10, markeredgecolor=(0,0,0), markerfacecolor='r', markeredgewidth=0)
        
        column_exit_offset = 0
        line_exit_offset = 0
        
        if self._column_goal == self._columns:
            column_exit_offset = 0.15
            exit_symbol = '>'
        elif self._column_goal == 0:
            column_exit_offset = -0.15
            exit_symbol = '<'
        elif  self._line_goal == self._lines:
            line_exit_offset = 0.15
            exit_symbol = '^'
        else:
            line_exit_offset = -0.15
            exit_symbol = 'v'
        # Draw the exit of the puzzle: red if it is on a path, black otherwise
        if self._dots[self._line_goal][self._column_goal] == 0:
            ax.plot(self._column_goal+column_exit_offset, self._line_goal+line_exit_offset, exit_symbol, markersize=10, markeredgecolor=(0,0,0), markerfacecolor='k', markeredgewidth=0)
        else:
            ax.plot(self._column_goal+column_exit_offset, self._line_goal+line_exit_offset, exit_symbol, markersize=10, markeredgecolor=(0,0,0), markerfacecolor='r', markeredgewidth=0)

        if save_file:
            plt.savefig(filename)
            plt.close()
        else:
            plt.show()
        
    def __successor_bfs(self, state):
        """
        Successor function use in the Breadth-first search (BFS) performed to validate a solution.
        An adjacent cell c' is amongst the successors of cell c if there is no segment (v_seg or h_seg) 
        separating cells c and c'.
        
        This method is meant to be called only from within GameState  
        """
        children = []
        # move up
        if state[0]+1 < self._cells.shape[0] and self._h_seg[state[0]+1][state[1]] == 0:
            children.append((state[0]+1, state[1]))
        # move down
        if state[0] > 0 and self._h_seg[state[0]][state[1]] == 0:
            children.append((state[0]-1, state[1]))
        # move right
        if state[1]+1 < self._cells.shape[1] and self._v_seg[state[0]][state[1]+1] == 0:
            children.append((state[0], state[1]+1))
        # move left
        if state[1] > 0 and self._v_seg[state[0]][state[1]] == 0:
            children.append((state[0], state[1]-1))
        return children
    
    
    def successors_parent_pruning(self, op):
        """
        Successor function used by planners trying to solve the puzzle. The method returns
        a list with legal actions for the state. The valid actions for the domain are {U, D, L, R}.
        
        The tip of the snake can move to an adjacent intersection in the grid as long as
        that intersection isn't already occupied by the snake and the intersection is valid
        (i.e., it isn't negative or larger than the grid size)
        
        op is the action taken at the parent; used here to perform parent pruning
        
        Mapping of actions:
        0 - Up
        1 - Down
        2 - Right
        3 - Left
        """
        actions = []
#         if self.has_tip_reached_goal():
#             return actions
        #moving up
        if op != 1 and self._line_tip + 1 < self._dots.shape[0] and self._v_seg[self._line_tip][self._column_tip] == 0 and self._dots[self._line_tip+1][self._column_tip] == 0:
            actions.append(0)
        #moving down
        if op != 0 and self._line_tip - 1 >= 0 and self._v_seg[self._line_tip-1][self._column_tip] == 0 and self._dots[self._line_tip-1][self._column_tip] == 0:
            actions.append(1)
        #moving right
        if op != 3 and self._column_tip + 1 < self._dots.shape[1] and self._h_seg[self._line_tip][self._column_tip] == 0 and self._dots[self._line_tip][self._column_tip+1] == 0:
            actions.append(2)
        #moving left
        if op != 2 and self._column_tip - 1 >= 0 and self._h_seg[self._line_tip][self._column_tip-1] == 0 and self._dots[self._line_tip][self._column_tip-1] == 0:
            actions.append(3)
        return actions
        
    def successors(self):
        """
        Successor function used by planners trying to solve the puzzle. The method returns
        a list with legal actions for the state. The valid actions for the domain are {U, D, L, R}.
        
        The tip of the snake can move to an adjacent intersection in the grid as long as
        that intersection isn't already occupied by the snake and the intersection is valid
        (i.e., it isn't negative or larger than the grid size)
        
        Mapping of actions:
        0 - Up
        1 - Down
        2 - Right
        3 - Left
        """
        actions = []
#         if self.has_tip_reached_goal():
#             return actions
        #moving up
        if self._line_tip + 1 < self._dots.shape[0] and self._v_seg[self._line_tip][self._column_tip] == 0 and self._dots[self._line_tip+1][self._column_tip] == 0:
            actions.append(0)
        #moving down
        if self._line_tip - 1 >= 0 and self._v_seg[self._line_tip-1][self._column_tip] == 0 and self._dots[self._line_tip-1][self._column_tip] == 0:
            actions.append(1)
        #moving right
        if self._column_tip + 1 < self._dots.shape[1] and self._h_seg[self._line_tip][self._column_tip] == 0 and self._dots[self._line_tip][self._column_tip+1] == 0:
            actions.append(2)
        #moving left
        if self._column_tip - 1 >= 0 and self._h_seg[self._line_tip][self._column_tip-1] == 0 and self._dots[self._line_tip][self._column_tip-1] == 0:
            actions.append(3)
        return actions
    
    def apply_action(self, a):
        """
        Applies a given action to the state. It modifies the segments visited by the snake (v_seg and h_seg),
        the intersections (dots), and the tip of the snake. 
        """
        #moving up
        if a == 0:
            self._v_seg[self._line_tip][self._column_tip] = 1
            self._dots[self._line_tip+1][self._column_tip] = 1
            self._line_tip += 1
        #moving down
        if a == 1:
            self._v_seg[self._line_tip-1][self._column_tip] = 1
            self._dots[self._line_tip-1][self._column_tip] = 1
            self._line_tip -= 1 
        #moving right
        if a == 2:
            self._h_seg[self._line_tip][self._column_tip] = 1
            self._dots[self._line_tip][self._column_tip+1] = 1
            self._column_tip += 1
        #moving left
        if a == 3:
            self._h_seg[self._line_tip][self._column_tip-1] = 1
            self._dots[self._line_tip][self._column_tip-1] = 1
            self._column_tip -= 1
                
    def has_tip_reached_goal(self):
        """
        Verifies whether the snake has reached the goal position. Note this is not a goal
        test. The goal test is performed by method is_solution, which uses has_tip_reached_goal
        as part of the verification.
        """
        return self._line_tip == self._line_goal and self._column_tip == self._column_goal
    
    def random_path(self):
        """
        Generates a path through a random walk, mostly used for debugging purposes. The random
        walk finishes as soon as the tip reaches the goal position or there are not more legal actions. 
        """
        self.reset()

        actions = self.successors()        
        while len(actions) > 0:
            a = random.randint(0, len(actions) - 1)
            self.apply_action(actions[a])
            # If the tip of random walk reached the goal, then stop 
            if self.has_tip_reached_goal():
                return
            actions = self.successors()
            
    def is_solution(self):
        """
        Verifies whether the state's path represents a valid solution. This is performed by verifying the following
        (1) the tip of the snake is at the goal position
        (2) a bullet of color c1 cannot reach a bullet of color c2 through a BFS search.
        
        The BFS uses the cells (line and column) as states and verifies whether cells with a bullet of a given color
        can only reach (and be reached) by cells with bullets of the same color (or of the neutral color, denoted as zero in this implementation)
        """
        if not self.has_tip_reached_goal():
            return False
        
        non_visited_states = set()
        current_color = 0
        closed_bfs = np.zeros((self._lines, self._columns))
        for i in range(self._cells.shape[0]):
            for j in range(self._cells.shape[1]):
                non_visited_states.add((i, j))
                
        while len(non_visited_states) != 0:
            root = non_visited_states.pop()
            # If root of new BFS search was already visited, then go to the next state
            if closed_bfs[root[0]][root[1]] == 1:
                continue
            current_color = self._cells[root[0]][root[1]]
            
            open_bfs = deque()
            open_bfs.append(root)
            closed_bfs[root[0]][root[1]] = 1
            while len(open_bfs) != 0:
                # remove first state from queue
                state = open_bfs.popleft()
                children = self.__successor_bfs(state)
                for c in children:
                    # If c is a duplicate, then continue with the next child
                    if closed_bfs[c[0]][c[1]] == 1:
                        continue
                    # If c's color isn't neutral (zero) and it is different from current_color, then state isn't a soution
                    if current_color != 0 and self._cells[c[0]][c[1]] != 0 and self._cells[c[0]][c[1]] != current_color:
                        return False
                    # If current_color is neutral (zero) and c's color isn't, then attribute c's color to current_color
                    if self._cells[c[0]][c[1]] != 0:
                        current_color = self._cells[c[0]][c[1]]
                    # Add c to BFS's open list
                    open_bfs.append(c)
                    # mark state c as visited
                    closed_bfs[c[0]][c[1]] = 1
        return True 
    
    def partition_cells(self):
        """
        Returns a list of list of cells, with one list of cells for each region defined by the snake. 
        
        This method assumes there is a valid snake (i.e., from start to goal) in the state.
        
        The partition of cells into regions is performed by running BFS several times, one for each region
        """
        regions = []
        non_visited_states = set()
        closed_bfs = np.zeros((self._lines, self._columns))
        for i in range(self._cells.shape[0]):
            for j in range(self._cells.shape[1]):
                non_visited_states.add((i, j))
                
        while len(non_visited_states) != 0:
            root = non_visited_states.pop()
            # If root of new BFS search was already visited, then go to the next state
            if closed_bfs[root[0]][root[1]] == 1:
                continue
            region = []            
            open_bfs = deque()
            open_bfs.append(root)
            closed_bfs[root[0]][root[1]] = 1
            while len(open_bfs) != 0:
                # remove first state from queue
                state = open_bfs.popleft()
                # adding state expanded into the current region
                region.append(state)
                children = self.__successor_bfs(state)
                for c in children:
                    # If c is a duplicate, then continue with the next child
                    if closed_bfs[c[0]][c[1]] == 1:
                        continue
                    # Add c to BFS's open list
                    open_bfs.append(c)
                    # mark state c as visited
                    closed_bfs[c[0]][c[1]] = 1
            regions.append(region)
        return regions 
    
    def save_state(self, filename):
        """
        Saves the state into filename. It doesn't save the path in the state.
        
        Here's an example of a file of a puzzle with 5 lines and 4 columns, with the snake
        starting position at 0, 0 and finishing position at 5, 4. The state has three bullets,
        each with a different color (1, 2, and 6). The bullets are located at (0,0), (2,2), (3, 3), 
        and (2, 0). Here the first number means the line and the second the column. 
        
        Size: 5 4
        Init: 0 0
        Goal: 5 4
        Colors: |0 0 1|2 2 2|3 3 6|2 0 1
        
        """
        file = open(filename, 'w')
        file.write('Size: ' + str(self._lines) + ' ' + str(self._columns) + '\n')
        file.write('Init: ' + str(self._line_init) + ' ' + str(self._column_init) + '\n')
        file.write('Goal: ' + str(self._line_goal) + ' ' + str(self._column_goal) + '\n')
        
        has_colors = False
        for i in range(self._cells.shape[0]):
            for j in range(self._cells.shape[1]):
                if self._cells[i][j] != 0:
                    has_colors = True
                    break
            if has_colors:
                break
        
        if has_colors:
            file.write('Colors: ')
            for i in range(self._cells.shape[0]):
                for j in range(self._cells.shape[1]):
                    if self._cells[i][j] != 0:
                        file.write('|' + str(i) + ' ' + str(j) + ' ' + str(int(self._cells[i][j])))
            file.close()
            
    def convert_2_dict(self):
        """
        Converts a state to a dictionary.
        
        Here's an example of a file of a puzzle with 5 lines and 4 columns, with the snake
        starting position at 0, 0 and finishing position at line 5 and column 4. The state has three bullets,
        each with a different color (1, 2, and 6). The bullets are located at (0,0), (2,2), (3, 3), 
        and (2, 0). Here the first number means the line and the second the column. 
        
        size: [4, 5]
        startPosition: [0, 0]
        endPosition: [4, 5, 1] [lineCol, lineRow, tipDirection<-North=1, E=2, S=3, W=4]
        filledSquares: [
            [0 0 1], [2 2 2], [3 3 6], [2 0 1]
            ]
        
        """
        
        data = {}
        data['size'] = [self._columns, self._lines]
        data['startPosition'] = [self._column_init, self._line_init]
        
        north = 1
        east  = 2
        south = 3
        west  = 4
        exit_direction = south
        
        if self._column_goal == self._columns:
            exit_direction = east
        elif self._column_goal == 0:
            exit_direction = west
        elif  self._line_goal == self._lines:
            exit_direction = north
        
        data['endPosition'] = [self._column_goal, self._line_goal, exit_direction]
        
        filled_squares = []
        for i in range(self._cells.shape[0]):
            for j in range(self._cells.shape[1]):
                if self._cells[i][j] != 0:
                    filled_squares.append([j+1, i+1, int(self._cells[i][j]) - 1])
        data['filledSquares'] = filled_squares
        
        return data
            
            
    def get_name(self):
        return self._filename
    
    def set_solution_depth(self, depth):
        self._solution_depth = depth
        
    def get_solution_depth(self):
        return self._solution_depth
    
    def copy(self):
        return copy.deepcopy(self)
    
#     def copy(self):
#         copy_state = WitnessState(self._lines, 
#                                   self._columns,
#                                   self._line_init,
#                                   self._column_init,
#                                   self._line_goal,
#                                   self._column_goal,
#                                   self._max_lines,
#                                   self._max_columns)
#         
#         copy_state._line_tip = self._line_tip
#         copy_state._column_tip = self._column_tip
#         copy_state._dots[self._line_tip][self._column_tip] = 1
#         
#         copy_state._v_seg = self._v_seg.copy()
#         copy_state._h_seg = self._h_seg.copy()
#         copy_state._cells = self._cells.copy()
#         
#         return copy_state    

    def read_state_from_string(self, puzzle):
        """
        Reads a puzzle from a string. It follows the format speficied in method save_state of this class.
        """
        for s in puzzle:
            s = s.replace('\n', '')
            if 'Size: ' in s:
                values = s.replace('Size: ', '').split(' ')
                self._lines = int(values[0])
                self._columns = int(values[1])
            if 'Init: ' in s:
                values = s.replace('Init: ', '').split(' ')
                self._line_init = int(values[0])
                self._column_init = int(values[1])
            if 'Goal: ' in s:
                values = s.replace('Goal: ', '').split(' ')
                self._line_goal = int(values[0]) 
                self._column_goal = int(values[1])
                self.__init_structures()
            if 'Colors: ' in s:
                values = s.replace('Colors: |', '').split('|')
                for t in values:
                    numbers = t.split(' ')
                    self._cells[int(numbers[0])][int(numbers[1])] = int(numbers[2])

        
    def read_state(self, filename):
        """
        Reads a puzzle from a file. It follows the format speficied in method save_state of this class.
        """
        file = open(filename, 'r')
        
        if '/' in filename:
            self._filename = filename[filename.index('/')+1:len(filename)]
        else:
            self._filename = filename
        puzzle = file.read().split('\n')
        for s in puzzle:
            if 'Size: ' in s:
                values = s.replace('Size: ', '').split(' ')
                self._lines = int(values[0])
                self._columns = int(values[1])
            if 'Init: ' in s:
                values = s.replace('Init: ', '').split(' ')
                self._line_init = int(values[0])
                self._column_init = int(values[1])
            if 'Goal: ' in s:
                values = s.replace('Goal: ', '').split(' ')
                self._line_goal = int(values[0]) 
                self._column_goal = int(values[1])
                self.__init_structures()
            if 'Colors: ' in s:
                values = s.replace('Colors: |', '').split('|')
                for t in values:
                    numbers = t.split(' ')
                    self._cells[int(numbers[0])][int(numbers[1])] = int(numbers[2])
                    
    def heuristic_value(self):
        return abs(self._column_tip - self._column_goal) + abs(self._line_tip - self._line_goal)
