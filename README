Implementation of the algorithms described in "Policy-Guided Heuristic Search with Guarantees" 
by L. Orseau and L. Lelis, published at AAAI'21.

PHS is dubbed BFSLevin (see src/bfs_levin.py). The same class is used to implement LTS, the 
tree search algorithm that uses a policy to guide search (see the paper "Single-Agent Policy 
Tree Search with Guarantees" by L. Orseau, L. Lelis, T. Lattimore, and T. Weber for details).

PHS can be trained for a small set of The Witness puzzles with the following command:

src/main.py --learned-heuristic 
			-a LevinStar 
			-l LevinLoss 
			-m model_test_witness 
			-p problems/witness/puzzles_3x3/ 
			-b 2000 
			-d Witness 
			--learn

The program will save a trained neural model with the name model_test_witness, which can be
used solve other instances with the following command (here we are solving the same set of
instances used to train the model).

python3 src/main.py --learned-heuristic 
					-a LevinStar 
					-l LevinLoss 
					-m model_test_witness 
					-p problems/witness/puzzles_3x3/ 
					-b 2000 
					-d Witness

Here are the options of search algorithms implemented:

AStar (A*, see file src/search/a_star.py)
GBFS (Greedy-Best First Search, see file src/search/gbfs.py)
PUCT (PUCT, see file src/search/puct.py)
LevinStar (PHS, see file src/search/bfs_levin.py)
Levin (LTS, see file src/search/bfs_levin.py)

The instances used to train and test the models are availble in the folder 'problems':

Sokoban
	Train: problems/sokoban/train_50000
	Test: problems/sokoban/train_10000

Sliding-Tile Puzzle
	Training: problems/stp/puzzles_5x5_train
	Test: problems/stp/puzzles_5x5_test
	
Witness
	Training: problems/witness/puzzles_4x4_50k_train
	Test: problems/witness/puzzles_4x4_50k_test