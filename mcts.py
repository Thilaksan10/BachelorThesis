# packages
import math
import random
from copy import deepcopy

# tree node class definition
class TreeNode:
    # class constructor (create tree node class istance)
    def __init__(self, scheduler, action, parent) -> None:
        # init associated board state
        self.scheduler = deepcopy(scheduler)

        # init nodes terminal flag
        if self.scheduler.hyper_period_reached() or self.scheduler.calculate_scores() <= 0:
            # we have terminal node
            self.is_terminal = True
        else:
            # we have a non-terminal node
            self.is_terminal = False

        # set fully expanded flag
        self.is_fully_expanded = self.is_terminal
        
        # init parent node if available
        self.parent = parent

        # init action let to this node
        self.action = action

        # init the number of node visits
        self.visits = 0

        # init the total score of the node
        self.score = 0

        # init current node's children
        self.children = {}

# MCTS class definition
class MCTS:

    def __init__(self) -> None:
        self.root = None

    # search for the best move in the current position
    def search(self, initial_state, current_node=None):
        # create root node
        if current_node:
            self.root = current_node
            current_node.parent = None
        else:
            self.root = TreeNode(initial_state, 0, None)

        # walk through 10 iterations
        for _ in range(10):
            # select a node (selection phase)
            node =  self.select(self.root)

            # score current node (simulation phase)
            score = 0
            simulations = 1
            for _ in range(simulations):
                score += self.rollout(node)
            score /= simulations
            
            # backpropagate results
            self.backpropagate(node, score)
        
        if not self.root.is_terminal:
            # pick up the best move in the current position
            return self.get_best_move(self.root, 0)

    
    # select most promising node
    def select(self, node):
        # make sure that we're dealing with non-terminal nodes
        while not node.is_terminal:
            # case where the node is fully expanded 
            if node.is_fully_expanded:
                node = self.get_best_move(node, 2)
            # case where the node is not fully expanded
            else:
                return self.expand(node)
        return node
        

    def expand(self, node):
        # generate legal states (moves) for the given node
        states = deepcopy(node).scheduler.generate_states()
        # loop over generated states (moves)
        for state in states:
            # make sure that current state (move) is not present in child node
            if state[0].to_string() not in node.children:
                # create a new node 
                new_node = TreeNode(state[0], state[1], node)
                # add child node to parent's node children list (dict)
                node.children[state[0].to_string()] = new_node
                # case when node is fully expanded
                if len(states) == len(node.children):
                    node.is_fully_expanded = True

                # return newly created node
                return new_node

        # debugging
        print(f'Should not get here !!!')

    # simulate the schedule via making random moves 
    def rollout(self, node):
    
        # make copy of scheduler to simulate on copy
        scheduler = deepcopy(node.scheduler)
        # init score
        score = scheduler.calculate_scores()
        # update preemption status of all ready jobs
        for job in scheduler.ready_list:
            job.preempt = False

        # init iteration count and coefficient of iteration
        iterations = 0
        coefficient = 1
        
        # simulate until hyperperiod is reached or deadline was missed
        while not scheduler.hyper_period_reached() and not score <= 0:
            # generate possible next states and simulate by picking random state
            states = scheduler.generate_states()
            scheduler = random.choice(states)[0]

            iterations += 1
            # add decayed score of current state to simulation score
            score += (scheduler.calculate_scores()/coefficient)
            coefficient += 0.1
            
        return score


    # backpropagate the number of visits and score up to the root node
    def backpropagate(self, node, score):
        # update node's visit count and score up to root node
        while node is not None:
            # update node's visits
            node.visits += 1
            
            # update node's score
            node.score += score 
           
            # set node to parent
            node = node.parent


    # select the best node basing on UCB1 formula
    def get_best_move(self, node, exploration_constant):
        # define best score & best moves
        best_score = float('-inf')
        best_moves = []
        print(len(node.children))

        # loop over child nodes
        for child_node in node.children.values():
            # get move score using UCT formula
            move_score = child_node.score / child_node.visits + exploration_constant * math.sqrt(math.log(node.visits / child_node.visits))
            # better move has been found
            if move_score > best_score:
                best_score = move_score
                best_moves = [child_node]

            # found as good as already available
            elif move_score == best_score:
                best_moves.append(child_node)

        # return one of the best moves randomly
        print(best_moves)
        return random.choice(best_moves)
