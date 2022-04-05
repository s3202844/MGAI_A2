#!/usr/bin/env python3

import time
import copy
import random
import botbowl
import numpy as np

from botbowl import ActionType
from botbowl.core import Square, Action

'''
Important functions:

game.get_team_side(self.my_team)
game.get_ball()
game.get_player_at(Square(x, y))
game.get_catcher(Square(x, y))
game.get_ball_carrier()
'''


class Node:
    def __init__(self, node_id=None, parent=None, action=None):
        self.node_id = node_id
        self.parent = parent
        self.action = action
        self.children = []
        self.children_id = []
        self.evaluations = []
        self.off_pool = []
        self.max_pool = 30

    def init_off_pool(self, available_actions):
        for action_choice in available_actions:
            if action_choice.action_type == ActionType.PLACE_PLAYER:
                continue
            for player in action_choice.players:
                new_action = Action(action_choice.action_type, player=player)
                new_node = Node(new_action.__repr__(), self, new_action)
                self.off_pool.append(new_node)
            for position in action_choice.positions:
                new_action = Action(action_choice.action_type, position=position)
                new_node = Node(new_action.__repr__(), self, new_action)
                self.off_pool.append(new_node)
            if len(action_choice.players) == len(action_choice.positions) == 0:
                new_action = Action(action_choice.action_type)
                new_node = Node(new_action.__repr__(), self, new_action)
                self.off_pool.append(new_node)
        if len(self.off_pool) > self.max_pool:
            random.shuffle(self.off_pool)
            # for o in self.off_pool:
            #     print(o.node_id)
            self.off_pool = self.off_pool[:self.max_pool]
            

    def num_visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        return np.average(self.evaluations)


class MyMctsBot(botbowl.Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)
        # hyperparameters
        self.steps = 0
        self.depth = 5
        self.epsilon = 0.5
        self.C = 2**0.5

    def new_game(self, game, team):
        self.my_team = team
        self.opp_team = game.get_opp_team(team)

    def _selection(self, node):
        # print("select")
        best_node = None
        for child in node.children:
            if best_node == None or child.score() > best_node.score():
                best_node = child
        return best_node

    def _expansion(self, node):
        # print("expansion")
        # while True:
        #     ind = np.random.randint(0, len(available_actions))
        #     if available_actions[ind].action_type != ActionType.PLACE_PLAYER:
        #         break
        # action_choice = available_actions[ind]
        # if len(action_choice.players) > 0:
        #     ind = np.random.randint(0, len(action_choice.players))
        #     player = action_choice.players[ind]
        #     new_action = Action(action_choice.action_type, player=player)
        # elif len(action_choice.positions) > 0:
        #     ind = np.random.randint(0, len(action_choice.positions))
        #     position = action_choice.positions[ind]
        #     new_action = Action(action_choice.action_type, position=position)
        # else:
        #     new_action = Action(action_choice.action_type)
        # if new_action.__repr__() in node.children_id:
        #     ind = node.children_id.index(new_action.__repr__())
        #     return node.children[ind]
        # new_node = Node(new_action.__repr__(), node, new_action)
        # node.children += [new_node]
        # node.children_id += [new_node.node_id]
        ind = np.random.randint(0, len(node.off_pool))
        res = node.off_pool.pop(ind)
        node.children.append(res)
        node.children_id.append(res.node_id)
        return res


    def act(self, game):
        game_copy = copy.deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True

        root_step = game_copy.get_step()
        root_node = Node()

        t = time.time()
        available_actions = game_copy.get_available_actions()
        root_node.init_off_pool(available_actions)
        budget = len(root_node.off_pool) * 2
        for _ in range(budget):
            prob = np.random.rand()
            if (prob > self.epsilon and len(root_node.children) > 0) or \
                len(root_node.off_pool) == 0:
                curr = self._selection(root_node)
            else:
                curr = self._expansion(root_node)
            print(curr.node_id)
        # if self.steps < 7:
        #     for a in root_node.off_pool:
        #         print(a.node_id)
            # print(root_node.off_pool)
        # for m in range(10):
        #     pre = root_node
        #     for n in range(self.depth):
        #         available_actions = game_copy.get_available_actions()
        #         # print(m, n, available_actions)
        #         prob = np.random.rand()
        #         if prob > self.epsilon and len(pre.children) > 0:
        #             curr = self._selection(pre)
        #         else:
        #             curr = self._expansion(pre, available_actions)
        #             if curr == None:
        #                 curr = self._selection(pre)
        #         try:
        #             game_copy.step(curr.action)
        #             while not game.state.game_over and len(game.state.available_actions) == 0:
        #                 game_copy.step()
        #             score = self._evaluate(game)
        #             curr.visit(score)
        #             pre, curr = curr, None
        #         except:
        #             break
        #     game_copy.revert(root_step)


        for action_choice in game_copy.get_available_actions():
            if action_choice.action_type == botbowl.ActionType.PLACE_PLAYER:
                continue
            for player in action_choice.players:
                temp_action = Action(action_choice.action_type, player=player)
                root_node.children.append(Node(temp_action.__repr__(), root_node, temp_action))
            for position in action_choice.positions:
                temp_action = Action(action_choice.action_type, position=position)
                root_node.children.append(Node(temp_action.__repr__(), root_node, temp_action))
            if len(action_choice.players) == len(action_choice.positions) == 0:
                temp_action = Action(action_choice.action_type)
                root_node.children.append(Node(temp_action.__repr__(), root_node, temp_action))

        best_node = None
        print(f"Evaluating {len(root_node.children)} nodes")
        for node in root_node.children:
            # if self.steps < 7:
            #     print(node.action.__repr__())
            game_copy.step(node.action)
            while not game.state.game_over and len(game.state.available_actions) == 0:
                game_copy.step()
            score = self._evaluate(game)
            node.visit(score)
            # print(f"{node.action.action_type}: {node.score()}")
            if best_node is None or node.score() > best_node.score():
                best_node = node

            game_copy.revert(root_step)
        best_node = None
        for child in root_node.children:
            if best_node == None or child.score() > best_node.score():
                best_node = child
        self.steps += 1

        print(f"{best_node.action.action_type} selected in {time.time() - t} seconds")

        return best_node.action

    def _evaluate(self, game):
        return np.random.rand()

    def end_game(self, game):
        print(self.steps)
        # pass


botbowl.register_bot("MCTS", MyMctsBot)

if __name__ == "__main__":
    # Load configurations, rules, arena and teams
    config = botbowl.load_config("web")
    ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)
    arena = botbowl.load_arena(config.arena)
    home_team = botbowl.load_team_by_filename("human", ruleset)
    away_team = botbowl.load_team_by_filename("human", ruleset)

    # Play 10 games
    num_games = 1
    wins = 0
    tds = 0
    for i in range(num_games):
        home_agent = botbowl.make_bot("MCTS")
        away_agent = botbowl.make_bot("random")
        home_agent.name = "MCTS Bot"
        away_agent.name = "Random Bot"
        config.debug_mode = False
        game = botbowl.Game(i, home_team, away_team, home_agent, away_agent,
                            config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        start = time.time()
        game.init()
        end = time.time()
        print(end - start)

        wins += 1 if game.get_winning_team() is game.state.home_team else 0
        tds += game.state.home_team.state.score
    print(f"won {wins}/{num_games}")
    print(f"Own TDs per game={tds/num_games}")
