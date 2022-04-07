#!/usr/bin/env python3

from logging import root
import time
import copy
import math
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
        self.C = 2**0.5

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
            self.off_pool = self.off_pool[:self.max_pool]


    def num_visits(self):
        return len(self.evaluations)

    def visit(self, score):
        self.evaluations.append(score)

    def score(self):
        T = 0
        for child in self.parent.children:
            T += child.num_visits()
        w = np.average(self.evaluations)
        # return w
        return w + self.C*(math.log(T)/self.num_visits())**0.5


class MyMctsBot(botbowl.Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.rnd = np.random.RandomState(seed)
        # hyperparameters
        self.steps = 0
        self.depth = 2
        self.epsilon = 0.3
        self.min_budget = 10

    def new_game(self, game, team):
        self.my_team = team
        self.opp_team = game.get_opp_team(team)

    def _selection(self, node):
        best_node = None
        for child in node.children:
            if best_node == None or child.score() > best_node.score():
                best_node = child
        return best_node

    def _expansion(self, node):
        ind = np.random.randint(0, len(node.off_pool))
        res = node.off_pool.pop(ind)
        node.children.append(res)
        node.children_id.append(res.node_id)
        return res

    def _simulation(self, game_copy):
        for _ in range(self.depth):
            flag = True
            while True:
                if len(game_copy.get_available_actions()) == 0:
                    flag = False
                    break
                action_choice = self.rnd.choice(game_copy.get_available_actions())
                if action_choice.action_type != botbowl.ActionType.PLACE_PLAYER:
                    break
            if flag:
                position = self.rnd.choice(action_choice.positions) if len(
                    action_choice.positions) > 0 else None
                player = self.rnd.choice(action_choice.players) if len(
                    action_choice.players) > 0 else None
                action = botbowl.Action(action_choice.action_type,
                                        position=position, player=player)
                game_copy.step(action)
            else:
                break

    def _backpropagation(self, game_copy, root_step, node):
        score = self._evaluate(game_copy, root_step)
        node.visit(score)
        game_copy.revert(root_step)
    
    def _brief_state(self, game):
        print(f"Step {self.steps}:")
        if game.get_ball_carrier() in self.my_team.players:
            print("Ball is carried by: My team")
        elif game.get_ball_carrier() in self.opp_team.players:
            print("Ball is carried by: Op team")
        else:
            print("Ball is carried by: Nobody")
        ball = game.get_ball()
        if ball != None:
            min_distance = 26
            for player in self.my_team.players:
                if player.position != None:
                    min_distance = min(min_distance, \
                        self._distance(ball.position, player.position))
            print(f"My team. Squares to the ball: {min_distance}")
            min_distance = 26
            for player in self.opp_team.players:
                if player.position != None:
                    min_distance = min(min_distance, \
                        self._distance(ball.position, player.position))
            print(f"Op team. Squares to the ball: {min_distance}")
        else:
            print("My team. Squares to the ball: None")
            print("Op team. Squares to the ball: None")

    def act(self, game):
        self._brief_state(game)
        game_copy = copy.deepcopy(game)
        game_copy.enable_forward_model()
        game_copy.home_agent.human = True
        game_copy.away_agent.human = True

        root_step = game_copy.get_step()
        root_node = Node()

        t = time.time()
        available_actions = game_copy.get_available_actions()
        if len(available_actions) == 0:
            return None
        root_node.init_off_pool(available_actions)
        budget = max(self.min_budget, len(root_node.off_pool) * 4)
        for _ in range(budget):
            prob = np.random.rand()
            if (prob > self.epsilon and len(root_node.children) > 0) or \
                len(root_node.off_pool) == 0:
                curr = self._selection(root_node)
            else:
                curr = self._expansion(root_node)
            game_copy.step(curr.action)
            self._simulation(game_copy)
            self._backpropagation(game_copy, root_step, curr)
        self.steps += 1
        res = self._selection(root_node)
        print(res.score())
        return res.action

    def _evaluate(self, game, root_step):
        grab_ball = False
        distance2target = []
        steps = game.revert(root_step)
        ball = game.get_ball()
        if ball != None:
            carrier = game.get_ball_carrier()
            if carrier != None and carrier in self.my_team.players:
                target = None
                grab_ball = True
            else:
                target = ball.position
        else:
            target = Square(8, 4)
        for i in range(len(self.my_team.players)):
            player = self.my_team.players[i]
            p = game.get_player(player.player_id)
            if p.position != None:
                if target == None:
                    distance2target += [p.position.x]
                else:
                    distance2target += [self._distance(target, p.position)]
            else:
                distance2target += [None]
        game.forward(steps)
        ball = game.get_ball()
        if ball != None:
            carrier = game.get_ball_carrier()
            if carrier != None and carrier in self.my_team.players:
                grab_ball = True if not grab_ball else False
            else:
                grab_ball = False
        else:
            grab_ball = False
        for i in range(len(distance2target)):
            player = self.my_team.players[i]
            p = game.get_player(player.player_id)
            if p.position != None and distance2target[i] != None:
                if target == None:
                    distance2target[i] -= p.position.x
                else:
                    distance2target[i] -= self._distance(target, p.position)
            else:
                distance2target[i] = None
        sum_delta = 0
        for e in distance2target:
            sum_delta += e if e != None else 0
        sum_delta += 10 if grab_ball else 0
        return sum_delta
    
    def _distance(self, pos0, pos1):
        dx = abs(pos0.x-pos1.x)
        dy = abs(pos0.y-pos1.y)
        return (dx**2+dy**2)**0.5

    def end_game(self, game):
        print(f"{self.steps} actions used in totoal.")


botbowl.register_bot("MCTS", MyMctsBot)

if __name__ == "__main__":
    config = botbowl.load_config("web")
    ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)
    arena = botbowl.load_arena(config.arena)
    home_team = botbowl.load_team_by_filename("human", ruleset)
    away_team = botbowl.load_team_by_filename("human", ruleset)

    home_agent = botbowl.make_bot("MCTS")
    away_agent = botbowl.make_bot("random")
    home_agent.name = "MCTS Bot"
    away_agent.name = "Random Bot"
    config.debug_mode = False
    game = botbowl.Game(0, home_team, away_team, home_agent, away_agent,
                        config, arena=arena, ruleset=ruleset)
    game.config.fast_mode = True

    start = time.time()
    game.init()
    end = time.time()
    print(end - start)

