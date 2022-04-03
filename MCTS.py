#!/usr/bin/env python3

import time
import copy
import botbowl
import numpy as np

from botbowl.core import Square

'''
Important functions:

game.get_team_side(self.my_team)
game.get_ball()
game.get_player_at(Square(x, y))
game.get_catcher(Square(x, y))
game.get_ball_carrier()
'''


class Node():

    def __init__(self, parent):
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0

    def isSameAction(self, act0, act1):
        return act0.__repr__() == act1.__repr__()


class MyMctsBot(botbowl.Agent):

    def __init__(self, name, seed=None):
        super().__init__(name)
        self.my_team = None
        self.my_tree = Node(None)
        self.opp_tree = self.my_tree
        self.rnd = np.random.RandomState(seed)

    def new_game(self, game, team):
        self.my_team = team
        self.opp_team = game.get_opp_team(team)
        self.simulation = copy.deepcopy(game)

    def act(self, game):
        # Select a random action type
        while True:
            action_choice = self.rnd.choice(game.state.available_actions)
            # Ignore PLACE_PLAYER actions
            if action_choice.action_type != botbowl.ActionType.PLACE_PLAYER:
                break

        # Select a random position and/or player
        position = self.rnd.choice(action_choice.positions) if len(
            action_choice.positions) > 0 else None
        player = self.rnd.choice(action_choice.players) if len(
            action_choice.players) > 0 else None

        # Make action object
        action = botbowl.Action(action_choice.action_type,
                                position=position, player=player)

        # Return action to the framework
        return action

    def end_game(self, game):
        print(game.get_ball())


botbowl.register_bot("MCTS", MyMctsBot)

if __name__ == "__main__":
    # Load configurations, rules, arena and teams
    config = botbowl.load_config("web")
    ruleset = botbowl.load_rule_set(config.ruleset, all_rules=False)
    arena = botbowl.load_arena(config.arena)
    home = botbowl.load_team_by_filename("human", ruleset)
    away = botbowl.load_team_by_filename("human", ruleset)

    # Play 10 games
    num_games = 10
    wins = 0
    tds = 0
    for i in range(num_games):
        home_agent = botbowl.make_bot("MCTS")
        away_agent = botbowl.make_bot("random")
        home_agent.name = "MCTS Bot"
        away_agent.name = "Random Bot"
        config.debug_mode = False
        game = botbowl.Game(i, home, away, home_agent,
                            away_agent, config, arena=arena, ruleset=ruleset)
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
