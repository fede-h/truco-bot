""" Benchmark module that simulates matches between two Agents n times."""
import random

from truco_bot.env.engine import TrucoHandEnv
from truco_bot.eval.arena import play_duplicate_match


def create_agent(agent_name:str):
    if agent_name == "random":
        from truco_bot.agents.baselines.random import RandomAgent
        return RandomAgent()
    elif agent_name == "equity":
        from truco_bot.agents.baselines.equity import EquityAgent
        return EquityAgent()
    elif agent_name == "heuristic":
        from truco_bot.agents.baselines.heuristic import HeuristicAgent
        return HeuristicAgent()
    else:
        raise ValueError(f"Unknown agent: {agent_name}")


def run_matches(agent_a, agent_b, n:int, seed:int=42) -> float:
    random.seed(seed)
    env = TrucoHandEnv()
    env.reset(seed=seed)

    results:list[int] = []
    for _ in range (n):
        seed_game = random.randint(1, n*100)
        results.append(play_duplicate_match(agent_a, agent_b, env, seed_game))
    
    return (sum(results) / len(results))


if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser()
    parser.add_argument('--p0', help='First Agent')
    parser.add_argument('--p1', help='Second Agent')
    parser.add_argument('--n', help='Number of matches played')
    args = parser.parse_args()

    p0 = create_agent(args.p0)
    p1 = create_agent(args.p1)

    print(run_matches(p0, p1, int(args.n)))