"""UCT (Upper Confidence Trees) search for optimal pitch sequencing.

Algorithm per iteration:
  1. Selection   — descend tree via UCB1 until an unexpanded or terminal node
  2. Expansion   — simulate one untried action, create a child node
  3. Rollout     — simulate randomly to a terminal outcome from the child
  4. Backprop    — update visit counts and value_sum along the path

Value convention (inherited from MCTSNode):
  value_sum stores cumulative (-run_value), so higher Q = better for pitcher.
"""

import random

import torch

from mcts.node      import MCTSNode
from mcts.simulator import PitchSimulator
from mcts.state     import AtBatState


def mcts_search(
    root:            MCTSNode,
    simulator:       PitchSimulator,
    pitcher_tensors: dict,
    pitcher_id:      int,
    batter_stand:    str,
    pitcher_throws:  str,
    actions:         list,
    n_iter:          int   = 500,
    c:               float = 1.4,
    max_rollout_depth: int = 12,
) -> MCTSNode:
    """Run n_iter UCT iterations and return the root (children hold visit stats)."""
    if not actions:
        return root

    with torch.no_grad():
        for _ in range(n_iter):
            _iterate(
                root, simulator, pitcher_tensors,
                pitcher_id, batter_stand, pitcher_throws,
                actions, c, max_rollout_depth,
            )

    return root


# ------------------------------------------------------------------
# One UCT iteration
# ------------------------------------------------------------------

def _iterate(
    root, simulator, pitcher_tensors,
    pitcher_id, batter_stand, pitcher_throws,
    actions, c, max_rollout_depth,
) -> float:

    # 1. Selection
    node = root
    path = [node]
    while node.is_fully_expanded() and not node.is_terminal and node.children:
        _, node = node.best_child(c)
        path.append(node)

    # 2. Terminal node revisited — replay stored value
    if node.is_terminal:
        run_value = node.terminal_run_value
        for n in path:
            n.update(run_value)
        return run_value

    # 3. Expansion — pop one untried action and simulate it
    if node.untried_actions:
        action = node.untried_actions.pop()
        new_window, new_state, rv, is_terminal = simulator.simulate_pitch(
            node.state, node.batter_window, pitcher_tensors,
            pitcher_id, action, batter_stand, pitcher_throws,
        )
        child = MCTSNode(
            state              = new_state,
            batter_window      = new_window,
            action             = action,
            untried_actions    = [] if is_terminal else list(actions),
            is_terminal        = is_terminal,
            terminal_run_value = rv if is_terminal else 0.0,
        )
        node.children[action] = child
        path.append(child)
        node = child
        run_value = rv
    else:
        run_value = 0.0

    # 4. Rollout — random simulation from leaf to terminal
    if not node.is_terminal:
        run_value = _rollout(
            node.state, node.batter_window, simulator,
            pitcher_tensors, pitcher_id, batter_stand, pitcher_throws,
            actions, max_rollout_depth,
        )

    # 5. Backpropagation
    for n in path:
        n.update(run_value)

    return run_value


def _rollout(
    state, batter_window, simulator, pitcher_tensors,
    pitcher_id, batter_stand, pitcher_throws,
    actions, max_depth,
) -> float:
    """Random rollout from a leaf node to a terminal outcome."""
    for _ in range(max_depth):
        action = random.choice(actions)
        batter_window, state, run_value, is_terminal = simulator.simulate_pitch(
            state, batter_window, pitcher_tensors,
            pitcher_id, action, batter_stand, pitcher_throws,
        )
        if is_terminal:
            return run_value
    return 0.0  # at-bat didn't terminate within depth limit (extremely rare)
