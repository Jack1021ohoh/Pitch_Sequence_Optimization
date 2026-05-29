"""Expectimax MCTS for optimal pitch sequencing.

Chance (the pitch outcome) is averaged analytically rather than sampled: each
expansion does one forward pass and folds the full outcome distribution into a
fixed expected immediate reward plus continuation branches for the two
non-terminal outcomes (ball, strike). The pitcher MINIMISES expected run value,
so a decision node's value is the minimum over its expanded action edges, and
each edge's value is its immediate reward plus the probability-weighted values
of its continuation children.

Per iteration:
  1. Selection  — descend via UCB (minimiser) into actions, then into the
                  least-visited non-terminal branch, until a node with an
                  untried action (or a leaf / depth limit) is reached.
  2. Expansion  — evaluate one untried action (one forward pass), creating its
                  edge and the ball/strike continuation nodes.
  3. Backup     — recompute edge and node values bottom-up via exact Bellman
                  backup (no Monte-Carlo averaging) and bump visit counts.

All values are offense run values (positive = good for the batter); the CLI
negates for display so higher reported Q is better for the pitcher.
"""

import torch

from mcts.node      import MCTSNode, ActionEdge
from mcts.simulator import PitchSimulator


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
    max_depth:       int   = 8,
) -> MCTSNode:
    """Run n_iter expectimax-MCTS iterations and return the root."""
    if not actions:
        return root

    with torch.no_grad():
        for _ in range(n_iter):
            _iterate(
                root, simulator, pitcher_tensors,
                pitcher_id, batter_stand, pitcher_throws,
                actions, c, max_depth,
            )

    return root


# ------------------------------------------------------------------
# One iteration
# ------------------------------------------------------------------

def _iterate(
    root, simulator, pitcher_tensors,
    pitcher_id, batter_stand, pitcher_throws,
    actions, c, max_depth,
) -> None:

    node       = root
    path_nodes = [node]            # decision nodes touched this iteration
    path_edges = []                # action edges touched this iteration

    # 1–2. Descend, expanding the first untried action encountered.
    while True:
        if node.untried_actions and node.depth < max_depth:
            action = node.untried_actions.pop()
            imm_reward, branch_data = simulator.evaluate_pitch(
                node.state, node.batter_window, pitcher_tensors,
                pitcher_id, action, batter_stand, pitcher_throws,
            )
            branches = []
            for prob, branch_window, branch_state in branch_data:
                child = MCTSNode(
                    state           = branch_state,
                    batter_window   = branch_window,
                    untried_actions = actions,
                    depth           = node.depth + 1,
                )
                branches.append((prob, child))
            edge = ActionEdge(imm_reward, branches)
            node.children[action] = edge
            path_edges.append(edge)
            break

        if not node.children:        # leaf: no actions to expand, no chance branches
            break

        # Selection: best action edge, then descend a non-terminal continuation.
        _, edge = node.select_edge(c)
        path_edges.append(edge)
        if not edge.branches:        # action is fully terminal — value already exact
            break
        node = edge.pick_branch_child()
        path_nodes.append(node)

    # 3. Backup — exact Bellman, bottom-up.
    for n in path_nodes:
        n.visits += 1
    for e in path_edges:
        e.visits += 1
    for n in reversed(path_nodes):
        for e in n.children.values():
            e.recompute()
        n.backup_value()
