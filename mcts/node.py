"""MCTSNode: one node in the UCT search tree.

Each node represents a game state reached after a sequence of pitches.
`batter_window` is the (400, 30) history with all pitches unmasked — ready to
be passed directly to PitchSimulator.simulate_pitch() for the next expansion.

Value convention: value_sum stores cumulative (-run_value), so higher Q is
better for the pitcher (pitcher minimises run expectancy).
"""

import math

import numpy as np

from mcts.state import AtBatState


class MCTSNode:
    __slots__ = (
        'state', 'batter_window', 'action',
        'untried_actions', 'visits', 'value_sum',
        'children', 'is_terminal', 'terminal_run_value',
    )

    def __init__(
        self,
        state:              AtBatState,
        batter_window:      np.ndarray,    # (400, 30), fully unmasked history
        action:             tuple | None,  # (pitch_type, zone) that reached this node
        untried_actions:    list,          # actions not yet expanded from this node
        is_terminal:        bool  = False,
        terminal_run_value: float = 0.0,  # stored so revisits replay the correct reward
    ):
        self.state              = state
        self.batter_window      = batter_window
        self.action             = action
        self.untried_actions    = list(untried_actions)  # copy so caller mutation is safe
        self.visits             = 0
        self.value_sum          = 0.0
        self.children: dict[tuple, MCTSNode] = {}
        self.is_terminal        = is_terminal
        self.terminal_run_value = terminal_run_value

    # ------------------------------------------------------------------
    # UCB1 selection
    # ------------------------------------------------------------------

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visits if self.visits > 0 else 0.0

    def ucb1(self, parent_visits: int, c: float) -> float:
        if self.visits == 0:
            return float('inf')
        return self.q_value + c * math.sqrt(math.log(parent_visits) / self.visits)

    def best_child(self, c: float) -> tuple[tuple, 'MCTSNode']:
        """Select child with highest UCB1 score (used during tree traversal)."""
        return max(self.children.items(), key=lambda kv: kv[1].ucb1(self.visits, c))

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, run_value: float) -> None:
        """Record one simulation result. Negated so higher value_sum = better for pitcher."""
        self.visits    += 1
        self.value_sum += -run_value

    # ------------------------------------------------------------------
    # Result extraction (called on root after search completes)
    # ------------------------------------------------------------------

    def best_action(self) -> tuple:
        """Return the most-visited child action (robust policy selection)."""
        return max(self.children, key=lambda a: self.children[a].visits)

    def ranked_actions(self) -> list[tuple]:
        """Return all child actions sorted by visit count descending."""
        return sorted(self.children, key=lambda a: self.children[a].visits, reverse=True)
