"""Search-tree structures for expectimax MCTS over pitch sequences.

The tree alternates between two kinds of vertices:

* **Decision node** (`MCTSNode`) — a game state the pitcher must act in. The
  pitcher MINIMISES expected run value, so a node's value is the minimum over
  its expanded action edges.
* **Action edge** (`ActionEdge`) — the pitcher throws one ``(pitch_type, zone)``.
  The outcome is a chance event: rather than sampling a single outcome (which
  froze one run value per node in the old design), we average analytically over
  the full predicted outcome distribution. Terminal outcomes (strikeout, in
  play, walk, …) collapse into a fixed expected immediate reward; only the two
  non-terminal outcomes (ball, strike) spawn child decision nodes to recurse on.

Value convention: every stored value is an **offense run value** (positive =
good for the batter). The pitcher minimises it. The CLI negates for display so
that a higher reported Q is better for the pitcher.
"""

import math

import numpy as np

from mcts.state import AtBatState


class ActionEdge:
    """One pitch choice from a decision node, with chance averaged analytically.

    value = imm_reward + Σ_branch  P(branch) · child_node.value

    where ``imm_reward`` already includes the probability-weighted run value of
    every terminal outcome (and of ball/strike, whose immediate run value is 0),
    and ``branches`` holds only the non-terminal continuations.
    """

    __slots__ = ('imm_reward', 'branches', 'visits', 'value')

    def __init__(self, imm_reward: float, branches: list):
        self.imm_reward = imm_reward
        self.branches   = branches          # list[(prob, MCTSNode)]
        self.visits     = 0
        self.value      = self._compute()

    def _compute(self) -> float:
        return self.imm_reward + sum(p * child.value for p, child in self.branches)

    def recompute(self) -> None:
        """Refresh value from current child-node values (exact Bellman backup)."""
        self.value = self._compute()

    def pick_branch_child(self) -> 'MCTSNode':
        """Least-visited non-terminal continuation (round-robins ball vs strike)."""
        return min(self.branches, key=lambda pc: pc[1].visits)[1]


class MCTSNode:
    """A decision node: a state the pitcher acts in, minimising run value."""

    __slots__ = (
        'state', 'batter_window', 'untried_actions',
        'visits', 'value', 'children', 'depth',
    )

    def __init__(
        self,
        state:           AtBatState,
        batter_window:   np.ndarray,   # (400, 30), fully unmasked history
        untried_actions: list,         # actions not yet expanded from this node
        depth:           int = 0,
    ):
        self.state           = state
        self.batter_window   = batter_window
        self.untried_actions = list(untried_actions)
        self.visits          = 0
        # Offense-run-value estimate. 0.0 is the neutral leaf prior used until an
        # action is expanded (and for continuations beyond the depth limit).
        self.value           = 0.0
        self.children: dict[tuple, ActionEdge] = {}
        self.depth           = depth

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def is_leaf(self) -> bool:
        """No actions left to expand and no chance branches to recurse into."""
        return not self.untried_actions and not self.children

    def select_edge(self, c: float) -> tuple[tuple, ActionEdge]:
        """UCB selection for a MINIMISER: pick the lowest-value edge, with an
        exploration bonus that lowers the score of rarely-visited edges."""
        best_action, best_edge, best_score = None, None, math.inf
        for action, edge in self.children.items():
            if edge.visits == 0:
                return action, edge
            score = edge.value - c * math.sqrt(math.log(self.visits) / edge.visits)
            if score < best_score:
                best_action, best_edge, best_score = action, edge, score
        return best_action, best_edge

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def backup_value(self) -> None:
        """Pitcher minimises: node value = min over expanded action edges."""
        if self.children:
            self.value = min(edge.value for edge in self.children.values())

    # ------------------------------------------------------------------
    # Result extraction (called on the root after search completes)
    # ------------------------------------------------------------------

    def q_value(self, action: tuple) -> float:
        """Pitcher-perspective value of an action (higher = better for pitcher)."""
        return -self.children[action].value

    def ranked_actions(self) -> list[tuple]:
        """Actions sorted by pitcher value descending (best pitch first).

        Values are exact analytic expectations, so this ranking is meaningful
        regardless of visit counts (unlike Monte-Carlo visit-count ranking)."""
        return sorted(self.children, key=lambda a: self.children[a].value)

    def best_action(self) -> tuple:
        return min(self.children, key=lambda a: self.children[a].value)
