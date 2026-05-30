"""At-bat game state and outcome transitions for MCTS simulation.

Outcome indices match OUTCOME_CLASSES in data/preprocess.py:
  0=Ball, 1=Strike, 2=Single, 3=Double, 4=Triple, 5=Home Run,
  6=Strikeout, 7=Walk, 8=Hit by Pitch, 9=Field Out
"""

from __future__ import annotations
from dataclasses import dataclass, replace

BALL         = 0
STRIKE       = 1
SINGLE       = 2
DOUBLE       = 3
TRIPLE       = 4
HOME_RUN     = 5
STRIKEOUT    = 6
WALK         = 7
HIT_BY_PITCH = 8
FIELD_OUT    = 9


@dataclass(frozen=True)
class AtBatState:
    balls:      int  = 0
    strikes:    int  = 0
    outs:       int  = 0
    on_1b:      bool = False
    on_2b:      bool = False
    on_3b:      bool = False
    inning:     int  = 1
    score_diff: int  = 0  # fielding_score - batting_score

    def re24_index(self) -> int:
        """Encode base-out state as 0–23 for RE24 table lookup."""
        return int(self.on_1b) + int(self.on_2b) * 2 + int(self.on_3b) * 4 + self.outs * 8

    def apply(self, outcome: int, re24: dict) -> tuple[AtBatState, float, bool]:
        """Apply a pitch outcome to this state.

        Args:
            outcome: pitch outcome index (0–9)
            re24:    dict mapping str(base-out index 0–23) → run expectancy

        Returns:
            new_state:   updated AtBatState
            run_value:   RE24 delta (positive = good for offense)
            is_terminal: True when the at-bat has ended
        """
        if outcome == BALL:
            new_balls = self.balls + 1
            if new_balls >= 4:
                # Safeguard: model should predict Walk, but force it if count overflows
                new_state = self._advance_walk()
                return new_state, self._re24_delta(new_state, 0, re24), True
            return replace(self, balls=new_balls), 0.0, False

        if outcome == STRIKE:
            new_strikes = self.strikes + 1
            if new_strikes >= 3:
                # Safeguard: model should predict Strikeout, but force it if count overflows
                new_state = replace(self, balls=0, strikes=0, outs=self.outs + 1)
                return new_state, self._re24_delta(new_state, 0, re24), True
            return replace(self, strikes=new_strikes), 0.0, False

        new_state   = self._advance_bases(outcome)
        runs_scored = self._runs_scored(outcome)
        rv          = self._re24_delta(new_state, runs_scored, re24)
        return new_state, rv, True

    def _re24_delta(self, new_state: 'AtBatState', runs_scored: int, re24: dict) -> float:
        """RE24 run value: (E[runs | new state] + runs scored) - E[runs | old state]."""
        old_exp = re24.get(str(self.re24_index()), 0.0)
        # outs=3 means end of inning → 0 run expectancy
        new_exp = re24.get(str(new_state.re24_index()), 0.0)
        return (new_exp + runs_scored) - old_exp

    def _runs_scored(self, outcome: int) -> int:
        """Runs that score during this terminal outcome (excluding those still on base)."""
        b1, b2, b3 = int(self.on_1b), int(self.on_2b), int(self.on_3b)
        if outcome == HOME_RUN:
            return 1 + b1 + b2 + b3
        if outcome == TRIPLE:
            return b1 + b2 + b3        # batter ends on 3rd, all runners score
        if outcome == DOUBLE:
            return b2 + b3             # b1 → 3rd, batter → 2nd
        if outcome == SINGLE:
            return b3                  # b2 → 3rd, b1 → 2nd, batter → 1st
        if outcome in (WALK, HIT_BY_PITCH):
            return b1 & b2 & b3        # force-scores 3rd only when bases loaded
        return 0                       # STRIKEOUT, FIELD_OUT

    # ------------------------------------------------------------------
    # Base transition helpers (simplified — no tag-up, no DP, no sac fly)
    # ------------------------------------------------------------------

    def _advance_walk(self) -> AtBatState:
        """Force-advance runners on a walk or HBP."""
        b1, b2, b3 = self.on_1b, self.on_2b, self.on_3b
        # Runner on 3rd forced home only when bases are loaded
        # Runner on 2nd forced to 3rd when 1st and 2nd occupied
        # Runner on 1st always forced to 2nd
        new_3b = b3 or (b1 and b2)
        new_2b = b2 or b1
        return replace(self, on_1b=True, on_2b=new_2b, on_3b=new_3b,
                       balls=0, strikes=0)

    def _advance_bases(self, outcome: int) -> AtBatState:
        b1, b2, b3 = self.on_1b, self.on_2b, self.on_3b
        if outcome == HOME_RUN:
            return replace(self, on_1b=False, on_2b=False, on_3b=False,
                           balls=0, strikes=0)
        if outcome == TRIPLE:
            return replace(self, on_1b=False, on_2b=False, on_3b=True,
                           balls=0, strikes=0)
        if outcome == DOUBLE:
            # b1 → 3rd; b2, b3 score
            return replace(self, on_1b=False, on_2b=True, on_3b=bool(b1),
                           balls=0, strikes=0)
        if outcome == SINGLE:
            # b3 scores; b2 → 3rd; b1 → 2nd; batter → 1st
            return replace(self, on_1b=True, on_2b=bool(b1), on_3b=bool(b2),
                           balls=0, strikes=0)
        if outcome in (WALK, HIT_BY_PITCH):
            return self._advance_walk()
        if outcome in (STRIKEOUT, FIELD_OUT):
            return replace(self, outs=self.outs + 1, balls=0, strikes=0)
        return self
