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

# Maps outcome index → run_values key in re24_table.json
_RUN_VALUE_KEY = {
    SINGLE:       'single',
    DOUBLE:       'double',
    TRIPLE:       'triple',
    HOME_RUN:     'home_run',
    STRIKEOUT:    'strikeout',
    WALK:         'walk',
    HIT_BY_PITCH: 'hit_by_pitch',
    FIELD_OUT:    'field_out',
}


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

    def apply(self, outcome: int, run_values: dict) -> tuple[AtBatState, float, bool]:
        """Apply a pitch outcome to this state.

        Returns:
            new_state:   updated AtBatState
            run_value:   run value of this outcome (positive = good for offense)
            is_terminal: True when the at-bat has ended
        """
        if outcome == BALL:
            new_balls = self.balls + 1
            if new_balls >= 4:
                # Safeguard: model should predict Walk, but force it if count overflows
                return self._advance_walk(), run_values.get('walk', 0.33), True
            return replace(self, balls=new_balls), 0.0, False

        if outcome == STRIKE:
            new_strikes = self.strikes + 1
            if new_strikes >= 3:
                # Safeguard: model should predict Strikeout, but force it if count overflows
                return replace(self, strikes=new_strikes, outs=self.outs + 1), \
                       run_values.get('strikeout', -0.28), True
            return replace(self, strikes=new_strikes), 0.0, False

        rv        = run_values.get(_RUN_VALUE_KEY.get(outcome, ''), 0.0)
        new_state = self._advance_bases(outcome)
        return new_state, rv, True

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
