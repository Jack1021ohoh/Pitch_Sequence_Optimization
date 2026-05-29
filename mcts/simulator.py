"""PitchSimulator: wraps the trained Phase 1 model to evaluate one pitch by
averaging analytically over the predicted outcome distribution.

Each call to evaluate_pitch():
  1. Slides the batter window (drop oldest pitch, append candidate pitch with mask=1)
  2. Runs one model forward pass to get the outcome distribution P(o)
  3. Sums the expected immediate run value Σ_o P(o)·rv(o) (no sampling)
  4. Returns the non-terminal outcomes (ball, strike) as continuation branches,
     each with its filled-in window and next AtBatState

The returned branch windows are ready to be passed back in for the next pitch.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcts.state import AtBatState, BALL, STRIKE

N_PITCH_OUTCOME = 10   # must match models/full_model.py

# Column indices in the 30-col model input — must match training/dataset.py
_CAT_START       = 15   # first categorical column
_CAT_PITCH_TYPE  = 15
_CAT_OUTCOME     = 16   # zeroed on candidate pitch, filled after sampling
_CAT_HIT_LOC     = 17   # zeroed on candidate pitch, filled after sampling (contact only)
_CAT_BALLS       = 18
_CAT_STRIKES     = 19
_CAT_OUTS        = 20
_CAT_ON_1B       = 21
_CAT_ON_2B       = 22
_CAT_ON_3B       = 23
_CAT_INNING      = 24
_CAT_STAND       = 25
_CAT_P_THROWS    = 26
_CAT_ZONE        = 27
_MASK_OUTCOME    = 28
_MASK_HIT_LOC    = 29


class PitchSimulator:
    def __init__(
        self,
        model,
        pitch_library: dict,
        run_values:    dict,
        vocabs:        dict,
        device:        torch.device,
    ):
        self.model         = model
        self.pitch_library = pitch_library
        self.run_values    = run_values
        self.vocabs        = vocabs
        self.device        = device
        self.model.eval()

    # ------------------------------------------------------------------
    # Action space
    # ------------------------------------------------------------------

    def available_actions(self, pitcher_id: int) -> list[tuple[str, int]]:
        """Return (pitch_type, zone) pairs with sufficient training data for this pitcher."""
        actions = []
        for pitch_type, zones in self.pitch_library.get(pitcher_id, {}).items():
            for zone in zones:
                if zone != '_mean':
                    actions.append((pitch_type, int(zone)))
        return actions

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def evaluate_pitch(
        self,
        state:           AtBatState,
        batter_window:   np.ndarray,        # (400, 30), unmasked history
        pitcher_tensors: dict,              # pre-batched tensors on device (B=1)
        pitcher_id:      int,
        action:          tuple[str, int],   # (pitch_type, zone)
        batter_stand:    str,
        pitcher_throws:  str,
    ) -> tuple[float, list]:
        """Evaluate one pitch by averaging analytically over the outcome
        distribution — no sampling.

        A single forward pass gives P(outcome). The expected immediate run value
        sums P(o)·rv(o) over all 10 outcome classes (ball/strike contribute 0,
        in-play and strikeout/walk contribute their RE24 run values). Only the
        non-terminal outcomes (ball, strike) continue the at-bat, so they are
        returned as branches for the search to recurse on.

        Returns:
            imm_reward:  Σ_o P(o)·rv(o)  (positive = good for offense)
            branches:    list of (prob, new_window, new_state) for the
                         non-terminal continuations (ball / strike). Empty when
                         every outcome ends the at-bat (e.g. a full count).
        """
        pitch_type, zone = action

        candidate_window = np.empty_like(batter_window)
        candidate_window[:-1] = batter_window[1:]
        candidate_window[-1]  = self._build_pitch_row(
            state, pitcher_id, pitch_type, zone, batter_stand, pitcher_throws,
        )

        batter_seq = torch.from_numpy(candidate_window).float().unsqueeze(0).to(self.device)
        preds = self.model(
            batter_seq,
            pitcher_tensors['pitcher_pitches'],
            pitcher_tensors['pitcher_pitch_mask'],
            pitcher_tensors['pitcher_context'],
            pitcher_tensors['pitcher_app_mask'],
        )
        probs = torch.softmax(preds['pitch_outcome_logits'][0], dim=-1).cpu().numpy()

        imm_reward = 0.0
        branches   = []
        for outcome in range(N_PITCH_OUTCOME):
            p = float(probs[outcome])
            new_state, run_value, is_terminal = state.apply(outcome, self.run_values)
            imm_reward += p * run_value
            # Only ball/strike that do NOT end the at-bat spawn a continuation.
            if not is_terminal and outcome in (BALL, STRIKE):
                branch_window = candidate_window.copy()
                branch_window[-1, _CAT_OUTCOME] = outcome
                branch_window[-1, _CAT_HIT_LOC] = 0          # no contact on ball/strike
                branch_window[-1, _MASK_OUTCOME] = 0.0
                branch_window[-1, _MASK_HIT_LOC] = 0.0
                branches.append((p, branch_window, new_state))

        return imm_reward, branches

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pitch_row(
        self,
        state:          AtBatState,
        pitcher_id:     int,
        pitch_type:     str,
        zone:           int,
        batter_stand:   str,
        pitcher_throws: str,
    ) -> np.ndarray:
        row = np.zeros(30, dtype=np.float32)

        pt_lib = self.pitch_library.get(pitcher_id, {}).get(pitch_type, {})
        row[0:11] = pt_lib.get(zone, pt_lib.get('_mean', row[0:11]))
        # cols [11:15] outcome-continuous stay zeroed (masked)

        row[_CAT_PITCH_TYPE] = self.vocabs['pitch_type'].get(pitch_type, 0)
        row[_CAT_OUTCOME]    = 0   # masked
        row[_CAT_HIT_LOC]    = 0   # masked
        row[_CAT_BALLS]      = state.balls
        row[_CAT_STRIKES]    = state.strikes
        row[_CAT_OUTS]       = state.outs
        row[_CAT_ON_1B]      = int(state.on_1b)
        row[_CAT_ON_2B]      = int(state.on_2b)
        row[_CAT_ON_3B]      = int(state.on_3b)
        row[_CAT_INNING]     = min(state.inning, 10)
        row[_CAT_STAND]      = self.vocabs['stand'].get(batter_stand, 1)
        row[_CAT_P_THROWS]   = self.vocabs['p_throws'].get(pitcher_throws, 1)
        row[_CAT_ZONE]       = zone

        row[_MASK_OUTCOME]  = 1.0
        row[_MASK_HIT_LOC]  = 1.0

        return row
