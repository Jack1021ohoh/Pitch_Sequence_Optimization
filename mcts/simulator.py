"""PitchSimulator: wraps the trained Phase 1 model to simulate one pitch at a time.

Each call to simulate_pitch():
  1. Slides the batter window (drop oldest pitch, append candidate pitch with mask=1)
  2. Runs one model forward pass
  3. Samples an outcome from the predicted distribution
  4. Fills the outcome back into the window (clears mask flags)
  5. Applies the outcome to the game state via AtBatState.apply()

The returned window is ready to be passed back in for the next pitch.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from mcts.state import AtBatState

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

    def simulate_pitch(
        self,
        state:           AtBatState,
        batter_window:   np.ndarray,        # (400, 30), unmasked history
        pitcher_tensors: dict,              # pre-batched tensors on device (B=1)
        pitcher_id:      int,
        action:          tuple[str, int],   # (pitch_type, zone)
        batter_stand:    str,
        pitcher_throws:  str,
    ) -> tuple[np.ndarray, AtBatState, float, bool]:
        """Simulate one pitch.

        Returns:
            new_window:   (400, 30) with candidate pitch outcome filled in
            new_state:    updated AtBatState
            run_value:    run value of the outcome (positive = good for offense)
            is_terminal:  True when the at-bat ends
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

        outcome_probs = torch.softmax(preds['pitch_outcome_logits'][0], dim=-1).cpu()
        outcome = int(torch.multinomial(outcome_probs, 1).item())

        loc_probs = torch.softmax(preds['hit_location_logits'][0], dim=-1).cpu()
        location  = int(torch.multinomial(loc_probs, 1).item())

        candidate_window[-1, _CAT_OUTCOME]  = outcome
        candidate_window[-1, _CAT_HIT_LOC]  = location
        candidate_window[-1, _MASK_OUTCOME]  = 0.0
        candidate_window[-1, _MASK_HIT_LOC]  = 0.0

        new_state, run_value, is_terminal = state.apply(outcome, self.run_values)
        return candidate_window, new_state, run_value, is_terminal

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
