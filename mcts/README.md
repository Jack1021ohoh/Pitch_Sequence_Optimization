# MCTS Pitch Sequencer (Phase 2)

UCT (Upper Confidence Trees) search over pitch sequences, using the Phase 1 transformer as a simulator. No additional training required.

## Algorithm

Each iteration:

1. **Selection** — descend the tree via UCB1 until reaching an unexpanded or terminal node
2. **Expansion** — pop one untried `(pitch_type, zone)` action, simulate it with the model, and create a child node
3. **Rollout** — simulate random actions from the new child until the at-bat ends (or depth limit)
4. **Backpropagation** — update visit counts and value sums along the path

**Value convention:** `value_sum` stores cumulative `(-run_value)`, so higher Q is better for the pitcher. The RE24 run expectancy table is the reward signal — outcomes that reduce run expectancy (strikeouts, weak contact) yield positive Q for the pitcher.

## Usage

```bash
python -m mcts.run_search --checkpoint checkpoints/best.pt \
                           --batter-id 592450 --pitcher-id 605483 \
                           --n-iter 500 --top-k 10
```

Omit `--batter-id` / `--pitcher-id` to sample a random at-bat from `--split`.

**Arguments:**

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | Path to `best.pt` or `latest.pt` |
| `--batter-id` | random | MLB player ID of the batter |
| `--pitcher-id` | random | MLB player ID of the pitcher |
| `--split` | `test` | Which split to sample from (`train`/`val`/`test`) |
| `--n-iter` | 500 | Number of UCT iterations |
| `--c` | 1.4 | UCB1 exploration constant |
| `--max-rollout` | 12 | Maximum rollout depth before declaring no outcome |
| `--top-k` | 10 | Rows to show in the output table |
| `--seed` | 42 | Random seed for sample selection and rollouts |

**Example output:**

```
── MCTS Recommendations (500 simulations) ─────────────────
  #    Pitch        Zone   Visits   Q (↑ pitcher)    Share
  ───────────────────────────────────────────────────────
  1    FF           6      142      0.1823           28.4%
  2    SL           14     98       0.1541           19.6%
  ...
```

## Files

### `state.py` — `AtBatState`

Immutable frozen dataclass representing the at-bat game state: balls, strikes, outs, base occupancy, inning, score differential. `apply(outcome, run_values)` returns a new state, a run value, and a terminal flag.

Base transitions are simplified (no tag-ups, no double plays, no sac flies). Count overflow safeguards handle the rare case where the model predicts `Ball` at a full count instead of `Walk`, or `Strike` at two strikes instead of `Strikeout`.

### `simulator.py` — `PitchSimulator`

Wraps the trained model to simulate one pitch at a time:

1. Slide the 400-pitch batter window (drop oldest, append candidate pitch with mask flags set)
2. Run one forward pass
3. Sample outcome and hit location from predicted distributions
4. Fill sampled values back into the window row (clear mask flags)
5. Apply outcome to game state via `AtBatState.apply()`

`available_actions(pitcher_id)` returns all `(pitch_type, zone)` pairs with sufficient data in the pitch library for that pitcher.

### `node.py` — `MCTSNode`

One node in the search tree. Stores:
- `state` — current `AtBatState`
- `batter_window` — `(400, 30)` pitch history ready for the next `simulate_pitch` call
- `untried_actions` — actions not yet expanded from this node
- `visits`, `value_sum` — for UCB1
- `terminal_run_value` — stored on terminal nodes so revisits replay the correct reward without re-simulating

`best_action()` returns the most-visited child (robust policy selection). `ranked_actions()` returns all children sorted by visit count.

### `search.py` — `mcts_search`

Runs `n_iter` UCT iterations under a single `torch.no_grad()` context. Terminal nodes that are revisited during selection replay their stored `terminal_run_value` rather than re-simulating.

### `run_search.py` — CLI entry point

Loads model + artifacts, selects a batter/pitcher pair, builds the root node from the historical batter window, runs the search, and prints the ranked action table.

The root batter window ends at `end_row - 1` (the 400 pitches leading up to the at-bat being optimized). The initial game state is read from `arr[end_row]` — the actual next pitch row in the batter array.

## Prerequisites

Phase 1 training and the following data artifacts must exist:

```
data/artifacts/pitch_library.pkl    # python data/build_pitch_library.py
data/re24_table.json                # python data/build_re24_table.py
data/artifacts/vocabs.json          # python data/preprocess.py
checkpoints/best.pt                 # python -m training.train
```
