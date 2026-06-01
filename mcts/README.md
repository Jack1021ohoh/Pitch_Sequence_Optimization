# MCTS Pitch Sequencer (Phase 2)

Expectimax MCTS over pitch sequences, using the Phase 1 transformer as a simulator. No additional training required.

## Algorithm

The pitch outcome is a chance event. Rather than sampling a single outcome per
node (which froze one run value per node and made every action look identical),
chance is **averaged analytically** over the model's full outcome distribution.

Each iteration:

1. **Selection** — descend via UCB (for a minimiser) into the best action edge, then into the least-visited non-terminal continuation (ball / strike), until a node with an untried action (or a leaf / depth limit) is reached
2. **Expansion** — pop one untried `(pitch_type, zone)` action and run a single forward pass. The full outcome distribution `P(o)` is folded into a fixed **expected immediate reward** `Σ_o P(o)·rv(o)`; only the non-terminal outcomes (ball, strike) spawn child decision nodes
3. **Backup** — recompute edge and node values bottom-up via **exact Bellman backup** (not Monte-Carlo averaging), and bump visit counts

**Value convention:** every stored value is an **offense run value** (positive = good for the batter). The pitcher MINIMISES it, so a decision node's value is the minimum over its expanded action edges, and an edge's value is `imm_reward + Σ_branch P(branch)·child.value`. The CLI negates for display, so a higher reported Q is better for the pitcher. The **RE288 run expectancy table** (288 states = 12 count states × 24 base-out states) is the reward signal — this means non-terminal ball and strike transitions are also penalised immediately via the count-state RE delta, not just at at-bat termination.

Because outcome chance is averaged analytically, reported Q-values are exact expectations (continuous, action-dependent) and the ranking does not depend on visit counts. The at-bat terminates naturally — ball/strike branches only exist while the count is below 4 balls / 3 strikes — so no random rollout is needed.

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
| `--max-rollout` | 12 | Maximum search depth (continuation pitches); at-bats terminate by ~6 so this is a safeguard |
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

Immutable frozen dataclass representing the at-bat game state: balls, strikes, outs, base occupancy, inning, score differential. `apply(outcome, re288)` returns a new state, a run value, and a terminal flag.

Base transitions are simplified (no tag-ups, no double plays, no sac flies). Count overflow safeguards handle the rare case where the model predicts `Ball` at a full count instead of `Walk`, or `Strike` at two strikes instead of `Strikeout`.

### `simulator.py` — `PitchSimulator`

Wraps the trained model to evaluate one pitch analytically:

1. Slide the 400-pitch batter window (drop oldest, append candidate pitch with mask flags set)
2. Run one forward pass to get the outcome distribution `P(o)`
3. Compute the expected immediate reward `Σ_o P(o)·rv(o)` via `AtBatState.apply()` for each outcome class (no sampling)
4. Return the non-terminal outcomes (ball, strike) as continuation branches, each with its filled-in window and next state

`available_actions(pitcher_id)` returns all `(pitch_type, zone)` pairs with sufficient data in the pitch library for that pitcher.

### `node.py` — `MCTSNode` / `ActionEdge`

`MCTSNode` is a **decision node** (a state the pitcher acts in). It stores `state`, `batter_window`, `untried_actions`, `visits`, `value` (offense run value; the pitcher minimises), `depth`, and `children` (one `ActionEdge` per expanded pitch).

`ActionEdge` is the **chance edge** for one pitch: `imm_reward` (expected reward over all outcomes) plus `branches` (the non-terminal ball/strike continuations as `(prob, child_node)`). Its `value = imm_reward + Σ P(branch)·child.value`.

`best_action()` / `ranked_actions()` order pitches by exact value (lowest offense run value = best for the pitcher), not by visit count.

### `search.py` — `mcts_search`

Runs `n_iter` expectimax-MCTS iterations under a single `torch.no_grad()` context. Each iteration expands at most one action edge; values are refreshed along the touched path via exact Bellman backup, so reported Q-values converge to true expectations rather than Monte-Carlo estimates.

### `run_search.py` — CLI entry point

Loads model + artifacts, selects a batter/pitcher pair, builds the root node from the historical batter window, runs the search, and prints the ranked action table.

The root batter window ends at `end_row - 1` (the 400 pitches leading up to the at-bat being optimized). The initial game state is read from `arr[end_row]` — the actual next pitch row in the batter array.

## Prerequisites

Phase 1 training and the following data artifacts must exist:

```
data/artifacts/pitch_library.pkl      # python data/build_pitch_library.py
data/artifacts/re288_table.json       # python data/build_re288_table.py
data/artifacts/vocabs.json            # python data/preprocess.py
checkpoints/best.pt                   # python -m training.train
```
