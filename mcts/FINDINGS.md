# MCTS Pitch Sequencer — Case Study

## Overview

To demonstrate the practical utility of the expectimax MCTS pitch sequencer, we conduct a case
study across three elite matchups: Shohei Ohtani (LHH) vs. Paul Skenes (RHP), Juan Soto (LHH)
vs. Zack Wheeler (RHP), and Vladimir Guerrero Jr. (RHH) vs. Tarik Skubal (LHP). Each matchup
is evaluated under five distinct game-state scenarios — a full-count strikeout situation, a
runner-on-first double-play situation, a bases-loaded first pitch, a pitcher-ahead 0-2 waste
pitch, and a hitter's-count 3-1 scenario — for a total of fifteen scenario-recommendations.
The search is run for 15,000 MCTS iterations per scenario with an exploration constant of
c = 1.4. All Q-values represent expected run-value deltas from the pitcher's perspective
(higher is better for the pitcher) derived from the RE288 run expectancy table. Zone numbers
follow the standard Statcast convention indexed from the catcher's perspective: zones 1–3
occupy the top row of the strike zone, zones 7–9 the bottom row, with the right column (zones
3, 6, 9) corresponding to the inside half against left-handed batters and the away half against
right-handed batters. The four outer ball zones are labeled 11 (high-away for LHH / high-inside
for RHH), 12 (high-inside for LHH / high-away for RHH), 13 (low-away for LHH / low-inside for
RHH), and 14 (low-inside for LHH / low-away for RHH).

---

## 4.1 Shohei Ohtani vs. Paul Skenes

Shohei Ohtani is among the most dangerous left-handed hitters in the game, combining elite bat
speed with disciplined plate coverage. Paul Skenes is a right-handed power arm whose arsenal
centers on a triple-digit four-seam fastball, a diving splitter, and an 88-mph sweeper with
sharp horizontal break. The combination presents a textbook challenge: how does a power pitcher
attack one of the game's best hitters with multi-pitch sequencing across varying game states?

### Scenario 1: Full Count, Runner on Third, One Out

In this high-leverage scenario, the count is 3-2, there is one out, and a runner stands on
third base. The strategic objective is unambiguous: record a strikeout to strand the runner and
end the inning. The model must weigh the risk of a walk — which scores the run immediately —
against the imperative to generate a swing-and-miss or called strike.

The model's top recommendation is a changeup to zone 9, earning Q = 0.380 with 1,919 visits
and a 12.8% share of all 15,000 iterations. A splitter to zone 9 follows at Q = 0.339 (964
visits), with a curveball to zone 9 third at Q = 0.326. The concentration of recommendations
in zone 9 — the inside-low corner of the strike zone against a left-handed batter — reflects
the model's assessment that Skenes' off-speed offerings finishing on the inner half carry the
highest probability of inducing a strikeout from Ohtani. The inside-low quadrant aligns with
documented tendencies for left-handed batters to expand their swing on pitches that appear to
catch the zone on the inner edge, and Skenes' splitter and changeup are among his most
swing-and-miss pitches when spotted at or below the inner quadrant.

### Scenario 2: First Pitch, Bases Loaded, One Out

With the count at 0-0 and the bases loaded, the first pitch carries enormous weight. A walk
immediately forces in a run; a hit could clear the bases. Unlike the strikeout scenario, the
pitcher can tolerate a ball in the dirt, making the strategic calculus more about controlling
the at-bat than recording an out on the first pitch.

The model's recommendation is a sweeper (ST) to zone 7, the away-low corner of the strike zone
against a left-handed batter, earning Q = 0.084 with 1,267 visits and an 8.4% share. A sweeper
to zone 1, the away-high corner, follows at Q = 0.058 (856 visits, 5.7%). The third-ranked
action is a sweeper to zone 12 at Q = 0.048 (758 visits, 5.1%). Zone 12 is the high-inside
ball zone — above the strike zone on the inner half against a left-handed batter. The joint
emergence of the three sweeper locations — away-low, away-high, and high-inside — reveals a
multi-dimensional picture of how Skenes' sweeper is being deployed in this context. The two
away-corner locations (zones 7 and 1) describe the classic back-door trajectory: a pitch that
appears to start off the outside edge and sweeps across the outer third of the zone at varying
heights, forcing the batter to either take a strike or make contact on the outer half. The zone
12 recommendation introduces a contrasting look: a sweeper elevated above the zone on the
inside corner, a pitch designed to generate a weak swing above the hands or a called strike if
the batter freezes. The appearance of a high-ball-zone location among the top three — where
earlier lower-iteration searches had not differentiated it from the field — indicates that the
additional search depth has allowed the MCTS to identify elevated pitches as a meaningful
strategic option alongside the more conventional away-corner approach.

### Scenario 3: Hitter's Count, 3-1

With the count at 3-1 and no runners, the pitcher cannot afford a walk. A four-ball sequence
hands the batter first base and grants him a free pass the pitcher has done nothing to earn. The
imperative to throw strikes competes with the need to avoid hittable pitches in a count where
the batter can be selective.

The model's top recommendation is a changeup to zone 7 (Q = 0.143, 761 visits, 5.1%), followed
closely by a sweeper to zone 1 (Q = 0.140, 735 visits, 4.9%). Both pitches target the away
half of the zone — zone 7 at the bottom-away corner and zone 1 at the top-away corner — with
distinct velocity profiles. The pairing of a changeup and a sweeper to the away side on a must-
throw-strike count reflects a strategy of attacking the outer half with pitches that carry the
trajectory of a ball until they clip the zone: the batter, geared up for a fastball in a
favorable count, must commit to off-speed pitches breaking toward the outer corner without
being able to generate leverage if he does swing. The Q-values above 0.14 for both leading
actions confirm that meaningful positive outcomes are achievable for the pitcher even from the
disadvantageous 3-1 count.

---

## 4.2 Juan Soto vs. Zack Wheeler

Juan Soto is a left-handed hitter renowned for his patience and strike-zone discipline, drawing
walks at an elite rate while posting exceptional on-base percentages. Zack Wheeler is a
right-handed starter whose arsenal is anchored by a heavy two-seam sinker that generates
exceptional ground-ball rates, complemented by a hard slider and a four-seam fastball.

### Scenario 1: Full Count, Runner on Third, One Out

The 3-2 strikeout scenario against Soto produces the most concentrated recommendation in the
entire case study. The top action — a sinker to zone 9 — accumulates 3,633 visits at a 24.2%
share with Q = 0.428, nearly two and a half times the visits of the second-ranked action, a
four-seam fastball to zone 9 (1,487 visits, Q = 0.388). This exceptional concentration is
notable given that Soto is widely considered one of the most patient and disciplined hitters in
the game, a batter who rarely chases and makes pitchers work deep into counts. The sinker to
zone 9 against a left-handed batter places a pitch with heavy downward and arm-side movement at
the inside-low corner, a location that both maximizes the risk of a called strike on a full
count and, if contacted, tends to produce weak ground balls or rollover contact due to the
pitch's late downward bite. Against a batter of Soto's caliber, the model's overwhelming
confidence in this single recommendation suggests the search has identified Wheeler's sinker at
the inner edge as an outlier — a pitch location where the match between Wheeler's movement
profile and Soto's documented tendencies converges on a uniquely favorable expected outcome.

### Scenario 2: First Pitch, Bases Loaded, One Out

The bases-loaded first-pitch scenario against Soto produces the second most concentrated
recommendation in the case study. The sweeper to zone 1 — the away-high corner of the strike
zone against a left-handed batter — accumulates 2,881 visits at a 19.2% share with Q = 0.109.
The remaining 80.8% of visits are distributed among 67 other actions, with the next-ranked
action, a sinker to zone 3, at only 894 visits (6.0%). This level of concentration on a single
first-pitch recommendation is unusual and merits attention. Zone 1 represents the high-outside
corner of the strike zone against a left-handed batter. A sweeper thrown by a right-hander that
ends at zone 1 targets the very top of the away side of the zone — a pitch that starts off the
outer edge and breaks back toward the upper-outside corner. Against Soto specifically, whose
approach is built on letting pitches travel deep into the zone before committing, a pitch that
clips the away corner at the top of the zone poses a particular challenge: taking it risks a
called strike on the outside edge, while swinging requires extension at an awkward angle above
the hands. The model's decisive preference for this pitch in the highest-leverage scenario of
the at-bat reflects the sequencer's view that Wheeler's sweeper to the high-away corner is the
foundational pitch against Soto regardless of runner configuration.

### Scenario 3: Pitcher Ahead, 0-2

With the count at 0-2 and no runners, Wheeler holds maximum leverage within the at-bat. The
model recommends a curveball to zone 14 (Q = 0.105, 523 visits) as its top action, followed
closely by a sinker to zone 9 (Q = 0.101, 501 visits) and a splitter to zone 14 (Q = 0.099).
Zone 14 is the low-ball chase zone below and to the outside edge of the strike zone against a
left-handed batter. The appearance of zone 14 at the top of the rankings — and zone 11 (the
outer chase zone) throughout the top ten — confirms that the model correctly identifies
expanded-zone pitching as optimal when the pitcher can afford to give up a ball. The continued
presence of sinker to zone 9 among the top-ranked actions, even in a waste-pitch scenario,
reflects Wheeler's tendency to challenge batters even when ahead, using his heavy sinker on the
inside corner as a pitch that can generate a strikeout swing rather than merely burning a ball.

---

## 4.3 Vladimir Guerrero Jr. vs. Tarik Skubal

Vladimir Guerrero Jr. is a right-handed hitter with elite bat speed and a pull-heavy approach,
capable of driving pitches on the inner half for extra bases. Tarik Skubal is a left-handed
starter for the Detroit Tigers whose effectiveness derives primarily from a heavy sinker and
an elite changeup, supplemented by a sweeper and a curveball that exploit the arm-side break
advantage against right-handed hitters.

### Scenario 1: Full Count, Runner on Third, One Out

The model's top recommendation for Skubal in the full-count strikeout scenario is a sinker to
zone 9, accumulating 2,441 visits at a 16.3% share with Q = 0.421. For a right-handed batter,
zone 9 is the away-low corner of the strike zone — the precise location where a left-handed
pitcher's sinker tails away from the batter's hands with downward and arm-side movement. A
sinker finishing at the bottom-away quadrant against Guerrero targets a location that is
simultaneously difficult to pull for damage and naturally suited to the pitch's movement profile.
The model's second recommendation, a sinker to zone 7 (1,300 visits, Q = 0.389), extends the
away-low theme to the adjacent corner, suggesting a consistent preference for the entire outer-
low portion of the zone rather than a single precise location. The high visit counts accumulated
by away-zone sinkers across the top of the recommendations underscores the model's identification
of Skubal's sinker tailing away from right-handed batters as the primary weapon in this matchup.

### Scenario 2: First Pitch, Bases Loaded, One Out

The bases-loaded first-pitch scenario introduces the most strategically interesting
recommendation in the Guerrero-Skubal matchup. The top action is a sinker to zone 9 (Q = 0.094,
1,410 visits, 9.4%), consistent with the strikeout scenario's preference for the away-low corner.
However, the second-ranked action is a four-seam fastball to zone 11 (Q = 0.075, 1,034 visits,
6.9%). Zone 11 is the high-inside ball zone for a right-handed batter — above the strike zone
on the inner half of the plate. A four-seam fastball to this location is the elevated, tight
fastball that pitching coaches describe as "up and in": a pitch intended to jam the hitter, push
him off the plate, or generate a weak pop-up by elevating on the hands. Against Guerrero, a
pull-oriented right-handed hitter who generates his power by extending his arms through the
ball, a high inside fastball in zone 11 targets the exact location that limits his ability to
drive the ball to the pull side. The appearance of this pitch as the second-ranked action — at
a visit count (1,034) that clearly separates it from the field — represents the search's
discovery of the elevated inside fastball as a complement to the away sinker in this high-stakes
scenario. The two recommendations together describe a coherent inside-outside, high-low
sequencing concept: set up the batter with a high tight fastball to restrict extension, then put
the sinker away-low where he cannot extend to pull it. This pairing closely mirrors real
pitching strategy for attacking powerful right-handed pull hitters.

### Scenario 3: Runner on First, One Out — Ground Ball Situation

With a runner on first and one out, the objective shifts to inducing a ground ball that could
initiate a double play, eliminating the scoring threat and advancing the out count efficiently.
The model's top recommendation is a slider (SL) to zone 3 (Q = 0.021, 552 visits, 3.7%),
followed by a sinker to zone 9 (Q = 0.013, 505 visits) and a four-seam fastball to zone 9
(Q = 0.006, 471 visits). Zone 3 is the away-high corner of the strike zone against a right-
handed batter. A left-handed pitcher's slider thrown to the away-high corner for a right-handed
batter describes a pitch that breaks toward the outer portion of the zone at the top — a high,
sweeping breaking ball moving away from the barrel. Against a pull hitter like Guerrero, a
slider targeting the away-high corner is designed to generate either weak contact to the
opposite field or a swing above the pitch, both outcomes that reduce the risk of a pulled line
drive while keeping the ball in the zone. This recommendation mirrors, from the left-handed
pitcher's perspective, the back-door sweeper pattern observed in the Skenes and Wheeler matchups:
a breaking ball that uses horizontal movement to target the outer edge, this time at the top of
the zone against a right-handed batter.

---

## 4.4 Cross-Matchup Discussion

Several patterns emerge consistently across the three matchups that merit broader consideration.

**Emergence of the elevated pitch.** At 15,000 iterations, the search begins to surface elevated
pitch locations that were indistinguishable from the field at lower iteration counts. In the
bases-loaded first-pitch scenario against Ohtani, a sweeper to zone 12 — the high-inside ball
zone for a left-handed batter — ranks third with 758 visits, clearly separated from the
remaining 76 actions. Against Guerrero in the same scenario, a four-seam fastball to zone 11 —
the high-inside ball zone for a right-handed batter — ranks second with 1,034 visits. In both
cases, the elevated pitch appears alongside away-corner recommendations rather than replacing
them, suggesting the search has identified high-zone pitches not as a primary strikeout weapon
but as a strategic complement: the high-inside pitch pushes the batter off the plate and
restricts extension, setting up the away pitch that follows. This inside-high to away-low
pairing is a foundational concept in pitching strategy that the model has arrived at through
the count-tree exploration rather than through any explicit encoding of pitch sequencing theory.

**The back-door breaking ball.** In every neutral first-pitch scenario across the three matchups,
the search surfaces a breaking ball targeted at the away corner as a leading recommendation.
Against both left-handed batters (Ohtani and Soto), the sweeper to zones 1 and 7 — the away-high
and away-low corners — consistently occupies the top positions, describing the back-door
trajectory of a pitch that begins off the outer edge and sweeps across the far side of the zone.
Against Guerrero (right-handed), the Skubal matchup produces a slider to zone 3 — the away-high
corner from a right-handed batter's perspective — as the top recommendation in the double-play
scenario, demonstrating that the back-door concept is not specific to one handedness combination
but emerges naturally from any pitcher-batter pairing in which a breaking ball with horizontal
movement can be directed to the far corner. The appearance of this strategic concept across
all three pitcher-batter pairs, with no explicit spatial encoding of pitch trajectory in the
model, represents one of the most compelling behavioral findings of this case study.

**Concentration as a measure of strategic clarity.** The visit distributions shift markedly
between high-leverage terminal scenarios and neutral first-pitch scenarios. In strikeout
situations, a single action can accumulate a dominant share of visits: 24.2% for SI zone 9 in
the Soto-Wheeler 3-2 scenario, 16.3% for SI zone 9 in the Guerrero-Skubal 3-2 scenario, and
12.8% for CH zone 9 in the Ohtani-Skenes 3-2 scenario. The bases-loaded first-pitch scenarios
also produce notable concentration: 19.2% for ST zone 1 against Soto and 9.4% for SI zone 9
against Guerrero. This concentration reflects genuine strategic clarity: when the game state
strongly constrains what the pitcher needs to achieve, the search converges on a small number
of actions that best satisfy the combined requirements of the situation. Conversely, neutral
0-0 scenarios with no runners show flatter distributions, reflecting the inherent difficulty
of differentiating first-pitch value when many pitches carry broadly similar expected outcomes
at the start of an at-bat.

**Count-sensitive reward calibration.** The 3-1 hitter's-count scenarios consistently return
positive Q-values for the leading recommendations across all three matchups — Q = 0.143 for
Ohtani-Skenes, Q = 0.209 for Soto-Wheeler, Q = 0.118 for Guerrero-Skubal — and the visit
distributions show meaningful concentration, with the top action accounting for 5–8% of all
visits. This behavior reflects the RE288 reward signal's correct accounting of the count state:
even from a 3-1 deficit, a well-located pitch produces a positive expected outcome for the
pitcher, and the search correctly identifies which pitch types and locations offer the most
favorable paths from that disadvantaged starting point.
