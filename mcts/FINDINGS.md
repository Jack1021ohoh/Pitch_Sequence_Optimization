# MCTS Pitch Sequencer — Case Study

## Overview

To demonstrate the practical utility of the expectimax MCTS pitch sequencer, we conduct a case
study across five elite matchups spanning all four pitcher-batter handedness combinations:
Shohei Ohtani (LHH) vs. Paul Skenes (RHP), Juan Soto (LHH) vs. Zack Wheeler (RHP), Vladimir
Guerrero Jr. (RHH) vs. Tarik Skubal (LHP), Freddie Freeman (LHH) vs. Jesús Luzardo (LHP), and
Fernando Tatis Jr. (RHH) vs. Yoshinobu Yamamoto (RHP). Each matchup is evaluated under five
distinct game-state scenarios — a full-count strikeout situation, a runner-on-first double-play
situation, a bases-loaded first pitch, a pitcher-ahead 0-2 waste pitch, and a hitter's-count
3-1 scenario — for a total of twenty-five scenario-recommendations. The search is run for
15,000 MCTS iterations per scenario with an exploration constant of c = 1.4. Zone numbers
follow the standard Statcast convention indexed from the catcher's perspective: zones 1–3
occupy the top row of the strike zone, zones 7–9 the bottom row, with the right column
(zones 3, 6, 9) corresponding to the inside half against left-handed batters and the away
half against right-handed batters. The four outer ball zones are labeled 11 (high-away for
LHH / high-inside for RHH), 12 (high-inside for LHH / high-away for RHH), 13 (low-away for
LHH / low-inside for RHH), and 14 (low-inside for LHH / low-away for RHH).

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
end the inning.

The search decisively favors a changeup to zone 9 as the top recommendation, drawing more than
twice the attention of any other action. A splitter to zone 9 ranks second, with a curveball to
zone 9 third. The concentration of top recommendations in zone 9 — the inside-low corner of the
strike zone against a left-handed batter — reflects the model's assessment that Skenes' off-speed
offerings finishing on the inner half carry the highest probability of inducing a strikeout from
Ohtani. The inside-low quadrant aligns with documented tendencies for left-handed batters to
expand their swing on pitches that appear to catch the zone on the inner edge, and Skenes'
splitter and changeup are among his most swing-and-miss pitches when spotted at or below the
inner quadrant.

### Scenario 2: First Pitch, Runner on First, One Out

With the count at 0-0, a runner on first, and one out, the primary objective shifts to inducing
early-count ground-ball contact for a double play.

The distribution is notably flat, with no single action standing clearly apart from the field.
A sweeper to zone 7 leads, barely ahead of a sweeper to zone 1, and all top recommendations
carry only marginally positive expected outcomes. The leading presence of the sweeper to both
away corners (zones 7 and 1) is consistent with the broader pattern for Skenes against
left-handed hitters: even in the flattest distribution of the Ohtani-Skenes matchup, the
back-door away-corner preference remains the weakly dominant signal.

### Scenario 3: First Pitch, Bases Loaded, One Out

With the count at 0-0 and the bases loaded, the first pitch carries enormous weight. A walk
immediately forces in a run; a hit could clear the bases.

The model's top recommendation is a sweeper to zone 7, the away-low corner, receiving
substantially more attention than any other action. A sweeper to zone 1, the away-high corner,
ranks second. The third recommendation is a sweeper to zone 12 — the high-inside ball zone for
a left-handed batter. The joint emergence of three sweeper locations — away-low, away-high, and
high-inside — reveals a multi-dimensional picture of how Skenes' sweeper is deployed in this
context. The two away-corner locations (zones 7 and 1) describe the classic back-door trajectory:
a pitch that appears to start off the outside edge and sweeps across the outer third of the zone
at varying heights. The zone 12 recommendation introduces a contrasting look: a sweeper elevated
above the zone on the inside corner, designed to generate a weak swing above the hands or a
called strike if the batter freezes. The appearance of an elevated ball-zone location among the
top three — where lower-iteration searches had not differentiated it from the field — indicates
that the additional search depth has allowed the MCTS to identify elevated pitches as a
meaningful strategic option alongside the away-corner approach.

### Scenario 4: Pitcher Ahead, 0-2

With the count at 0-2 and no runners, Skenes holds maximum leverage within the at-bat and can
afford to give up a ball.

Five distinct pitch types all converge on zone 14 — the low-inside ball zone for a left-handed
batter — at nearly equal expected outcomes: splitter, curveball, slider, sweeper, and changeup
all target zone 14 in the top five, followed by several pitches to zone 11 (the high-away ball
zone for LHH). The preferred chase location against Ohtani in the 0-2 count is the low-inside
ball zone rather than the more conventional low-away. Skenes' splitter and changeup break
naturally toward the inner half with downward action, making zone 14 the natural terminus for
his two best swing-and-miss pitches. The characteristic flatness across pitch types reflects
the 0-2 scenario's inherent equivalence — the game state already strongly favors the pitcher.

### Scenario 5: Hitter's Count, 3-1

With the count at 3-1 and no runners, the pitcher cannot afford a walk.

The top recommendation is a changeup to zone 7, the away-low corner, closely followed by a
sweeper to zone 1, the away-high corner. Both pitches target the away half with distinct velocity
profiles. The pairing of a changeup and a sweeper to the away side in a must-throw-strike count
reflects a strategy of attacking the outer half with pitches that carry the trajectory of a ball
until they clip the zone: the batter, geared up for a fastball, must commit to off-speed pitches
breaking toward the outer corner without generating leverage. Both leading options produce
meaningfully favorable outcomes for the pitcher even from the disadvantageous 3-1 count.

---

## 4.2 Juan Soto vs. Zack Wheeler

Juan Soto is a left-handed hitter renowned for his patience and strike-zone discipline, drawing
walks at an elite rate while posting exceptional on-base percentages. Zack Wheeler is a
right-handed starter whose arsenal is anchored by a heavy two-seam sinker that generates
exceptional ground-ball rates, complemented by a hard slider and a four-seam fastball.

### Scenario 1: Full Count, Runner on Third, One Out

The 3-2 strikeout scenario against Soto produces the most concentrated recommendation in the
entire case study. A sinker to zone 9 — the inside-low corner against a left-handed batter —
draws nearly two and a half times the attention of the second-ranked action, a four-seam
fastball to zone 9. This exceptional concentration is notable given that Soto is widely
considered one of the most patient and disciplined hitters in the game, a batter who rarely
chases and makes pitchers work deep into counts. The sinker to zone 9 places a pitch with heavy
downward and arm-side movement at the inside-low corner, a location that both maximizes the
risk of a called strike on a full count and, if contacted, tends to produce weak ground balls
or rollover contact due to the pitch's late downward bite. The model's overwhelming confidence
in this single recommendation suggests the search has identified Wheeler's sinker at the inner
edge as an outlier — a location where Wheeler's movement profile and Soto's tendencies converge
on a uniquely favorable expected outcome.

### Scenario 2: First Pitch, Runner on First, One Out

The double-play scenario against Soto produces the only uniformly negative expected-outcome
distribution in the case study. A sweeper to zone 1 leads, followed by a sweeper to zone 7,
but every top-ranked action is expected to result in a slight batter advantage rather than a
pitcher advantage. The sweeper to the away corners still leads the rankings, maintaining the
away-corner preference seen in higher-leverage scenarios, but no pitch offers a favorable
expected run-value delta against Soto in this game state. This is a direct reflection of
Soto's elite plate discipline: even with a runner on first and one out, the model cannot
identify a first pitch expected to benefit the pitcher. The best available pitch merely
minimizes the expected loss rather than generating a gain.

### Scenario 3: First Pitch, Bases Loaded, One Out

The bases-loaded first-pitch scenario produces the second most concentrated recommendation in
the case study. A sweeper to zone 1 — the away-high corner against a left-handed batter —
receives nearly three times the attention of any other action. Zone 1 targets the very top of
the away side of the zone: a pitch that starts off the outer edge and breaks back toward the
upper-outside corner. Against Soto, whose approach is built on letting pitches travel deep into
the zone before committing, a pitch that clips the away corner at the top poses a particular
challenge: taking it risks a called strike, while swinging requires extension at an awkward
angle above the hands. The contrast with Scenario 2 is notable: the same batter who offers no
positive first-pitch options in the double-play state becomes clearly attackable when the bases
are loaded, because the run-expectancy gap between retiring the batter and allowing a walk grows
sharply with the bases full.

### Scenario 4: Pitcher Ahead, 0-2

With the count at 0-2 and no runners, Wheeler holds maximum leverage within the at-bat. The
top recommendation is a curveball to zone 14, the low-away ball zone for a left-handed batter,
followed closely by a splitter to zone 14 and a curveball to zone 11, the high-away ball zone.
The appearance of zone 14 at the top confirms that the model correctly identifies low-zone
expansion pitching as optimal when the pitcher can afford a ball. The continued presence of a
sinker to zone 9 among the top-ranked actions, even in the waste-pitch scenario, reflects
Wheeler's tendency to challenge hitters even when ahead, using his heavy sinker on the inside
corner as a pitch that can generate a strikeout swing rather than merely burning a ball.

### Scenario 5: Hitter's Count, 3-1

The 3-1 scenario produces the strongest first-pitch recommendation across all five matchups'
3-1 scenarios. A sweeper to zone 7 leads decisively — more than a third more attention than the
second-ranked action, a sweeper to zone 1. The back-door sweeper pattern that dominates the
bases-loaded first-pitch scenario carries into the must-throw-strike count: even when Wheeler
cannot afford a ball, the model recommends leading with a breaking ball to the away corners
rather than a fastball or sinker into a hittable zone. The curveball to zone 7 and the sweeper
to zone 4 round out the top four, all targeting the away half. Wheeler's sweeper to the outer
corner is particularly well-matched to Soto's tendencies even under count pressure.

---

## 4.3 Vladimir Guerrero Jr. vs. Tarik Skubal

Vladimir Guerrero Jr. is a right-handed hitter with elite bat speed and a pull-heavy approach,
capable of driving pitches on the inner half for extra bases. Tarik Skubal is a left-handed
starter whose effectiveness derives primarily from a heavy sinker and an elite changeup,
supplemented by a sweeper and a curveball that exploit arm-side break against right-handed
hitters.

### Scenario 1: Full Count, Runner on Third, One Out

The model's top recommendation in the 3-2 strikeout scenario is a sinker to zone 9, drawing
substantially more attention than any other action. For a right-handed batter, zone 9 is the
away-low corner — the precise location where a left-handed pitcher's sinker tails away from
the batter's hands with downward and arm-side movement. A sinker finishing at the bottom-away
quadrant against Guerrero targets a location simultaneously difficult to pull for damage and
naturally suited to the pitch's movement profile. The second recommendation, a sinker to zone 7,
extends the away-low theme to the adjacent corner, suggesting a preference for the entire
outer-low portion of the zone rather than a single precise location.

### Scenario 2: First Pitch, Runner on First, One Out

With a runner on first and one out, the objective shifts to inducing a ground ball for a double
play. The top recommendation is a slider to zone 3 — the away-high corner for a right-handed
batter — followed by a sinker to zone 9 and a four-seam fastball to zone 9. A left-handed
pitcher's slider to the away-high corner describes a pitch that breaks toward the outer portion
of the zone at the top — a sweeping breaking ball moving away from the barrel against a
pull-oriented right-handed hitter. This recommendation mirrors, from the left-handed pitcher's
perspective, the back-door sweeper pattern seen in the Skenes and Wheeler matchups: a breaking
ball using horizontal movement to target the far corner of the zone.

### Scenario 3: First Pitch, Bases Loaded, One Out

The top action is a sinker to zone 9, consistent with the strikeout scenario's preference for
the away-low corner. The second-ranked action, however, is a four-seam fastball to zone 11 —
the high-inside ball zone for a right-handed batter — receiving well more than half the
attention of the top action and clearly separating itself from the remaining field. Zone 11 is
the elevated tight fastball intended to jam the hitter, push him off the plate, or generate a
weak pop-up. Against Guerrero, who generates his power by extending through the ball to the
pull side, zone 11 targets the exact location that limits his ability to drive the ball. The
two top recommendations together describe a coherent inside-outside, high-low sequencing
concept: set up the batter with a high tight fastball to restrict extension, then put the
sinker away-low where he cannot extend to pull it.

### Scenario 4: Pitcher Ahead, 0-2

With the count at 0-2, five different pitch types all converge on zone 13 — the low-inside ball
zone for a right-handed batter — at nearly equal expected outcomes: curveball, slider, sinker,
changeup, and four-seamer all appear in the top five targeting zone 13. Zone 14 (low-away for
RHH) also appears in the upper rankings as an alternative. For Skubal, a left-handed pitcher,
zone 13 (low-inside for RHH) is the arm-side low direction where his sinker and changeup tail
naturally. The 0-2 waste-pitch scenario therefore extends the arm-side tendency observed
throughout this matchup: even the chase zone targets the inner half, consistent with Skubal's
natural movement profile.

### Scenario 5: Hitter's Count, 3-1

A slider to zone 3 leads the 3-1 recommendations, the same back-door pitch that topped the
double-play scenario. Zone 9 (away-low for RHH) then appears five consecutive times: changeup,
sinker, slider, curveball, and four-seamer all target the same outer-low corner. The 3-1 count
produces a clear two-zone structure against Guerrero: the slider reaches the away-high corner
via its break (zone 3), while five different pitch types are commanded to the away-low corner
(zone 9). The consistent preference for the outer half across both zones reflects a coherent
strategy of attacking Guerrero's pull tendency by forcing him to the opposite field.

---

## 4.4 Freddie Freeman vs. Jesús Luzardo

Freddie Freeman is a left-handed veteran hitter known for elite bat-to-ball skills, exceptional
plate coverage, and a disciplined two-strike approach that makes him difficult to expand beyond
the zone. Jesús Luzardo is a left-handed starter whose arsenal centers on a mid-90s four-seam
fastball, a sinking two-seamer, a changeup, a slider, and a sweeper. As a left-handed pitcher
facing a left-handed batter, Luzardo operates in the same-handed configuration — the sole
left-left matchup in this case study — where the pitcher's arm-side run moves toward the inner
half of the batter rather than away from it, and the conventional platoon break is absent.

### Scenario 1: Full Count, Runner on Third, One Out

The 3-2 strikeout scenario produces an unusual recommendation: the top two actions are
essentially tied. A four-seam fastball to zone 7 ranks first with only a two-visit margin over
a sinker to zone 7. Zone 7 is the away-low corner of the strike zone against a left-handed
batter. The near-tie between the two pitch types at the same location reflects strategic
convergence on a single optimal zone with uncertainty about delivery: both the fastball and the
sinker are directed to the outer edge, against their natural arm-side run that would carry them
inside. The model values the zone itself more than the kinematic path by which the pitch reaches
it. Below the top two, a sinker to zone 14 ranks third, extending the away preference to the
ball zone below the outer corner. Zone-7 and zone-14 recommendations dominate the top five
actions collectively, indicating that the away-low quadrant is the dominant strategic region in
this high-leverage count.

### Scenario 2: First Pitch, Runner on First, One Out

The double-play first-pitch scenario shows moderate zone-7 clustering. A sinker to zone 7 leads,
followed by a changeup to zone 7 and a four-seam fastball to zone 7. A sinker to zone 1
(away-high) ranks fourth. Three of the top four actions target the away side, and all top
recommendations carry positive expected outcomes — unlike the Soto-Wheeler double-play scenario
where every option was negative — suggesting that Luzardo maintains a favorable expected outcome
against Freeman even in a neutral game state. The zone-7 preference is consistent across the
double-play and higher-leverage scenarios, though the concentration weakens significantly without
the pressure of bases loaded.

### Scenario 3: First Pitch, Bases Loaded, One Out

The bases-loaded first-pitch scenario produces the strongest zone-clustering in the five-matchup
study. A sinker to zone 7 leads by a wide margin, followed by a four-seam fastball to zone 7
and a changeup to zone 7 — the second and third ranked actions are nearly tied with each other
and both targeting the same location as the leader. Three entirely distinct pitch types
converge on zone 7, collectively accounting for the highest combined single-zone concentration
in the study. The zone-7 clustering includes pitches that naturally move there (the slider and
sweeper, whose glove-side break carries them toward zone 7 for a LHP) alongside pitches directed
there against their natural run (the sinker and fastball, which would naturally tail to the
inner half). The model's simultaneous recommendation of arm- and glove-side pitches to the
same away-low target demonstrates that the location's strategic value against Freeman outweighs
movement-direction considerations.

### Scenario 4: Pitcher Ahead, 0-2

With the count at 0-2, the waste-pitch distribution is flat and spread across zones 13 and 14.
A sweeper to zone 13 and a slider to zone 13 share the top ranking — the low-away ball zone for
a left-handed batter — immediately followed by a changeup to zone 14 (low-inside ball zone for
LHH). The LHP's sweeper and slider, whose glove-side break carries them toward the away side
for LHH, appear as the natural chase pitches to zone 13. The presence of sinker and fastball
to zone 7 in the lower top-10 rankings indicates the away-low preference persists even in the
waste-pitch scenario, though diffused across a broader set of options at similar expected
outcomes.

### Scenario 5: Hitter's Count, 3-1

The top recommendation is a changeup to zone 8 — the bottom-center of the strike zone —
receiving notably more attention than the second-ranked action, a slider to zone 7. Zone 8 is
a location that neither concedes the inner half nor commits fully to the outer corner. In a
same-handed matchup without the conventional platoon break, a changeup's principal contribution
is velocity differential rather than horizontal movement away from the batter. The model's
preference for a changeup to the bottom-center in a must-throw-strike count reflects a strategy
of disrupting Freeman's timing with a slower pitch in the lower third of the zone. The slider
to zone 7 at rank 2 preserves the away-low preference as a secondary option. This matchup
produces the most favorable pitcher outcomes among all five 3-1 scenarios, suggesting Luzardo's
varied four-shape arsenal provides a wider set of viable strike-throwing paths from the
disadvantaged count.

---

## 4.5 Fernando Tatis Jr. vs. Yoshinobu Yamamoto

Fernando Tatis Jr. is a right-handed batter with explosive bat speed, an aggressive swing
philosophy, and a pull-oriented approach that generates significant power but creates
vulnerability on the inner half and below the zone. Yoshinobu Yamamoto is a right-handed
starter whose arsenal — a four-seam fastball with elite ride, a diving splitter, a sinking
two-seamer, a cutter, and a curveball — is among the deepest in the sport. As the sole
right-right matchup in this case study, Yamamoto's arm-side run moves toward the inner half of
a right-handed batter, and his breaking pitches carry the potential to expand toward the outer
third.

### Scenario 1: Full Count, Runner on Third, One Out

The top recommendation in the 3-2 strikeout scenario is a splitter to zone 13 — the low-inside
ball zone for a right-handed batter. A Yamamoto splitter to zone 13 describes the pitch in its
most difficult-to-handle location: diving sharply below and inside the zone, beneath the hands
of a right-handed pull hitter who cannot extend to drive it. The second-ranked action is a cutter
to zone 11 — the high-inside ball zone for a right-handed batter. The pairing of a low-inside
splitter and a high-inside cutter describes deliberate vertical expansion around the inner
quadrant: the splitter dives below the zone at the inner edge, the cutter rides above and in.
Ranks 3 through 7 are occupied entirely by curveballs distributed across multiple zones,
reflecting the search's recognition that Yamamoto's curveball provides a versatile secondary
weapon capable of challenging the batter across multiple locations once the splitter is
established as the primary anchor.

### Scenario 2: First Pitch, Runner on First, One Out

The double-play first-pitch scenario produces the flattest distribution in the Tatis-Yamamoto
matchup. A cutter to zone 11 barely leads, followed closely by a splitter to zone 13, a slider
to zone 9, and a splitter to zone 9 — all essentially equal. The zone-11 high-inside preference
that dominates the bases-loaded scenario is only faintly present here, suggesting the inner-half
signal is a high-leverage discovery that weakens substantially when the game state is neutral.
No action meaningfully separates itself from the field, confirming that early-count first-pitch
differentiation is genuinely low across all five matchups in the double-play scenario.

### Scenario 3: First Pitch, Bases Loaded, One Out

The bases-loaded first-pitch scenario produces the most concentrated endorsement of the
high-inside pitch concept in the entire study. A cutter to zone 11 leads by a wide margin,
followed by a four-seam fastball to zone 11 and — after a splitter to zone 13 at rank 3 — a
sinker to zone 11 at rank 4. Three distinct pitch types — cutter, four-seamer, and sinker —
converge on the high-inside ball zone, collectively accounting for more than a quarter of all
iterations. This three-way zone convergence is the strongest expression of the elevated inner-
half concept across any scenario in the case study. For Tatis, an aggressive pull hitter whose
swing path generates maximum leverage on middle-away pitches, zone 11 targets the location where
extension is impossible and barrel speed is minimized. The splitter to zone 13 at rank 3
restates the low-inside preference from the strikeout scenario, creating a spatial pair: the
cutter, four-seamer, and sinker ride above and inside on the hands while the splitter dives
below them in the same inner column, describing a systematic inner-half strategy that attacks
Tatis from both the elevated and buried trajectories.

### Scenario 4: Pitcher Ahead, 0-2

With the count at 0-2, the recommendations consolidate on low chase zones. A slider to zone 14
leads — the low-away ball zone for a right-handed batter — followed immediately by a splitter
to zone 14, a splitter to zone 13 (low-inside), and a sinker to zone 14. The near-equal
expected outcomes across the top four actions reflect the characteristic flatness of pitcher's-
count scenarios, where the game state already so strongly favors the pitcher that multiple
pitch-location combinations carry broadly equivalent value. The contrast with the bases-loaded
first-pitch scenario is instructive: the zone-11 inner-half cluster is entirely absent from the
top rankings here, replaced by low-zone expansion pitches that serve the different objective of
generating a swing-and-miss in the dirt rather than controlling the inner half of the plate.

### Scenario 5: Hitter's Count, 3-1

A splitter to zone 9 leads — the away-low corner for a right-handed batter — closely followed
by a slider to zone 9, a slider to zone 6 (away-middle), and a cutter to zone 9. The
must-throw-strike count against Tatis produces away-zone recommendations rather than the
inner-half zone-11 cluster that dominated the bases-loaded scenario. When the pitcher needs
a strike, the outer-low corner becomes the primary target, with elevated inside pitches dropping
out of the top rankings entirely. The away-low corner of the zone emerges as the correct
strike-throwing target against Tatis from the disadvantaged count, contrasting with the
inner-half attack that defines the high-leverage, first-pitch scenarios.

---

## 4.6 Cross-Matchup Discussion

Several patterns emerge and are reinforced across the five matchups that merit broader
consideration.

**Elevated inner-half pitch as a consistent high-leverage signal.** At 15,000 iterations, the
search surfaces elevated inner-half pitches in the bases-loaded first-pitch scenario for all
three right-handed batters in the study. Against Guerrero (RHH vs. LHP), a four-seam fastball
to zone 11 ranks second, clearly separated from the remaining field. Against Tatis (RHH vs.
RHP), three distinct pitch types — cutter, four-seamer, and sinker — all converge on zone 11,
collectively accounting for more than a quarter of all iterations. For left-handed batters, the
analogous elevated pitch is zone 12 (high-inside for LHH): a sweeper to zone 12 ranks third
in the Ohtani bases-loaded scenario. In every case, the elevated inner-half pitch appears
alongside away-corner or low-zone recommendations as a strategic complement rather than a
replacement, consistent with the inside-high to away-low sequencing logic that restricts the
batter's extension before targeting the opposite corner. This pairing is a foundational concept
in pitching strategy that the model arrives at through count-tree search rather than through
any explicit encoding of pitch sequencing theory.

**Away-corner preference for LHH persists regardless of pitcher handedness; mechanism changes.**
In all three LHH matchups — Ohtani vs. Skenes (RHP), Soto vs. Wheeler (RHP), and Freeman vs.
Luzardo (LHP) — the away-low corner (zone 7) and away-high corner (zone 1) consistently occupy
the top first-pitch recommendations. For the cross-handed RHP matchups, this is driven by the
back-door sweeper: a pitch that uses its natural horizontal break to start off the outer edge
and sweep across the far side of the zone. For the same-handed LHP matchup, the same
away-corner zones are recommended, but via a different mechanism — multiple pitch types
including sinkers and fastballs directed against their natural arm-side run are commanded to
zone 7, suggesting that the location's strategic value against Freeman outweighs movement-
direction considerations. The same-handed matchup reveals that the away-low corner is the
correct attack zone for left-handed batters regardless of how the pitcher's arsenal reaches it.

**RHH inner-half preference also persists across handedness.** In both RHH matchups, the inner
half dominates high-leverage scenarios. Against Guerrero (RHH vs. LHP), the sinker to zone 9
leads the strikeout scenario with zone-11 appearing as the key second recommendation in the
bases-loaded scenario. Against Tatis (RHH vs. RHP), a splitter to zone 13 leads the strikeout
scenario and three pitch types converge on zone 11 in the bases-loaded scenario. The inner-half
preference for right-handed pull hitters is consistent across left-handed and right-handed
pitcher matchups, suggesting it is a batter-specific finding rather than a pitcher-handedness
artifact.

**Concentration as a measure of strategic clarity.** Single-action concentration is highest in
terminal strikeout scenarios and in bases-loaded first-pitch situations across all five matchups.
The most decisive individual recommendation in the study is Wheeler's sinker to zone 9 in the
Soto 3-2 scenario, drawing nearly two and a half times the attention of the next-best action.
Freeman's bases-loaded scenario produces the most extreme zone-level concentration, with three
pitch types all pointing to the same away-low corner. The Tatis bases-loaded scenario produces
the strongest multi-pitch convergence on a single ball zone (zone 11), with three pitch types
collectively accounting for more than a quarter of all iterations. Conversely, the double-play
first-pitch scenarios produce the flattest distributions across all five matchups — confirming
that the difficulty of differentiating first-pitch value in neutral game states reflects genuine
low signal rather than insufficient iteration budget.

**Negative expected outcomes in the Soto double-play scenario.** The only uniformly negative
expected-outcome distribution in the study appears in the Soto-Wheeler double-play scenario:
every first-pitch option, including the best available, is expected to result in a slight batter
advantage. This stands in sharp contrast to the same matchup's bases-loaded scenario, where the
top recommendation is strongly pitcher-favorable. The RE288 framework explains the asymmetry:
with the bases empty and a runner on first, the run-expectancy starting point already reflects
Soto's elite offensive quality, and no single pitch can shift the expected outcome to the
pitcher's favor. With the bases loaded, the potential run-expectancy reduction from retiring
Soto without allowing a run is large enough that well-located pitches generate clearly favorable
pitcher outcomes.

**Count-sensitive reward calibration.** The 3-1 hitter's-count scenarios return favorable
pitcher outcomes for the leading recommendations across all five matchups. The Freeman-Luzardo
3-1 scenario produces the most favorable outcomes, suggesting Luzardo's varied four-shape
arsenal provides the widest set of viable strike-throwing paths from the disadvantaged count.
The Soto-Wheeler 3-1 scenario produces the second most favorable, driven by Wheeler's sweeper
to the away corners — the same pitch that dominates the bases-loaded scenario carrying its
strategic value into the must-throw-strike count. In every matchup, the search correctly
identifies positive-expectation pitches from the 3-1 starting point, confirming that RE288's
count-state reward signal gives the pitcher credit for throwing quality strikes even when the
count is unfavorable.
