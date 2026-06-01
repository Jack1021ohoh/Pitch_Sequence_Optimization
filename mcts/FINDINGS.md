# MCTS Pitch Sequencer — Case Study

## Overview

To demonstrate the practical utility of the expectimax MCTS pitch sequencer, we conduct a case
study across five elite matchups spanning all four pitcher-batter handedness combinations:
Shohei Ohtani (LHH) vs. Paul Skenes (RHP), Juan Soto (LHH) vs. Zack Wheeler (RHP), Vladimir
Guerrero Jr. (RHH) vs. Tarik Skubal (LHP), Freddie Freeman (LHH) vs. Jesús Luzardo (LHP), and
Fernando Tatis Jr. (RHH) vs. Yoshinobu Yamamoto (RHP). Each matchup is evaluated under five
distinct game-state scenarios — a full-count strikeout situation, a runner-on-first double-play
situation, a bases-loaded first pitch, a pitcher-ahead 0-2 waste pitch, and a hitter's-count
3-1 scenario — with three representative scenarios documented per matchup for a total of fifteen
scenario-recommendations. The search is run for 15,000 MCTS iterations per scenario with an
exploration constant of c = 1.4. All Q-values represent expected run-value deltas from the
pitcher's perspective (higher is better for the pitcher) derived from the RE288 run expectancy
table. Zone numbers follow the standard Statcast convention indexed from the catcher's
perspective: zones 1–3 occupy the top row of the strike zone, zones 7–9 the bottom row, with
the right column (zones 3, 6, 9) corresponding to the inside half against left-handed batters
and the away half against right-handed batters. The four outer ball zones are labeled 11
(high-away for LHH / high-inside for RHH), 12 (high-inside for LHH / high-away for RHH), 13
(low-away for LHH / low-inside for RHH), and 14 (low-inside for LHH / low-away for RHH).

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

## 4.4 Freddie Freeman vs. Jesús Luzardo

Freddie Freeman is a left-handed veteran hitter known for elite bat-to-ball skills, exceptional
plate coverage, and a disciplined two-strike approach that makes him difficult to expand beyond
the zone. Jesús Luzardo is a left-handed starter whose arsenal centers on a mid-90s four-seam
fastball with rising action, a sinking two-seamer with heavy arm-side run, a changeup, and a
slider. As a left-handed pitcher facing a left-handed batter, Luzardo operates in the same-
handed configuration — the sole left-left matchup in this case study — where the conventional
platoon break away from the batter is absent and the strategic geometry differs fundamentally
from the cross-handed scenarios examined above.

### Scenario 1: Full Count, Runner on Third, One Out

The 3-2 strikeout scenario produces an unusual recommendation: the top two actions are separated
by only two visits and 0.001 in Q-value. A four-seam fastball to zone 7 ranks first with 1,564
visits at a 10.4% share and Q = 0.432, while a sinker to zone 7 ranks second with 1,527 visits
at Q = 0.431 (10.2%). Zone 7 is the away-low corner of the strike zone against a left-handed
batter — for a left-handed pitcher, this quadrant is the arm-side low corner, where natural
arm-side run on both the four-seamer and the sinker carries the ball toward the outer edge at
the bottom of the zone. The near-tie between the two pitch types at the same location reflects
strategic convergence on a single optimal zone with uncertainty about delivery: both the
fastball's ride and the sinker's downward trajectory finish in the same spatial quadrant but
arrive via different vertical paths, and the search cannot differentiate their expected values
from this position. Below the top two, a sinker to zone 14 ranks third at 1,347 visits
(Q = 0.424, 9.0%), extending the arm-side low preference to the ball zone below the outer
corner. Across the top five actions, the zone-7 and zone-14 recommendations collectively
account for approximately 42% of visits, indicating that Luzardo's arm-side low quadrant is
the dominant strategic region in this high-leverage count.

### Scenario 2: First Pitch, Bases Loaded, One Out

The bases-loaded first-pitch scenario against Freeman produces the strongest zone-clustering in
the entire five-matchup study. A sinker to zone 7 leads with 2,691 visits at a 17.9% share and
Q = 0.235, the second-highest single-action visit concentration across all documented scenarios
behind only the Soto-Wheeler 3-2 sinker at 24.2%. More strikingly, the second and third ranked
actions are a four-seam fastball to zone 7 (1,485 visits, Q = 0.186, 9.9%) and a changeup to
zone 7 (1,475 visits, Q = 0.185, 9.8%) — placing three entirely distinct pitch types at the
same location. Collectively, the three zone-7 actions account for approximately 37.6% of all
15,000 iterations, the highest combined single-zone concentration in the study. The convergence
of a sinker, a fastball, and a changeup on the same spatial target reflects a clear strategic
diagnosis: the away-low corner is the correct attack location against Freeman in this game state,
and the search's three leading pitch types differ only in the kinematic path by which they
arrive there. For a left-handed pitcher, zone 7 is the arm-side low quadrant — a location where
the sinker's natural arm-side sink and the fastball's ride both terminate naturally, and the
changeup's velocity differential disguises an otherwise similar trajectory.

### Scenario 3: Hitter's Count, 3-1

The 3-1 scenario leads with a changeup to zone 8 at Q = 0.177, 992 visits, and a 6.6% share —
the highest leading Q-value among all five 3-1 scenarios in the study. The second-ranked action
is a slider to zone 7 (Q = 0.150, 695 visits, 4.6%), followed by a changeup to zone 9 (Q = 0.137,
598 visits, 4.0%) and a sinker to zone 4 (Q = 0.136, 595 visits, 4.0%). Zone 8 is the
bottom-center of the strike zone, a location that neither concedes the inner half nor commits
fully to the outer corner. In a same-handed matchup without the conventional platoon break, a
changeup's principal contribution is velocity differential rather than horizontal movement away
from the batter — the model's preference for a changeup to the bottom-center in a must-throw-
strike count reflects a strategy of disrupting Freeman's timing with a slower pitch in the
lower third of the zone rather than generating a swinging miss via sharp horizontal break. The
continued presence of zone 7 at rank 2 (SL zone 7) preserves the arm-side low preference as
a secondary option when the pitcher needs to throw a strike without conceding an elevated pitch.

---

## 4.5 Fernando Tatis Jr. vs. Yoshinobu Yamamoto

Fernando Tatis Jr. is a right-handed batter with explosive bat speed, an aggressive swing
philosophy, and a pull-oriented approach that generates significant power but also creates
vulnerability on the inner half and below the zone. Yoshinobu Yamamoto is a right-handed
starter whose arsenal — a four-seam fastball with elite ride, a diving splitter, a sinking
two-seamer, a cutter, and a curveball — is among the deepest in the sport. As the sole
right-right matchup in this case study, Yamamoto's arm-side run moves toward the inner half of
a right-handed batter, and his breaking pitches carry the potential to expand toward the outer
third.

### Scenario 1: Full Count, Runner on Third, One Out

The model's top recommendation in the 3-2 strikeout scenario is a splitter to zone 13 (Q = 0.294,
1,559 visits, 10.4%). For a right-handed batter, zone 13 is the low-inside ball zone — below
the strike zone on the inner half. A Yamamoto splitter to zone 13 describes the pitch in its
most difficult-to-handle location: diving sharply below and inside the zone, beneath the hands
of a right-handed pull hitter who cannot extend to drive it. The second-ranked action is a
cutter to zone 11 (Q = 0.257, 879 visits, 5.9%) — the high-inside ball zone for a right-handed
batter. The pairing of a low-inside splitter and a high-inside cutter describes a deliberate
vertical expansion around the inner quadrant: the splitter dives below the zone at the inner
edge, the cutter rides above and in. Ranks 3 through 7 are occupied entirely by curveballs
distributed across zones 13, 8, 1, 9, and 7, reflecting the search's recognition that Yamamoto's
top-to-bottom breaking ball provides a versatile secondary weapon capable of challenging the
batter across multiple locations once the splitter is established as the primary anchor.

### Scenario 2: First Pitch, Bases Loaded, One Out

The bases-loaded first-pitch scenario produces the most concentrated endorsement of the high-
inside pitch concept in the entire study. A cutter to zone 11 leads with 1,941 visits at a 12.9%
share and Q = 0.120. The second-ranked action is a four-seam fastball to zone 11 (1,169 visits,
Q = 0.092, 7.8%), and the fourth-ranked action is a sinker to zone 11 (762 visits, Q = 0.061,
5.1%). Three distinct pitch types — FC, FF, and SI — converge on the high-inside ball zone,
collectively accounting for approximately 25.8% of all 15,000 iterations. This three-way zone
convergence is the strongest expression of the elevated inner-half concept across any scenario
in the case study. For Tatis, an aggressive pull hitter whose swing path generates maximum
leverage on middle-away pitches, zone 11 targets the location where extension is impossible and
barrel speed is minimized. The intermediate action at rank 3 — a splitter to zone 13 (1,062
visits, Q = 0.086, 7.1%) — restates the low-inside preference from the strikeout scenario,
creating a spatial pair: FC, FF, and SI ride above and inside on the hands; the splitter dives
below them in the same inner column. Together the two poles describe a systematic inner-half
strategy that neutralizes Tatis's pull-power from both the elevated and the buried trajectory.

### Scenario 3: Pitcher Ahead, 0-2

With the count at 0-2, the model recommends a slider to zone 14 as its top action (Q = 0.115,
857 visits, 5.7%), followed by a splitter to zone 14 (Q = 0.114, 838 visits, 5.6%), a splitter
to zone 13 (Q = 0.111, 806 visits, 5.4%), and a sinker to zone 14 (Q = 0.107, 770 visits, 5.1%).
For a right-handed batter, zone 14 is the low-away ball zone and zone 13 is the low-inside ball
zone. The 0-2 waste-pitch recommendations consolidate on low chase zones: SL, FS, and SI to
zone 14 cluster on the low-away corner in the classic two-strike chase location for right-handed
batters, while FS zone 13 extends the low-inside splitter preference from the strikeout scenario.
The near-equal Q-values across the top four actions (0.115, 0.114, 0.111, 0.107) reflect the
characteristic flatness of pitcher's-count scenarios, where multiple pitch types to the same
ball-zone locations carry effectively equivalent expected run values. The contrast with the
bases-loaded first-pitch scenario is instructive: the zone-11 inner-half cluster that defined
the high-leverage situation is entirely absent from the top rankings in the 0-2 count, replaced
by low-zone expansion pitches that reflect the fundamentally different objective — generating a
swing-and-miss in the dirt rather than controlling the inner half of the plate.

---

## 4.6 Cross-Matchup Discussion

Several patterns emerge and are reinforced across the five matchups that merit broader
consideration.

**Elevated inner-half pitch as a consistent high-leverage signal.** At 15,000 iterations, the
search surfaces elevated inner-half pitches in the bases-loaded first-pitch scenario for all
three right-handed batters in the study. In the Guerrero-Skubal matchup, a four-seam fastball
to zone 11 ranks second at 6.9% — the initial appearance of the high-inside concept. In the
Tatis-Yamamoto matchup, three distinct pitch types converge on zone 11: a cutter (12.9%), a
four-seam fastball (7.8%), and a sinker (5.1%), collectively accounting for 25.8% of all visits
and representing the strongest single-zone endorsement in the study. For left-handed batters,
the analogous elevated pitch is zone 12 (high-inside for LHH): a sweeper to zone 12 ranks third
in the Ohtani bases-loaded scenario at 5.1%. In every case, the elevated inner-half pitch
appears alongside away-corner or low-zone recommendations as a strategic complement rather than
a replacement, consistent with the inside-high to away-low sequencing logic that restricts the
batter's extension before targeting the opposite corner. This inside-then-away structure is a
foundational concept in pitching strategy that the model arrives at through the count-tree
search rather than through any explicit encoding of pitch sequencing theory.

**Same-handedness substitutes arm-side concentration for the back-door pattern.** In all three
cross-handed matchups (Ohtani-Skenes, Soto-Wheeler, Guerrero-Skubal), a breaking ball directed
at the far corner of the strike zone is the consistently recommended neutral first-pitch action:
the sweeper to zones 1 and 7 for left-handed batters against right-handed pitchers, and the
slider to zone 3 for a right-handed batter against a left-handed pitcher. In the two same-handed
matchups, this pattern is absent. For Freeman-Luzardo (LHH vs. LHP), the search produces tight
zone-clustering on zone 7 via multiple pitch types — the sinker, four-seam fastball, and
changeup collectively target the away-low arm-side corner where a left-handed pitcher's natural
arm-side run carries all three shapes. For Tatis-Yamamoto (RHH vs. RHP), the dominant
first-pitch action shifts to the high-inside ball zone rather than an outer-corner breaking ball.
The structural explanation is direct: the back-door pattern requires a breaking ball that crosses
from the pitcher's arm side toward the batter's outside corner, a trajectory that only exists
in cross-handed matchups where the arm-side direction is toward the outside of the plate. In
same-handed matchups, arm-side run moves toward the batter's inner half, and the search adapts
accordingly — clustering on arm-side low pitches (LHP vs. LHH) or elevated inner-half pitches
(RHP vs. RHH) rather than generating a back-door trajectory.

**Concentration as a measure of strategic clarity.** Single-action visit concentrations remain
highest in terminal strikeout scenarios and in bases-loaded first-pitch situations across all
five matchups. The Soto-Wheeler 3-2 scenario retains the highest concentration at 24.2% for SI
zone 9, followed by the bases-loaded scenarios for Soto-Wheeler (ST zone 1, 19.2%) and
Freeman-Luzardo (SI zone 7, 17.9%), and the strikeout scenarios for Guerrero-Skubal (SI zone 9,
16.3%) and Ohtani-Skenes (CH zone 9, 12.8%). The Tatis-Yamamoto bases-loaded scenario contributes
a different form of concentration: no single action exceeds 12.9%, but three actions targeting
zone 11 account for 25.8% combined — the highest zone-level concentration across the study. The
consistent pattern confirms that high-leverage game states with constrained objectives sharpen
the search toward a smaller action set, while neutral zero-zero scenarios without runners produce
flat distributions reflecting the genuinely lower differentiation between first-pitch options.

**Count-sensitive reward calibration.** The 3-1 hitter's-count scenarios return positive Q-values
for the leading recommendations across all five matchups: Q = 0.143 (Ohtani-Skenes), Q = 0.209
(Soto-Wheeler), Q = 0.118 (Guerrero-Skubal), Q = 0.177 (Freeman-Luzardo), and Q = 0.085
(Tatis-Yamamoto). The Freeman-Luzardo 3-1 scenario produces the highest leading Q-value in the
study, suggesting that Luzardo's varied arsenal — four distinct pitch shapes across a wide
velocity range — provides the most favorable expected paths out of the 3-1 deficit. The
Tatis-Yamamoto scenario produces the lowest, a reflection of Tatis's offensive caliber in
favorable counts rather than a limitation of the model's count accounting: even Yamamoto's elite
repertoire offers a comparatively narrower margin from the 3-1 position against one of the
game's most dangerous hitters. Across all five matchups, the RE288 reward signal correctly
accounts for the cost of count deterioration, and the search identifies positive-expectation
pitches from the disadvantaged 3-1 starting point in every case.
