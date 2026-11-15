# More than a Hack 2027, Phase 1 (ball possession): rules summary

Source: <https://github.com/metrica-sports/more-than-a-hack-2027>, commit `f8f2629ff1e0985f81d7e59d8413678514ee1530`
(read on 2026-11-15). The repo says it is authoritative: DATA.md for data, EVALUATION.md for scoring, RULES.md for what is
allowed, and `evaluation/possession_metrics.py` for the metric definitions. Where this summary and the repo disagree, the
repo wins. The repo is checked out read-only as the submodule `reference/more-than-a-hack-2027/`.

## 1. The task and the labels

One match: **FC Barcelona (home, `H`) vs Athletic Club (away, `A`)**, LaLiga 2024-25, matchday 2. You write one label for
each of the **199,319 frames of the tactical video** (25 fps). 158,857 of those frames are graded: the 1st half is video
frames 11,185–90,651 and the 2nd half is 110,808–190,197.

| label | meaning |
|---|---|
| `H` | home in possession, ball alive |
| `A` | away in possession, ball alive |
| `D` | dead ball: the ball is not in play. Nobody "owns" a dead ball. |
| `X` | no answer. Free on ungraded frames. On graded frames it costs differently per metric (§2). |

Possession definitions (CHALLENGE.md):
- While the ball is alive, **one team always has possession**. There is no "contested", "transition" or "unknown" in the
  ground truth (GT). Every graded frame is H, A or D.
- Possession changes only when an opponent gains **controlled** possession. **Contested or loose balls stay with the
  last team that had possession.** These do not change possession: a simple touch without control, a deflection that does
  not intercept the pass, or an aerial duel won without that team then controlling the ball. Their suggested proxies for
  control: a player keeps the ball for a few seconds, or a team completes more than two consecutive passes.
- After a restart there is always a clear possessor.
- A possession is a **maximal run of H, or of A**. `H H D D H H` counts as two possessions. D is a stoppage, never a
  possession.
- **The GT is derived mechanically from Metrica's *corrected* tracking.** A frame is D wherever the corrected tracking
  marks the ball dead. Otherwise it takes the team that the corrected tracking names as owning the ball. It is not
  smoothed and not hand-adjusted. It is not taken from event data and not from a human watching the video. "The 5 public
  minutes show what that means; if your reading of the definition disagrees with them, the GT is right."
- Trap: the 2nd half is graded from frame 110,808, but kick-off is not until **113,742**. Those **2,934 frames are graded
  and are all D**. The 1st half is graded from its kick-off frame.

## 2. Scoring (`evaluation/possession_metrics.py`, `run_evaluation.py`)

- **Ranking: `spellF1` (`spell_f1_mean`).**
  - Spells are the maximal runs of H or of A, built separately inside each graded run. The graded runs are the
    contiguous stretches where the GT is not X, which here means the two halves.
  - GT and predicted spells are matched **one-to-one, same team only**, by temporal IoU (intersection / union of the
    two frame intervals).
  - The matching is greedy: pairs are taken in order of descending IoU, ties broken by (gt index, pred index).
  - At each IoU threshold in **0.3, 0.4, 0.5, 0.6, 0.7**: precision = matched / predicted spells, recall = matched / GT
    spells, then F1. spellF1 is the **mean of the five F1s**. A headline F1 at 0.5 is also reported.
  - D frames are not spells. Dead ball is scored only through macroF1 and the extra metrics.
  - **X is forward-filled** with the last non-X label in the same half (back-filled at the start of a half) before
    spells and transitions are computed. So on the ranking metric, X means "repeat my last answer".
- **First tie-break: `macroF1`.** This is the unweighted mean of the per-class F1 for H, A and D over the graded frames.
  A predicted X on a graded frame counts as a miss (a false negative). It is never a false positive.
- **Further tie-breaks:** transition F1 at τ = 25 frames, then mean absolute latency, then filename. Transitions are
  matched one-to-one within ±12 and ±25 frames and must have the same *destination* label.
- **Noise bands:** spellF1 gaps below about **0.04** are noise on the full match; macroF1 gaps below about **0.01** are
  noise.
- **Timing sensitivity:** a 1-frame lag is free; 3 frames late gives 0.98, 8 frames 0.89, **1 s gives 0.65**.
- **Baselines.** All four are the same algorithm: the nearest tracked player to the ball, then a majority vote over
  ±0/25/50/125 frames. Frames without a visible ball are D.

  | baseline | full match macroF1 | full match spellF1 | public 5 min macroF1 | public 5 min spellF1 |
  |---|---|---|---|---|
  | baseline_raw | 0.6532 | 0.0486 | 0.5972 | 0.0378 |
  | baseline_smooth25 | 0.6752 | 0.1996 | 0.6200 | 0.2868 |
  | **baseline_smooth50 (published floor)** | 0.6724 | **0.2433** | 0.6207 | 0.3256 |
  | baseline_smooth125 | 0.6356 | 0.2242 | 0.5812 | 0.4242 |

  On the public window, smooth125 looks best; on the full match, smooth50 wins. **The public 5 minutes cannot rank
  versions of a pipeline.**
- No statistic of the full-match GT is published (possession count, durations, share). The public window has 17
  possessions with a mean of 14.2 s, and 21 segments (H 3,454 / A 2,577 / D 1,469 frames).

**Run it locally** (checked 2026-11-15 in this repo's `.venv`, Python 3.13.0, numpy 2.5.4, pandas 3.0.6; it reproduces
the public-window numbers above exactly):

```powershell
.\.venv\Scripts\python.exe reference\more-than-a-hack-2027\evaluation\run_evaluation.py --out outputs\scorer
.\.venv\Scripts\python.exe reference\more-than-a-hack-2027\evaluation\validate_submission.py <file>.csv
```

Useful options: `--results <dir>` (score every CSV in a folder), `--gt public|private|<path>`, `--bootstrap 1000`
(confidence intervals), `--no-detail`. By default it writes `scoreboard.csv` and `reports/` into `evaluation/`, so pass
`--out outputs\scorer` to keep the submodule clean.

## 3. The data (DATA.md; not downloaded yet, so none of this has been checked by us)

| | tactical (the reference) | broadcast (optional) |
|---|---|---|
| video | 199,319 frames, 25 fps, 1920×1080 | its own timeline, with cuts and replays; **not synchronised, and no mapping is provided** |
| tracking (ATD) | 199,319 records, 46 track slots | 164,488 records, 51 slots |
| homography | 199,319 rows, 100 % valid | 164,488 rows, 65.8 % valid |
| Smart Tagging | 638 instances | none |
| ball visible | 56.9 % of all frames, **59.3 % of graded frames** (41 % missing) | 50.5 % |

- **ATD** (Metrica FIFA/EPTS export: a metadata `.xml` plus a raw `.txt`). Each line looks like
  `frame_id:p1;p2;...;p46:ball`, and each group is `x,y,speed`, with NaN where missing.
  - `frame_id` runs 1…199,319, so **video_frame = frame_id − 1**.
  - Coordinates are **normalised 0–1**. The pitch is 105.3 × 68.0 m (`games.py`, measured from the corrected tracking).
    The metadata says 105 × 68.
  - Origin and axis direction follow kloppy's Metrica convention. We have not checked them, and we have not checked
    which side each team attacks.
  - Speed is in m/s.
  - **No player identities.** The 46 slots are tracklets ("Track 1…46"), and `ShirtNumber` is just the slot index.
    Slots appear, vanish and swap identity.
  - **Each slot has a team** (`teamId` in the metadata; the first team listed is home).
  - About 20.5 tracked objects per graded frame on average, sometimes more than 22 when the tracker duplicates someone.
  - Gaps, fragments and ID switches are expected.
  - The ball is a third group of x, y and a third value. Nothing says whether it includes z.
- **Smart Tagging** (SportsCode XML, automatic, **not** ground truth). Times are seconds from the first frame of the
  tactical video, so frame = round(s × 25).
  - Codes: phases (BUILD UP, PROGRESSION, CREATING CHANCES, DEFENDING IN …, ATTACKING / DEFENSIVE TRANSITION, LONG BALL,
    BALL IN FINAL THIRD / THE BOX, PLAYERS IN THE BOX), plus SET PIECES (59) and GOALS (4).
  - Tags: `Team` (Barcelona 314 / "Atletic Club" 288 / N/A 36, **including every throw-in**), `Half`, and `Type` for set
    pieces (throw-in 26, corner 12, goal kick 10, free kick 5, kick-off 6).
  - The intervals are *phases*, not possessions.
- **Homography**: one 3×3 matrix per frame (CSV, row-major `h00…h22`), mapping normalised pitch coordinates (0–1) to
  pixels. Each row also has `reprojection_error` and `valid`.
- **Labels you get**: the public GT for the **first 5 minutes of play** (video frames 11,185–18,684, 7,500 frames). It is
  in the repo as a full-length CSV, with X everywhere else. It is **only for checking the pipeline; not for training, and
  not for tuning** (a rule, not advice).
- Data access: enrol on the MWC landing page and sign the terms, then a download link arrives. Unpack it into `data/`
  (or set `MTAH_DATA_DIR`).

## 4. Submission and rules (SUBMISSION.md, RULES.md, OVERVIEW.md, FAQ.md)

- **CSV**:
  - header exactly `frame,label` (case and surrounding whitespace are tolerated);
  - exactly 199,319 rows, with `frame` = 0…199,318 in order;
  - labels H, A, D or X (they are stripped and upper-cased);
  - no blank cells.
  - The scorer drops a file that fails validation. Name it `<team_name>.csv`.
- **Three deliverables**, through one Google Form (the URL comes by email; you need a Google account; one response per
  team, editable, and the last one before the deadline counts):
  1. the CSV;
  2. a **public code repository** with the source, pinned dependencies, and the command that regenerates the file.
     It must reproduce **the same file, "not approximately"**. The organisers may re-run the winner's repo;
  3. a short technical document (`<team_name>.pdf` or `.md`): the approach, which assets were used and which dropped,
     **all training data with its source, who labelled it and how much**, the possession rules on loose balls,
     challenges and restarts, limitations, and how to reproduce the result.
- **Deadline: 17 November 2026. No time zone is given anywhere in the repo.** Results come before 26 November 2026.
  There is no live leaderboard and no feedback.
- **Teams: 3–5 members.** A team with fewer than 3 is not eligible for the Innovation Challenge or the Final Pitch.
- **Allowed:** any method, language, framework or pre-trained model, and **external data about other games**, including
  hand labels on other games in any amount. This holds provided the organisers can access and inspect it and the
  technical document explains it. A *private in-house model or an unavailable dataset is ruled out*; ask first if
  unsure. Improving or replacing the provided tracking with algorithms is welcome. Watching the video to design features
  is fine.
- **Not allowed:**
  1. any hand labels or hand corrections on the provided game (labels, tracking fixes, or rules and constants tied to
     specific frames of this game), whether for the submission, for training or for validation. Algorithmic pseudo-labels
     and unsupervised learning on it are fine;
  2. the private GT or anything derived from it, and **any human-produced data about this game** (event data,
     corrected tracking, possession data);
  3. a file the repo cannot regenerate.
- **LLMs are allowed anywhere, BUT** you may not upload or disclose to any third-party hosted or cloud LLM any videos,
  datasets, confidential information or other restricted materials, **or any extracts of them** (OVERVIEW.md and FAQ.md,
  quoting the T&Cs).
- **Data licence:** competition only. Do not redistribute it, do not commit it to a public repo, and do not publish
  derived data that would reconstruct it (frame-level tracking exports, clipped video). The repo code itself is MIT.

## 5. What this changes for our plan

1. **LOOSE maps to the last team, not to a gap.** "Contested or loose: possession stays with the last team that had it."
   This is our `had(..., loose="last")`, the default in `barca_labels.scoring`. TRAVELLING maps to the passer's team.
   So our 4 states collapse to 3, and TRAVELLING vs LOOSE no longer matters for the score. The extra PathCRF
   travelling-vs-loose head is not needed for the output.
2. **The GT's notion of control and dead ball is Metrica's corrected tracking, not PFF events.** A touch without control,
   a non-intercepting deflection, or an aerial duel without control must not switch the team. That is stricter than our
   labels v2, where a single controlled touch can become CONTROL. Where the GT puts dead-ball start and end (the whistle?
   the ball leaving the pitch? the restart kick?) can only be seen in the public 5 minutes. **We cannot tune on those 5
   minutes.** We can only check our reading of the definition against them.
3. **X is almost never worth it.** On spellF1 it means "repeat the last label". That is free in the middle of a
   possession and harmful at a hand-over. On macroF1 it is always a miss. A model that always commits does at least as
   well, except where a guess would split a real possession. So the only use is to write X instead of a short spurious
   flip, and the same effect comes from not flipping at all. Use X only on ungraded frames.
4. **Spells are the score.** A 1 s timing error costs about 35 % of a spell, so boundary timing matters. So does the
   number of spells: no flicker, but no merging of real hand-overs either. Restarts (D→H/A) must get the right team.
   The 2nd-half pre-kick-off D stretch (2,934 graded frames) is free macroF1 if handled explicitly.
5. **No player identities, no persistent tracks, 25 fps, 0–1 coordinates, 41 % of the ball missing.** Anything keyed by
   a PFF player id or jersey (C3's owner ranker, PathCRF's 22 fixed player nodes) has to work on per-frame team-labelled
   slots or on re-linked tracklets. The ball-from-players model matters more than it did on PFF.
6. **Training data must be from other games** (PFF is allowed in principle, as data about other games). The organisers
   must be able to access it, and the document must describe who labelled it. PFF World Cup data is public on
   registration; check that its licence allows this use and that the organisers can get it. Our labels v2 are
   algorithmic, derived from PFF's human event data.
7. **Exact reproducibility.** The repo must regenerate the *same file*. The sensible route is to ship trained weights (if
   their licence allows) and make inference deterministic (CPU, fixed seeds). GPU training is not bit-reproducible.
   Weights trained only on PFF are not "about the game", but PFF's terms decide whether they can be published.
8. **The LLM clause constrains how we work.** Hosted assistants, including the one that wrote this file, must not see
   the match data or extracts of it. That covers printed rows, plots, per-frame outputs and the downloaded files. Code
   can be written against the documented formats. Anything run on the data should print only aggregate scores, or be
   inspected by a human.
9. **Time: the deadline is 17 November 2026, two days after this summary was written**, and the data must first be
   obtained through enrolment. Scope accordingly.
