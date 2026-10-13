# Room benchmark

One furnished room, four Stray Scanner captures on an iPhone 14 Pro Max, nine staged changes and a set of
look-alike objects that never moved. The captures are on Hugging Face ([amitabh3/revisit-room-captures](https://huggingface.co/datasets/amitabh3/revisit-room-captures),
fetched by `python scripts/download_data.py --room`) and on YouTube for a quick look:
[Take 1](https://youtu.be/VDI5xJ1UHZw), [Take 2](https://youtu.be/iZ1sGiMjBXQ),
[Take 3](https://youtu.be/vNPY1HJV00I), [Take 4](https://youtu.be/2T7ERQ7c4KE).

Directions below follow a sketch of the room with door A at the top, door B at the bottom, and the left and
right walls as drawn. Distances and angles in the staging plan were never measured, so none are given here
and none are scored.

## Captures

| Take | Room state | Route | Length | Role |
|:--|:--|:--|--:|:--|
| 1 | before | from door A, clockwise at chest height, then a waist-height lap around the rug | 110 s | first scan |
| 2 | before, untouched | from door B, counter-clockwise | 85 s | no-change control (different start and direction) |
| 3 | after | from door A; the bottom-left corner is kept off screen | 104 s | second scan |
| 4 | after | a second walk around the room (about 35 m) that also films the bottom-left corner | 99 s | re-look at the corner Take 3 skipped |

All takes: 30 fps, phone held sideways with the same side up, room lights on, curtains closed, screens off, and a 2 s
hold on a static object at the start and end.

## The nine changes (applied between Take 2 and Take 3)

| # | Expected result | Before | After |
|--:|:--|:--|:--|
| 1 | moved | chair on the top side of the bottom-right table | up the right-hand walkway toward the top-right table, turned to face the rug |
| 2 | added | floor between door A and the back of the sofa is clear | a tall stack of books there |
| 3 | removed | top-right table with two identical chairs | the chair nearer the room centre is gone |
| 4 | moved (one of twins) | two identical chairs on the rug facing the sofa | left one untouched; right one pulled back toward door B and turned |
| 5 | removed (flat, under 1 cm) | flat item on the rug, left of the left rug chair | gone |
| 6 | moved to another table | ceramic cup on the left end of the coffee table | on the bottom-right table |
| 7 | UNVERIFIED with Take 3, removed once Take 4 is added | plant pot in the bottom-left corner | gone; the corner is only filmed in Take 4 |
| 8 | replaced | black square pillow, left end of the sofa | multicoloured square pillow in the same spot |
| 9 | swapped | dark brown hat on the top-right table, stack of 3 books on the bottom-right table | the two traded places |

## Controls that never moved

- A light greyish-brown decoy hat on the top-left table, close in colour to the dark brown hat of change 9.
- Identical sets: 6 corner chairs (3 per corner table), the remaining top-right chair, the untouched rug chair,
  and the second black square pillow.
- Sofa pillows, a blue can on the coffee table, the sofa, coffee table, rug, six wall paintings and both doors.

## Comparisons

| Comparison | Captures | Ideal output |
|:--|:--|:--|
| no-change pair | Take 1 vs Take 2 | nothing |
| before/after | Take 1 vs Take 3 | changes 1-6, 8, 9; change 7 UNVERIFIED with a look-from pose toward the corner |
| with re-look | Take 1 vs Take 3, re-look Take 4 | change 7 becomes REMOVED |

## Results (final configuration)

Every report was checked by hand against the staged changes. **Right**: an active verdict names the right
object with the right type. **Partly right**: the right object with the wrong type, the wrong destination or
only half of a swap. **False reports**: active verdicts that match no staged change. Distances were never
measured, so displacements are not judged.

| Verifier | No-change pair: false changes | Before/after: right / partly / UNVERIFIED / missed / false | With re-look: right / partly / missed / false |
|:--|--:|:--|:--|
| Qwen3-VL-8B + GDPO+RLCR | 11 of 17 | 2 / 4 / 1 / 2 / 3 | 4 / 3 / 2 / 6 |
| SenseNova-SI-8B + GDPO+RLCR | 10 of 17 | 3 / 4 / 1 / 1 / 3 | 5 / 3 / 1 / 7 |

### Per-change audit

Verdict and p(change) of the proposal that matches each change, from `results/room/<verifier>/`.

| # | Qwen, before/after | Qwen, with re-look | SenseNova, before/after | SenseNova, with re-look |
|--:|:--|:--|:--|:--|
| 1 | partly: MOVED 6.8 m to a look-alike chair across the room (.91) | partly: MOVED 6.6 m, same (.89) | partly: same as Qwen (.91) | partly: same as Qwen (.89) |
| 2 | right: ADDED stack of books (.79) | right (.81) | right (.79) | right (.80) |
| 3 | missed: rejected (.34) | missed: rejected (.31) | right: REMOVED chair (.51) | right (.87) |
| 4 | right: MOVED chair 0.83 m (.97) | right: 0.80 m (.97) | right (.93) | right (.93) |
| 5 | UNVERIFIED, Take 3 saw the spot in 1 of 28 keyframes | right: REMOVED, labelled book (.70) | UNVERIFIED | right (.53) |
| 6 | partly: ADDED cup at the new table (.62) | partly (.60) | partly (.43) | partly (.42) |
| 7 | missed: no proposal (corner never filmed) | right: REMOVED potted plant (.73) | missed: no proposal | right (.73) |
| 8 | partly: ADDED pillow (.67) | missed: no proposal | partly (.51) | missed: no proposal |
| 9 | partly: books REMOVED (.71), no report names the hat | partly (.64) | partly (.54) | partly (.63) |

The three UNVERIFIED proposals of the before/after pair are the flat item of change 5 and two reflections of
posters in a glossy table. The plant of change 7 never gets a proposal before the re-look: SceneDiff costs exist
only where the second scan looked, and the ledger only inspects proposals. The 6.6 to 6.8 m moves link the
top-left and bottom-right corner tables, which carry identical sets of three chairs; the no-change pair has a
match of the same kind.

### Keyword score

`scripts/score_room.py` scores runs without manual judgment, and `reproduce/room.sh` uses it, together with a
check that every verdict matches the reference reports, so the audit above also holds for a reproduced run.

- **False changes** on the no-change pair: every moved/added/removed/replaced/swapped verdict.
- **Found** on the other two pairs: a change counts when an active verdict's label matches its keywords
  (for example `chair`, `book`, `cup`, `plant|pot|vase`). Change 7 counts as found in the before/after pair
  when any proposal is UNVERIFIED.
- **Static false flags**: active verdicts whose label names something that never changed (door, painting,
  sofa, rug, table, lamp, wall, floor).

| Verifier | No-change pair: false changes / proposals | Before/after: found / static false flags | With re-look: found / static false flags |
|:--|--:|--:|--:|
| Qwen3-VL-8B + GDPO+RLCR | 11 / 17 | 7 / 1 | 6 / 2 |
| SenseNova-SI-8B + GDPO+RLCR | 10 / 17 | 8 / 1 | 7 / 3 |

The keyword score over-credits. In the before/after pair it counts change 7 for the UNVERIFIED reflections and
flat item, and change 5 for the removed books of the swap, which also carry the label `book`; it counts a
partly right change (cup, pillow, the chair of change 1) as found. Use it to check a rerun, not to measure
recall. Per-change hits are in `results/room/scoreboard.json` (its `hidden_plant` field follows the same rule),
and the mask ablation in [reproducing.md](reproducing.md#mask-backend-and-keyframes) uses the same score.

The keyframe count (28 per scan) and the mask backend were chosen on these same captures; there is no
held-out room.
