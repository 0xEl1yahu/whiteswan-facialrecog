# M2 Gallery Baseline — 2026-09-26

This file preserves the M2 gallery results before later additions or tuning. Future runs
should be recorded separately and compared with this baseline rather than replacing it.

## Decision

- D1 pin strategy: **`all`**, selected by Eli on 2026-09-26.
- Reason: retain both curated examples per character as separate pins instead of averaging
  away their variation. The embedding cache remains strategy-independent.
- Threshold and normalization remain provisional at `0.30` and `base`; they are not M2
  decisions.

## Gallery

| Character | Found | Used | Skipped |
|---|---:|---:|---:|
| Harry Potter | 2 | 2 | 0 |
| Hermione Granger | 2 | 2 | 0 |
| Prof. McGonagall | 2 | 2 | 0 |
| Prof. Severus Snape | 2 | 2 | 0 |
| Ron Weasley | 2 | 2 | 0 |

Total: 10 valid photos, 10 per-photo embeddings, 10 `all` pins or 5 derived `mean` pins.

## Cache timings

| Run | Time | Model work |
|---|---:|---|
| Cold real-gallery build | 27.520 s | 10 photos embedded |
| Warm real-gallery replay | 0.007 s | Model access disabled; no inference |
| Add one photo in an isolated gallery copy | 5.331 s | Exactly 1 model load and 1 embedding |
| Delete that photo | 0.007 s | No model access; only its record removed |

## Leave-one-out results

Each photo was matched against a gallery rebuilt without that photo. Results are at the
provisional cosine threshold `0.30` with `base` normalization. With only two photos per
character, `mean` and `all` produced the same displayed results.

| Held-out photo | Expected | Nearest | Distance | Assigned |
|---|---|---|---:|---|
| `daniel-radcliffe_actor_age-12_2001-premiere.jpg` | Harry Potter | Harry Potter | 0.372545 | Unknown |
| `daniel-radcliffe_age-12_philosophers-stone.jpg` | Harry Potter | Harry Potter | 0.372545 | Unknown |
| `emma-watson_actor_age-11_2001-premiere.jpg` | Hermione Granger | Hermione Granger | 0.238956 | Hermione Granger |
| `emma-watson_age-11_philosophers-stone.jpg` | Hermione Granger | Hermione Granger | 0.238956 | Hermione Granger |
| `MV5BMTc0NTM0NzIyMF5BMl5BanBnXkFtZTYwOTg0NTg3._V1_.jpg` | Prof. McGonagall | Prof. McGonagall | 0.345453 | Unknown |
| `maggie-smith_age-66_philosophers-stone.jpg` | Prof. McGonagall | Prof. McGonagall | 0.345453 | Unknown |
| `alan-rickman_actor_age-55_2001-portrait.jpg` | Prof. Severus Snape | Prof. Severus Snape | 0.395217 | Unknown |
| `alan-rickman_age-55_philosophers-stone.jpg` | Prof. Severus Snape | Prof. Severus Snape | 0.395217 | Unknown |
| `rupert-grint_actor_age-13_2001-premiere.jpg` | Ron Weasley | Ron Weasley | 0.201827 | Ron Weasley |
| `rupert-grint_age-13_philosophers-stone.jpg` | Ron Weasley | Ron Weasley | 0.201827 | Ron Weasley |

Aggregate for both strategies: nearest identity **10/10 correct**; assignments **4 correct,
6 Unknown, 0 wrong**.

## Comparison rule

When owner-curated photos are added, rerun the same leave-one-out evaluation and save a new
dated result. Compare nearest-identity accuracy, correct/Unknown/wrong assignments, and
per-character distances against this file. Do not overwrite this baseline.
