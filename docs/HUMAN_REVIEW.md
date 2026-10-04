# Human review — 20 eligible fit pairs

Stage 3 is pending your ratings. All response fields are intentionally blank. The overall score and human-dependent dimensions remain null until you complete this review.

## How to enter ratings

For guided entry, run `.venv/bin/python -m dating_compat review` in the VS Code
terminal from the project folder. Enter your name/alias, then rate each field:
`1` acceptable, `2` ambiguous, `3` unacceptable. Notes are required for ambiguous
or unacceptable judgments. Type `yes` at each summary to attest that these are
your human judgments and save the pair. `q` pauses; rerun to resume unfinished
pairs. Unsaved answers for the current pair are discarded on quitting. Use
`--pair NUMBER` to revisit one pair; Enter keeps existing ratings and notes.
After entering all responses, run the quality command below.

Edit docs/human_review_responses.csv in VS Code or a spreadsheet. Keep the sample IDs, profile text, labels, metadata, row order, and column names unchanged. You can also send your ratings in chat using the sample numbers; they will be transcribed exactly.

For every pair, fill six fields with exactly `acceptable`, `ambiguous`, or `unacceptable`. Write a short note for any ambiguous or unacceptable judgment. Add your name or alias in `reviewer` and set `human_reviewed` to `yes` on every completed row. Blank or incomplete rows keep the gate pending. Ambiguous judgments are preserved and count zero toward the acceptable fraction.

| Field | What to judge |
|---|---|
| label_agreement | Does the supplied compatible/incompatible label reasonably follow the two profile texts? Different hobbies alone need not imply incompatibility. |
| naturalness | Would a person plausibly say these sentences, with fluent wording? |
| coherence | Are each profile’s statements understandable and internally consistent? Multiple different preferences can coexist. |
| preference_realism | Are the stated preferences and combinations plausible for real people? This does not validate actual romantic matching. |
| hierarchy_consistency | Does the label respect stated priorities: dealbreakers first, then core values, lifestyle, interests, and activities? Mark ambiguous if priorities or conflict are insufficiently specified. |
| clarity | Is there enough information to make the judgment without unresolved ambiguity or contradictory cues? |

Use acceptable for a defensible judgment, ambiguous for insufficient or conflicting evidence, and unacceptable for a clear disagreement or quality problem. Notes should explain the actual uncertainty or conflict. Do not use pair_type as the label.

The assignment’s preference hierarchy gives dealbreakers weight 10, core values 8–10, lifestyle 5–7, interests 3–5, and activities 1–3. These are assignment assumptions, not validated universal relationship rules.

## Sample and scoring provenance

Seed 42; 20 unique normalized pairs selected only from the 4,792 eligible fit rows. Two pairs per each of nine pair types, then two random remaining fit pairs; presentation shuffled. No evaluation, validation, or quarantined IDs are sampled. Counts and hashes are in reports/human_review_sample.json.

This stratified sample overrepresents rare pair types and is diagnostic, not a population estimate. Each human component is 5 times the acceptable fraction of these 20 pairs. All five dimension totals must be at least 12/20 and the overall total at least 60/100. The report is a project heuristic, not proof of fairness or real-world matchmaking.

Source: dipenbhuva/dating-compatibility-pairs at revision 69b750dd024d5732ca19596d86bffc253abee92e, dataset card declares Apache-2.0. Text below is a 20-pair review excerpt; originals are retained in ignored data/.

After completing the CSV, run `.venv/bin/python -m dating_compat quality`. Passing requires `QUALITY GATE PASS` and exit 0. Pending or failed quality returns exit 1; never weaken the rubric or invent ratings to pass.

## Review pairs

### Pair 1

- Profile 1: boy: I'm drawn to coding for fun
- Profile 2: girl: I get excited about karaoke nights
- Supplied label: compatible (1).
- Metadata: subtle_mismatch; lifestyle; tech_savvy_vs_extroverted.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 2

- Profile 1: boy: I would never date someone who is into drama
- Profile 2: girl: I'm known for drama
- Supplied label: incompatible (0).
- Metadata: dealbreaker; dealbreakers; personality.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 3

- Profile 1: girl: I treasure book clubs and I live for exploring new places
- Profile 2: boy: I live for coffee connoisseurship and I value adventure travel
- Supplied label: incompatible (0).
- Metadata: llm_judged_incompatible; interests_and_lifestyle; score_3.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 4

- Profile 1: girl: laziness is where I draw the line
- Profile 2: boy: I'm guilty of laziness
- Supplied label: incompatible (0).
- Metadata: dealbreaker; dealbreakers; lifestyle.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 5

- Profile 1: girl: I'm drawn to big crowds
- Profile 2: boy: I'm obsessed with house parties
- Supplied label: compatible (1).
- Metadata: compatible; lifestyle; extroverted.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 6

- Profile 1: girl: I thrive on watching indie films and I thrive on artistic expression
- Profile 2: boy: I get excited about wine tasting and I really enjoy creative problem solving
- Supplied label: incompatible (0).
- Metadata: llm_judged_incompatible; interests_and_values; score_3.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 7

- Profile 1: girl: I absolutely love artistic expression
- Profile 2: boy: I adore creative problem solving
- Supplied label: compatible (1).
- Metadata: compatible; values; creativity.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 8

- Profile 1: girl: I'm drawn to gardening and I have no patience for sustainability
- Profile 2: boy: I dislike gardening and I genuinely appreciate sustainability
- Supplied label: incompatible (0).
- Metadata: complex_incompatible; interests_and_values; outdoors_vs_environment.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 9

- Profile 1: girl: I live for zero-waste living
- Profile 2: boy: I value starting my own business
- Supplied label: compatible (1).
- Metadata: subtle_mismatch; values; environment_vs_career.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 10

- Profile 1: girl: I thrive on smoothie bowls
- Profile 2: boy: I'm not into intermittent fasting
- Supplied label: incompatible (0).
- Metadata: incompatible; lifestyle; health_conscious.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 11

- Profile 1: girl: I'm deeply into watching animal videos
- Profile 2: boy: I have no patience for talking to my pets like they understand me
- Supplied label: incompatible (0).
- Metadata: incompatible; lifestyle; pet_lover.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 12

- Profile 1: boy: I love going out until 3am every weekend. The nightlife is my life
- Profile 2: girl: I wake up at 5am for yoga. I need a partner with a healthy routine
- Supplied label: incompatible (0).
- Metadata: curated_realistic; realistic_example; fundamentally_different_daily_rhythms_and_lifestyle_priorities.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 13

- Profile 1: boy: I stay away from martial arts however I'm deeply into beach walks
- Profile 2: girl: I'm passionate about martial arts and I can't get enough of beach walks
- Supplied label: compatible (1).
- Metadata: complex_compatible; lifestyle_and_interests; active_vs_outdoors.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 14

- Profile 1: boy: I have no patience for watching animal videos but I'm into gardening
- Profile 2: girl: I love watching animal videos and I really enjoy gardening
- Supplied label: compatible (1).
- Metadata: complex_compatible; lifestyle_and_interests; pet_lover_vs_outdoors.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 15

- Profile 1: boy: I treasure strategy games and I live for trying new cuisines but I am completely against party lifestyle
- Profile 2: girl: I genuinely appreciate philosophy books but I am crazy about sushi making and I would never date someone into controlling behavior
- Supplied label: compatible (1).
- Metadata: llm_judged_compatible; dealbreakers_and_interests; score_6.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 16

- Profile 1: boy: I thrive on picnics in the park and my pets are my world
- Profile 2: girl: I can't stand team sports leagues and I really enjoy animals
- Supplied label: compatible (1).
- Metadata: llm_judged_compatible; interests_and_lifestyle; score_5.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 17

- Profile 1: boy: I thrive on networking events
- Profile 2: girl: I find joy in casual walks
- Supplied label: compatible (1).
- Metadata: subtle_mismatch; lifestyle; extroverted_vs_relaxed.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 18

- Profile 1: girl: I love binge-watching tech reviews however I avoid watercolor painting
- Profile 2: boy: I can't deal with binge-watching tech reviews and I treasure watercolor painting
- Supplied label: incompatible (0).
- Metadata: complex_incompatible; lifestyle_and_interests; tech_savvy_vs_arts.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 19

- Profile 1: boy: I love quiet weekends and reading but I value deep friendships
- Profile 2: girl: I enjoy social gatherings and parties but I cherish meaningful relationships
- Supplied label: compatible (1).
- Metadata: curated_realistic; realistic_example; different_social_styles_but_shared_value_of_meaningful_relationships.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___

### Pair 20

- Profile 1: girl: I'm passionate about grilling
- Profile 2: boy: I avoid grilling
- Supplied label: incompatible (0).
- Metadata: incompatible; interests; food.
- Ratings: label agreement ___; naturalness ___; coherence ___; preference realism ___; hierarchy consistency ___; clarity ___.
- Notes: ___
