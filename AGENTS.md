# Workspace instructions for agents

This workspace contains the Amazon ML Challenge 2026 business entity-resolution project.

## Required project memory workflow

1. Read `PROJECT_MEMORY.md` completely before analyzing data, proposing a method, modifying code, or answering project questions.
2. Treat `PROJECT_MEMORY.md` as durable project context. Do not replace it with a short summary.
3. After any material analysis, experiment, implementation, validation run, modeling decision, discovered constraint, or corrected assumption, update `PROJECT_MEMORY.md` before ending the task.
4. Preserve previously verified details. If a result is superseded, retain the old result, mark it as superseded, and add the new result with the date, method, and reason.
5. Clearly distinguish:
   - exact full-dataset measurements;
   - deterministic-sample measurements;
   - estimates or inferences;
   - proposed ideas that have not yet been validated.
6. Record enough methodological detail for another agent to reproduce each statistic, including source files, selection rules, normalization definitions, denominators, and caveats.
7. Do not store temporary credentials, secrets, or private environment information in project memory.

## Challenge-specific rules

- Read TSV files with an explicit tab delimiter. Commas occur inside addresses and match-ID lists.
- Do not use external business databases, identity-resolution services, registration lookups, geocoding APIs, or internet-derived business identity data. The challenge explicitly prohibits external data lookup.
- Do not assume the set of countries is fixed to US and India. France occurs in the test set, and future labels must be treated as open-set strings.
- Source 1 is the deduplicated anchor/reference source. Source 2 and Source 3 contain candidate target records.
- A Source 1 anchor can have zero, one, or multiple matches in Source 2 and Source 3.
- The leaderboard metric is macro entity-level F-beta with beta 0.5. Optimize and validate the final grouped prediction, not only pairwise classification metrics.
- `candidate_pairs.tsv` must contain the exact final candidate set passed to the matching model at inference, after earlier blocking and cheap filtering stages.
- Every final match must be present in `candidate_pairs.tsv`.
- Before submission, run `utils/validate_submission.py` and preserve the result and experiment version.

## File preservation

- Preserve the supplied PDFs, README, documentation template, validator, and raw train/test TSV files.
- Put reusable implementation under the challenge submission structure described in `README.md` when implementation begins.
- Avoid committing temporary EDA scripts or rendered PDF pages unless the user explicitly wants them retained. Durable findings belong in `PROJECT_MEMORY.md`.

