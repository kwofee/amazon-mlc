# Amazon ML Challenge 2026 project memory

Last updated: 2026-09-25, Asia/Kolkata.

This file is durable context for future agents and chats. Read it completely before working on this project. Update it after new analysis, experiments, implementation decisions, validation results, or corrections. Do not reduce it to a short summary.

## 1. Project and challenge

The task is business entity resolution across three independent record sources.

- Source 1 (`S1`) is the deduplicated reference source.
- Source 2 (`S2`) and Source 3 (`S3`) contain noisy records that may or may not refer to an S1 business.
- For every S1 record, the system must output all matching S2 and S3 record IDs.
- One S1 record may have zero, one, or multiple matching target records.
- Training data contains US and India.
- Test data contains US, India, and an unseen country, France.
- Country labels must be handled as open-set strings. Do not build a pipeline that only accepts US and India.
- External business identity lookup, registration lookup, geocoding, or external-data augmentation is prohibited.
- A final model must comply with the challenge's MIT/Apache 2.0 licensing and parameter-limit rules.

Primary supplied files:

- `6ab5628d5a817_amazon_ml_challenge_problem_statement.pdf`
- `6ab56657b4f1a_guidelines_and_key_instructions_amazon_ml_challenge_2026.pdf`
- `README.md`
- `Documentation_template.md`
- `utils/validate_submission.py`
- `dataset/train/train_source1.tsv`
- `dataset/train/train_source2.tsv`
- `dataset/train/train_source3.tsv`
- `dataset/train/train_ground_truth.tsv`
- `dataset/test/test_source1.tsv`
- `dataset/test/test_source2.tsv`
- `dataset/test/test_source3.tsv`

All data and submission files are tab-separated. Always specify the tab delimiter. A normal comma-separated read will silently parse the data incorrectly because addresses and ID lists contain commas.

## 2. Terminology

### 2.1 Anchor

An anchor is one row from Source 1.

It is called an anchor because it is the reference record around which matching is performed. The question for each anchor is: which S2 and S3 records describe this same real-world business?

An anchor is not a positive pair, class label, or cluster count. It is a business record from the deduplicated reference source.

Example:

```text
S1 anchor:
S1-00001    Acme Technologies Pvt Ltd    12 MG Road, Bengaluru    India
```

The required prediction for this anchor may contain no records, one record, or several records.

### 2.2 Target record

A target record is a row from Source 2 or Source 3 that could potentially be linked to an S1 anchor.

Example target pool:

```text
S2-10001    Acme Technologies Private Limited    12 M G Rd, Bangalore    India
S2-10002    Acme Tech Pvt Ltd                     Bengaluru, MG Road 12    India
S3-90001    एक्मे टेक्नोलॉजीज प्राइवेट लिमिटेड    12 MG Road, Karnataka   India
S3-90002    Different Business                    ...                      India
```

The first three could be true targets for `S1-00001`; the fourth is a negative/decoy target.

Training targets are not all positive. Approximately one quarter of the training target records are not linked to any S1 anchor in the supplied ground truth.

### 2.3 Anchor-target pair

An anchor-target pair is one proposed or labeled comparison, such as:

```text
(S1-00001, S2-10001)
```

One anchor can produce multiple true pairs. Therefore, the number of true pairs is larger than the number of anchors.

### 2.4 Singleton

A singleton is an S1 anchor whose ground-truth `matched_entity_ids` list is empty.

Under the challenge's macro F0.5 scoring, a singleton scores 1.0 only when the system predicts an empty list. Any predicted match for a true singleton gives that anchor a score of 0.0.

## 3. Exact dataset sizes

These are exact row counts excluding the header row.

### 3.1 Training

| File | Role | Rows |
| --- | --- | ---: |
| `train_source1.tsv` | Training anchors | 2,206,821 |
| `train_source2.tsv` | S2 training targets | 5,034,616 |
| `train_source3.tsv` | S3 training targets | 5,285,603 |
| `train_ground_truth.tsv` | One label row per S1 anchor | 2,206,821 |

Total training target records:

```text
5,034,616 + 5,285,603 = 10,320,219
```

Therefore, the phrase “2,206,821 training anchors and 10,320,219 training target records” means:

- there are 2,206,821 reference businesses to resolve in training;
- there are 10,320,219 S2/S3 rows that form the target search pool;
- the system must learn which target rows belong to each reference anchor;
- the comparison space cannot be evaluated exhaustively.

### 3.2 Test

| File | Role | Rows |
| --- | --- | ---: |
| `test_source1.tsv` | Test anchors requiring predictions | 1,732,544 |
| `test_source2.tsv` | S2 test targets | 4,887,273 |
| `test_source3.tsv` | S3 test targets | 5,082,316 |

Total test target records:

```text
4,887,273 + 5,082,316 = 9,969,589
```

Every one of the 1,732,544 test anchors must have exactly one output row, including France anchors and anchors predicted to have no matches.

## 4. Exact training-label structure

These measurements were computed across the complete `train_ground_truth.tsv` file.

### 4.1 Number of true targets per anchor

| True matches for an anchor | Number of anchors |
| ---: | ---: |
| 0 | 123,247 |
| 1 | 119,157 |
| 2 | 375,212 |
| 3 | 530,841 |
| 4 | 484,115 |
| 5 | 321,957 |
| 6 | 164,868 |
| 7 | 63,968 |
| 8 | 18,680 |
| 9 | 4,205 |
| 10 | 534 |
| 11 | 37 |

Derived exact values:

- Singleton anchors: 123,247 / 2,206,821 = 5.5848%.
- Total positive anchor-target pairs: 7,638,365.
- Average true target links per anchor: 7,638,365 / 2,206,821 = 3.4613.
- Maximum observed total match count: 11.

### 4.2 Which target sources appear for each anchor

| Source composition | Anchors |
| --- | ---: |
| No match | 123,247 |
| S2 only | 143,029 |
| S3 only | 164,498 |
| Both S2 and S3 | 1,776,047 |

`1,776,047 / 2,206,821 = 80.4799%` of anchors have at least one match in both target sources.

### 4.3 Full S2/S3 match-count matrix

Rows are the number of S2 matches. Columns are the number of S3 matches.

| S2 matches \ S3 matches | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 123,247 | 60,407 | 57,041 | 31,526 | 12,260 | 3,017 | 247 |
| 1 | 58,750 | 269,681 | 251,224 | 140,393 | 54,575 | 13,437 | 1,048 |
| 2 | 48,490 | 223,054 | 208,159 | 116,076 | 45,167 | 10,957 | 876 |
| 3 | 25,037 | 114,405 | 105,817 | 59,427 | 23,315 | 5,504 | 452 |
| 4 | 8,898 | 40,618 | 38,338 | 20,852 | 8,131 | 2,085 | 156 |
| 5 | 1,854 | 8,252 | 7,796 | 4,169 | 1,668 | 378 | 37 |

The training maximum is five S2 records and six S3 records for one anchor. Treat these as soft empirical priors, not guaranteed test limits.

### 4.4 Positive targets and global exclusivity

- Positive S2 target records: 3,693,619.
- Positive S3 target records: 3,944,746.
- Total positive target IDs: 7,638,365.
- Unique positive target IDs: 7,638,365.
- Duplicate assignments of a target ID to multiple S1 anchors: 0.

Thus, every labeled positive target belongs to exactly one S1 anchor in training.

This suggests a useful inference constraint: an S2 or S3 record should normally be assigned to no more than one S1 anchor. S1 anchors remain one-to-many because each anchor may own several target records.

Training target coverage:

- 3,693,619 / 5,034,616 = 73.3645% of S2 records are positive targets.
- 3,944,746 / 5,285,603 = 74.6319% of S3 records are positive targets.
- Roughly one quarter of S2/S3 training records are unmatched decoys.

## 5. Exact country distributions

### 5.1 Training

| Source | US | India | Total |
| --- | ---: | ---: | ---: |
| S1 | 1,323,633 | 883,188 | 2,206,821 |
| S2 | 3,016,817 | 2,017,799 | 5,034,616 |
| S3 | 3,170,056 | 2,115,547 | 5,285,603 |

Training S1 is 59.979% US and 40.021% India.

### 5.2 Test

| Source | US | India | France | Total |
| --- | ---: | ---: | ---: | ---: |
| S1 | 663,106 | 809,986 | 259,452 | 1,732,544 |
| S2 | 1,871,330 | 2,312,565 | 703,378 | 4,887,273 |
| S3 | 1,945,701 | 2,405,000 | 731,615 | 5,082,316 |

Test S1 shares:

- US: 38.274%.
- India: 46.751%.
- France: 14.975%.

France contains 259,452 anchors and cannot be ignored or treated as a minor edge case.

## 6. Exact missing-field measurements

No business names are missing in any train or test source file.

S1 names and addresses are complete in both training and test.

Target address missingness:

| File | Missing US addresses | Missing India addresses | Missing France addresses | Total missing | Percent of file |
| --- | ---: | ---: | ---: | ---: | ---: |
| `train_source2.tsv` | 111,121 | 57,846 | n/a | 168,967 | 3.3561% |
| `train_source3.tsv` | 110,968 | 64,948 | n/a | 175,916 | 3.3282% |
| `test_source2.tsv` | 55,107 | 52,764 | 21,537 | 129,408 | 2.6479% |
| `test_source3.tsv` | 55,317 | 59,240 | 21,541 | 136,098 | 2.6779% |

In the labeled-pair sample described below, 4.349% of positive target records had an empty address. Empty target addresses must not be treated as an automatic negative.

Target address strings also sometimes contain the literal text `null` rather than being structurally empty. Treat `null` as a missing/noise token during feature extraction.

## 7. Deterministic positive-pair sample

### 7.1 Sampling method

This was not a random `pandas.sample` call.

For every training S1 ID, the numeric suffix after `S1-` was parsed. The anchor was included when:

```text
numeric_suffix % 50 == 0
```

Because IDs appear randomly distributed, this selects approximately 2% of anchors in a deterministic and reproducible way.

All ground-truth S2/S3 matches for each selected anchor were then retrieved from the full target files.

### 7.2 Sample size

- Sampled anchors: 44,286.
- Sampled US anchors: 26,422.
- Sampled India anchors: 17,864.
- Sampled singleton anchors: 2,393.
  - US singletons: 1,414, or 5.3516% of sampled US anchors.
  - India singletons: 979, or 5.4803% of sampled India anchors.
- Sampled anchors with one or more matches: 41,893.
- True positive anchor-target pairs from the selected anchors: 153,667.
- Average true pairs per sampled anchor: 153,667 / 44,286 = 3.4699.
- Average true pairs per sampled non-singleton anchor: 153,667 / 41,893 = 3.6681.

The phrase “a deterministic sample of 153,667 true pairs” refers to 153,667 links, not 153,667 anchors.

Example:

```text
Anchor S1-A -> S2-X, S2-Y, S3-Z
```

This is one sampled anchor but three true pairs:

```text
(S1-A, S2-X)
(S1-A, S2-Y)
(S1-A, S3-Z)
```

Pair breakdown:

- US positive pairs: 91,707.
- India positive pairs: 61,960.
- S2 positive pairs: 74,374.
- S3 positive pairs: 79,293.

All sample percentages in Sections 8 through 11 use these denominators unless stated otherwise. They are sample estimates, not exact full-dataset measurements.

## 8. Exact normalization used in the sampled-pair EDA

The 21.5% exact-normalized-name result came from a deliberately basic normalization. It did not use transliteration, stemming, edit-distance correction, semantic embeddings, or a learned model.

### 8.1 Basic tokenization and normalization

For a name or address string:

1. Apply Unicode NFKC normalization.
2. Apply Unicode `casefold()`.
3. Replace `&` with the word `and` surrounded by spaces.
4. Extract Unicode alphanumeric word tokens using the equivalent of Python regular expression `[^\W_]+`.
5. Join extracted tokens with one ordinary space.

Equivalent pseudocode:

```python
def tokens(text):
    text = unicodedata.normalize("NFKC", text or "")
    text = text.casefold().replace("&", " and ")
    return re.findall(r"[^\W_]+", text)

def normalized(text):
    return " ".join(tokens(text))
```

Examples:

```text
"ACME, INC."              -> "acme inc"
"Acme Inc"                -> "acme inc"
"  Acme---Inc  "          -> "acme inc"
"A&B Technologies Pvt Ltd" -> "a and b technologies pvt ltd"
```

Two names counted as an exact normalized-name match only when both normalized strings were nonempty and identical in the same token order.

Therefore:

```text
"ACME, INC." vs "acme inc"                 -> exact normalized match
"Acme Inc" vs "Acme Incorporated"          -> not an exact normalized match
"Global Tech Ltd" vs "Tech Global Limited" -> not an exact normalized match
"Buex" vs "#búex"                          -> not an exact normalized match
```

Basic normalization did not:

- equate `inc` with `incorporated`;
- equate `pvt` with `private`;
- equate `ltd` with `limited`;
- reorder tokens;
- remove accents;
- stem singular/plural forms;
- correct typos or digit substitutions;
- transliterate Hindi or other Indic scripts into Latin.

### 8.2 Legal/core-name normalization

A second representation removed low-information/legal tokens from the normalized name.

The EDA removal set was:

```text
inc, incorporated, corp, corporation, co, company,
llc, llp, ltd, limited, plc, pvt, private,
sarl, sas, sa, gmbh,
group, holdings, enterprise, enterprises,
the, and, of, dba
```

Examples:

```text
"Acme Technologies Pvt Ltd"        -> "acme technologies"
"Acme Technologies Private Limited" -> "acme technologies"
```

These two are not equal under the basic normalized-name definition, but they are equal under the core-name definition.

This raised positive-pair exact equality from 21.517% to 43.439%.

The list was a heuristic for analysis. Words such as `group`, `holdings`, or `enterprise` can sometimes be identity-bearing. A production model should preserve both the full and core representations rather than permanently deleting information.

### 8.3 Order-insensitive name representation

Core tokens were sorted before joining.

Example:

```text
"Global Tech Limited" -> core tokens [global, tech] -> "global tech"
"Tech Global Ltd"     -> core tokens [tech, global] -> "global tech"
```

45.535% of sampled true pairs had the same order-insensitive core-name token representation.

### 8.4 Address canonicalization used in EDA

Basic address normalization used NFKC, casefolding, punctuation removal, token extraction, and space joining.

A second canonical-address representation additionally mapped:

```text
rd -> road
st -> street
ave -> avenue
ln -> lane
dr -> drive
blvd -> boulevard
hwy -> highway
ctr -> center
ct -> court
apt -> apartment
ste -> suite
fl -> floor
bldg -> building
dist -> district
opp -> opposite
nr -> near
r -> rue
```

For order-insensitive address-token comparisons, these low-information address tokens were removed:

```text
road, rd, street, st, avenue, ave, lane, ln, drive, dr,
boulevard, blvd, highway, hwy, route, rue, de, la, le, du,
near, opposite, opp, behind, at, in, and, the, no, number,
district, dist, state, city, county, nagar, colony, phase, sector
```

The production system should keep full, canonical, numeric, and token-set views. Do not rely on only one destructively normalized address.

## 9. Positive-pair similarity findings

Overall sampled positive-pair results, denominator 153,667:

| Feature | Positive pairs satisfying feature |
| --- | ---: |
| Same country label | 100.000% |
| Exact basic normalized name | 21.517% |
| Exact core name after legal-token removal | 43.439% |
| Same order-insensitive core-name tokens | 45.535% |
| Exact basic normalized address | 8.252% |
| Exact canonical address | 11.543% |
| Same order-insensitive canonical address tokens | 16.777% |
| Exact normalized name or address | 28.492% |
| Shared core-name token | 84.958% |
| Shared useful address token | 95.624% |
| Shared core-name or useful address token | 99.993% |
| Shared address digit token | 79.727% |
| Shared address digit token of length at least four | 28.247% |
| Cross-name acronym equality heuristic | 0.161% |
| Name-script bucket differs | 7.386% |
| Target address is empty | 4.349% |

“Shared core-name token” means exact token equality after removing the legal/low-information token set in Section 8.2.

For example:

```text
Anchor: "Sharma Technologies Pvt Ltd"
Target: "Sharma Technologies Private Limited"
```

Basic tokens:

```text
[sharma, technologies, pvt, ltd]
[sharma, technologies, private, limited]
```

Core tokens:

```text
[sharma, technologies]
[sharma, technologies]
```

They share the non-legal tokens `sharma` and `technologies`.

Another example:

```text
Anchor: "Amazon Web Services Inc"
Target: "Amazon Services LLC"
```

After removing `inc` and `llc`, the exact shared tokens are `amazon` and `services`.

The 84.958% figure did not use stemming or fuzzy matching. `technology` and `technologies` are different tokens under this measurement.

## 10. Country-specific positive-pair findings

### 10.1 US, 91,707 positive pairs

| Feature | Percent |
| --- | ---: |
| Exact normalized name | 25.531% |
| Exact core name | 45.782% |
| Same core-name token set | 48.755% |
| Exact normalized address | 8.749% |
| Exact canonical address | 14.263% |
| Same canonical address-token set | 20.882% |
| Exact name or address | 32.657% |
| Shared core-name token | 91.495% |
| Shared useful address token | 95.286% |
| Shared name or address token | 99.990% |
| Shared digit token | 78.027% |
| Shared long digit token | 42.687% |
| Name-script mismatch | 0.021% |
| Empty target address | 4.677% |

The 0.021% script-mismatch rate corresponds to approximately 19 of the 91,707 sampled US positive pairs.

### 10.2 India, 61,960 positive pairs

| Feature | Percent |
| --- | ---: |
| Exact normalized name | 15.575% |
| Exact core name | 39.973% |
| Same core-name token set | 40.768% |
| Exact normalized address | 7.518% |
| Exact canonical address | 7.518% |
| Same canonical address-token set | 10.702% |
| Exact name or address | 22.327% |
| Shared core-name token | 75.282% |
| Shared useful address token | 96.125% |
| Shared name or address token | 99.997% |
| Shared digit token | 82.243% |
| Shared long digit token | 6.874% |
| Name-script mismatch | 18.288% |
| Empty target address | 3.864% |

The 18.288% script-mismatch rate corresponds to approximately 11,332 of the 61,960 sampled India positive pairs.

### 10.3 Meaning of “changed writing system”

This measurement concerns the Unicode writing system used in the business name, not a change in the underlying business or necessarily a change in spoken language.

The EDA classified each name as containing:

- Latin characters;
- Devanagari characters;
- other alphabetic scripts;
- or a mixture.

Latin detection used Unicode ranges U+0041–U+024F and U+1E00–U+1EFF. Devanagari used U+0900–U+097F. Other alphabetic characters were placed in an `other` bucket, which can include Gujarati, Tamil, Bengali, Kannada, Malayalam, Telugu, and other scripts.

A pair counted as a script mismatch when the anchor and target script-bucket labels differed.

Illustrative true pair from the sample:

```text
S1 name: Jain Industries Private Limited
S3 name: जैन इंडस्ट्रीज प्राइवेट लिमिटेड
```

The first name is written with Latin/Roman letters. The second is written in Devanagari. They represent the same business name, but a basic Latin character n-gram matcher sees almost no name-character overlap.

The associated addresses were:

```text
S1: 09, Bamora Gambhiriya, Sagar, Madhya Pradesh
S3: 09, Jabalpur Region, null, मध्य प्रदेश
```

The shared number `09` and the state appearing once in Latin and once in Devanagari are useful signals even though the name scripts differ.

Other true examples from the sample:

```text
Gold Southern Products Private Limited
ગોલ્ડ સધર્ન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ
```

The target uses Gujarati script.

```text
Raj Products Limited
ராஜ் புராடக்ட்ஸ் லிமிடெட்
```

The target uses Tamil script.

```text
First Infotech Private Limited
ফার্স্ট ইনফোটেক প্রাইভেট লিমিটেড
```

The target uses Bengali script.

This is why India requires original-script features plus transliteration or a multilingual/character-level fallback. Removing punctuation and lowercasing alone cannot align these names.

The US is mostly Latin-to-Latin. Accented text such as `Buex` versus `#búex` remains in the Latin bucket and is a spelling/diacritic problem, not a writing-system mismatch under this measurement.

### 10.4 Caveats for the script statistic

- It is a deterministic-sample statistic, not a full-data exact count.
- The script detector is intentionally simple.
- A mixed-script name and a single-script name count as different bucket labels even when they share some characters.
- The statistic does not distinguish transliteration from translation.
- It measures the presence of characters by Unicode range; it does not prove linguistic equivalence.

## 11. Target-source differences

### 11.1 S2, 74,374 sampled positive pairs

| Feature | Percent |
| --- | ---: |
| Exact normalized name | 21.163% |
| Exact core name | 44.244% |
| Same core-name token set | 46.350% |
| Exact normalized address | 12.448% |
| Exact canonical address | 19.247% |
| Same canonical address-token set | 28.030% |
| Shared core-name token | 83.173% |
| Shared useful address token | 95.648% |
| Name-script mismatch | 9.522% |
| Empty target address | 4.348% |

### 11.2 S3, 79,293 sampled positive pairs

| Feature | Percent |
| --- | ---: |
| Exact normalized name | 21.848% |
| Exact core name | 42.685% |
| Same core-name token set | 44.771% |
| Exact normalized address | 4.317% |
| Exact canonical address | 4.317% |
| Same canonical address-token set | 6.222% |
| Shared core-name token | 86.632% |
| Shared useful address token | 95.602% |
| Name-script mismatch | 5.383% |
| Empty target address | 4.350% |

S2 and S3 use measurably different corruption patterns. Address equality is much more common in S2. Use a source indicator, source-specific calibration, or separate S2/S3 classifiers.

## 12. Positive similarity distributions

Overall sampled positive-pair quantiles:

### 12.1 Name character-trigram Dice

| Quantile | Score |
| ---: | ---: |
| Minimum | 0.000 |
| 1% | 0.000 |
| 5% | 0.000 |
| 10% | 0.286 |
| 25% | 0.653 |
| 50% | 0.824 |
| 75% | 0.943 |
| 90% | 1.000 |

### 12.2 Address character-trigram Dice

| Quantile | Score |
| ---: | ---: |
| Minimum | 0.000 |
| 1% | 0.000 |
| 5% | 0.182 |
| 10% | 0.423 |
| 25% | 0.679 |
| 50% | 0.816 |
| 75% | 0.906 |
| 90% | 1.000 |

### 12.3 Core-name token Jaccard

| Quantile | Score |
| ---: | ---: |
| Minimum | 0.000 |
| 1% | 0.000 |
| 5% | 0.000 |
| 10% | 0.000 |
| 25% | 0.333 |
| 50% | 0.667 |
| 75% | 1.000 |

### 12.4 Useful address-token Jaccard

| Quantile | Score |
| ---: | ---: |
| Minimum | 0.000 |
| 1% | 0.000 |
| 5% | 0.125 |
| 10% | 0.273 |
| 25% | 0.444 |
| 50% | 0.625 |
| 75% | 0.833 |
| 90% | 1.000 |

A single global threshold on one similarity metric will miss important positive regimes.

## 13. Token-rarity experiment

To estimate whether rare-token blocking is viable, a separate deterministic 1% document-frequency sample of training S2/S3 targets was used:

```text
numeric target ID suffix % 100 == 0
```

Document frequencies were counted separately by country and field. For every sampled positive pair, the least frequent shared core-name or useful address token was found.

Overall positive-pair coverage:

| Minimum shared-token document frequency in 1% target sample | Positive-pair coverage |
| ---: | ---: |
| 0 | 40.615% |
| At most 1 | 55.991% |
| At most 5 | 75.770% |
| At most 10 | 83.444% |
| At most 50 | 95.125% |
| At most 100 | 97.586% |

The rough full-corpus equivalents are approximately 100 times the sampled counts, but this scaling is only an estimate. For example, sample frequency 50 roughly corresponds to 5,000 full-corpus target documents.

This supports rare-token blocking, but a fuzzy and cross-script fallback is still required for the remaining positive links.

## 14. One-percent field-profile samples

Each source file was also profiled on records whose numeric ID suffix satisfied `% 100 == 0`.

Sample sizes:

| File | Sample rows |
| --- | ---: |
| `train_source1.tsv` | 22,224 |
| `train_source2.tsv` | 50,580 |
| `train_source3.tsv` | 53,149 |
| `test_source1.tsv` | 17,706 |
| `test_source2.tsv` | 48,830 |
| `test_source3.tsv` | 50,764 |

Selected findings:

- Sampled S1 names in US, India, and France were Latin-script.
- India target records include Devanagari and several other Indic scripts.
- In the training S2 India sample of 20,371 records:
  - Latin: 15,617.
  - Devanagari: 2,628.
  - Other non-Latin: 1,936.
  - Latin plus Devanagari: 113.
  - Latin plus other: 77.
- In the training S3 India sample of 21,474 records:
  - Latin: 18,641.
  - Devanagari: 1,441.
  - Other non-Latin: 1,051.
  - Latin plus Devanagari: 193.
  - Latin plus other: 148.
- Approximately 3% to 4% of sampled target names look like websites or contain domains; sampled S1 names do not.
- Approximately 2% to 3% of sampled target addresses contain the literal token `null`.
- Target names contain more symbol noise and numeric substitutions than S1 names.
- France S1 names are shorter on average than India S1 names and commonly end with `SARL`, `SAS`, `EURL`, `SA`, `SASU`, or `SCI`.

## 15. Representative hard positive pairs

These examples are from the deterministic positive-pair sample and show cases that exact/fuzzy Latin matching alone can miss.

### 15.1 Missing target address and typo/digit substitution

```text
S1: K+ Elm, Inc
S3: K+ E1m,

S1 address: 27 Parkview Drive, Eastchester, NY
S3 address: [empty]
```

### 15.2 Missing target address and accent/symbol noise

```text
S1: Buex
S3: #búex

S1 address: Saint Louis, Fl 0, MO, 2501 Hackman Drive
S3 address: [empty]
```

### 15.3 Latin to Devanagari

```text
S1: Jain Industries Private Limited
S3: जैन इंडस्ट्रीज प्राइवेट लिमिटेड
```

### 15.4 Latin to Gujarati with partial address

```text
S1 name: Gold Southern Products Private Limited
S3 name: ગોલ્ડ સધર્ન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ

S1 address: 63, Armanpark, Uttarsanda Road, Vill: Nadiad (Mog), Tal: Nadiad,
            District: Kheda, Nadiad, Nadiad, Kheda, Gujarat
S3 address: GJ, Mitral, 63
```

### 15.5 Latin to Tamil

```text
S1 name: Raj Products Limited
S3 name: ராஜ் புராடக்ட்ஸ் லிமிடெட்

S1 address: 5A1, Chidambaranathan Street, Agastheeswaram, Kanyakumari, Tamil Nadu
S3 address: Nagercoil, TN, 5A1
```

### 15.6 Changed descriptor and empty address

```text
S1: AW Fuse
S2: AW Service

S1 address: 1008 3rd Street, Lincoln, IL
S2 address: [empty]
```

These very hard pairs should not dictate a permissive threshold that creates many false positives. They motivate fallback retrieval, graph/sibling evidence, and calibrated abstention.

## 16. Metric and decision implications

The leaderboard metric is macro entity-level F-beta with beta 0.5:

```text
F0.5 = 1.25 * precision * recall / (0.25 * precision + recall)
```

It weights precision more than recall and computes a score independently for each S1 anchor before averaging.

Consequences:

- Pairwise AUC is not the optimization target.
- Candidate recall sets an upper bound, but permissive final thresholds can damage the score.
- For a one-match entity, predicting the correct target plus one false target gives precision 0.5, recall 1.0, and F0.5 approximately 0.556.
- For a four-match entity, predicting three correct targets and no false targets gives precision 1.0, recall 0.75, and F0.5 0.9375.
- A borderline link should often be omitted when it risks a false merge.
- Singleton decisions require an explicit anchor-level confidence model or rule.

## 17. Proposed solution architecture

The following is a proposed design, not yet implemented or validated.

### 17.1 Multi-view normalization

Keep several representations rather than one destructive canonical string:

- Unicode NFKC plus casefolding.
- Accent-preserving and accent-folded variants.
- Full name tokens.
- Core/legal-suffix-stripped name tokens.
- Order-insensitive token set.
- Original script plus transliterated/phonetic view for Indian records.
- URL/domain stem for website-style names.
- Original address text.
- Canonical address tokens.
- Numeric address components.
- State/locality components learned only from provided data.
- Conservative leetspeak variants such as `1/l/i`, `8/b`, and `0/o` as features, not irreversible substitutions.

### 17.2 Candidate generation

Generate candidates separately by observed country and target source, subject to a complete-label audit before making country a hard gate.

Candidate routes:

1. Exact core-name index.
2. Exact sorted-name-token index.
3. Exact canonical-address index.
4. Rare core-name token postings.
5. Rare useful address token postings.
6. Building/house number plus locality or state token.
7. Name character 3–5-gram or BM25 top-K retrieval.
8. Address character retrieval.
9. Transliteration-aware Indian name retrieval.
10. URL/domain-stem retrieval.

Union the routes, use inexpensive retrieval scores to limit the set, and sweep final candidate caps such as 5, 10, 20, and 50 per target source.

The `candidate_pairs.tsv` file must describe the exact final set passed to the expensive matching model, not an earlier raw retrieval union.

### 17.3 Pairwise features

Candidate features should include:

- exact normalized-name equality;
- exact core-name equality;
- order-insensitive name-token equality;
- name character n-gram similarity;
- edit distance and Jaro-Winkler-style similarity;
- name token Jaccard;
- IDF-weighted shared-name tokens;
- acronym compatibility;
- original-script and transliterated similarities;
- exact/canonical address equality;
- directional address containment;
- address character similarity;
- address token Jaccard;
- exact and contradictory number features;
- building, postal, locality, state, and street agreement;
- missingness and length ratios;
- target source;
- retrieval route and rank;
- best-score versus runner-up margin;
- cross-source sibling support.

### 17.4 Model

Start with a gradient-boosted tree classifier over engineered features.

Reasons:

- The dataset contains millions of positives and potentially tens of millions of blocked negatives.
- Tabular interactions between name, address, numeric, missingness, and source features are important.
- Trees are cheaper to train and rerank at this scale than a transformer.
- Probability calibration and feature attribution are easier.

Use separate S2/S3 models or source-specific calibration because address corruption differs materially between the two sources.

Do not train primarily on random same-country negatives. Mine hard negatives from the actual candidate generator, competing anchors, similar names at different addresses, similar addresses with different names, and prior false positives.

### 17.5 Global decoding

- Allow several targets per S1 anchor.
- Allow each S2/S3 target to be assigned to at most one S1 anchor, based on the exact training-label exclusivity finding.
- When anchors compete for a target, compare calibrated absolute score and best-versus-second-best margin.
- Leave a target unmatched when confidence is insufficient.
- Use the observed per-source match-count distributions only as soft priors.
- Add conservative graph/sibling support when high-confidence S2/S3 records for one anchor corroborate a weaker record.

### 17.6 Singleton and count decision

Train or calibrate an anchor-level decision using:

- top S2 score;
- top S3 score;
- best-versus-runner-up margins;
- number of strong candidates;
- name/address agreement;
- cross-source corroboration;
- source-specific evidence;
- predicted number of matches.

Consider choosing the number of retained links per anchor by maximizing validation macro F0.5 over sorted calibrated candidates rather than using one universal 0.5 probability threshold.

### 17.7 France zero-shot behavior

- Do not require a country one-hot category learned during training.
- Reuse language-agnostic string, token, numeric, and address features.
- Learn frequent low-information French suffixes from the supplied test corpus without using external identity data.
- Use conservative calibration transferred from Latin-script US pairs.
- Diagnose high-confidence France pseudo-pairs created by exact core name plus strong address agreement, but do not blindly self-train.
- Measure France candidate counts and score distributions separately before submission.

## 18. Validation design

The validation split must be made by S1 anchor/cluster.

For a selected validation anchor, all of its S2/S3 positive records belong to the validation group. This prevents variants of the same business from leaking across train and validation.

Recommended validation procedure:

1. Split S1 anchors, stratifying by country, match count, and S2/S3 composition.
2. Build the validation candidate corpus and run the real blocking pipeline.
3. Measure candidate recall before evaluating the classifier.
4. Train on hard negatives produced by the train blocker.
5. Generate grouped predictions for every validation anchor.
6. Compute the exact macro entity-level F0.5.
7. Report:
   - candidate recall;
   - average and tail candidates per anchor;
   - pair precision/recall;
   - macro F0.5;
   - singleton accuracy;
   - target conflict rate;
   - US and India slices;
   - S2 and S3 slices;
   - missing-address slice;
   - script-mismatch slice;
   - match-count slices.
8. Simulate zero-shot behavior by training on one training country and evaluating on the other, while recognizing that France is closer to the US in script but not in address/legal vocabulary.
9. Tune thresholds and count decisions only on out-of-fold predictions.

## 19. Submission requirements

The live leaderboard scores only `matching_results.tsv`, but the final package requires both:

```text
output/matching_results.tsv
output/candidate_pairs.tsv
```

Rules include:

- one row for every test S1 anchor;
- exact headers;
- tab-separated output;
- comma-separated target IDs within the second column;
- empty list for a predicted singleton;
- no duplicate S1 rows;
- no duplicate target IDs within one list;
- only test S2/S3 IDs may be predicted;
- every final match should be present in the corresponding candidate list.

Run:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The optional `--check-ids` validation loads the S2/S3 ID set and may require several gigabytes of memory.

## 20. Current status and next implementation steps

No production matching pipeline has been implemented yet.

Completed work:

- Read the supplied PDFs, README, documentation template, and validator.
- Counted all train/test rows and countries.
- Computed complete ground-truth multiplicity and source-composition distributions.
- Verified that every positive target ID is assigned to exactly one S1 anchor in training.
- Measured missing-field counts.
- Profiled deterministic anchor, positive-pair, target-token-frequency, and field samples.
- Defined the exact normalization used for the reported percentages.
- Identified cross-script Indian matches and representative hard positives.
- Proposed a multi-stage blocker, tree classifier, and global decoder.

Recommended next work:

1. Implement reusable normalization with unit tests covering punctuation, suffixes, abbreviations, Unicode, websites, numeric components, and Indic scripts.
2. Build a validation scorer for the exact macro F0.5 definition.
3. Implement a disk-efficient per-country/per-source rare-token inverted index.
4. Measure blocking recall and candidate count on held-out anchors.
5. Add fuzzy name/address fallback retrieval.
6. Train the first gradient-boosted matcher with hard negatives.
7. Add source-specific calibration and target-to-anchor exclusivity.
8. Build the anchor-level singleton/count decoder.
9. Create France diagnostics.
10. Generate and validate both required output TSV files.

## 21. Exact Devanagari audit

This section replaces the earlier 1% script-profile estimates with exact full-file counts.

### 21.1 Counting definition

A field is counted as containing Devanagari when it contains at least one Unicode codepoint in the inclusive range U+0900 through U+097F. This is deliberately a literal, reproducible codepoint test. It does not attempt language identification: Hindi, Marathi, Sanskrit, or another language written in Devanagari all count the same way.

For Devanagari-bearing names:

- `Devanagari-only script` means the name contains at least one U+0900–U+097F codepoint and contains no Latin letter and no alphabetic character from another script. Digits and punctuation are permitted.
- `Devanagari + Latin` means the name contains both a Devanagari codepoint and a Latin letter. No names in this audit combined Devanagari with a third alphabetic script.
- `Either field` means the name or address, or both, contains Devanagari. It is a record count, so a record with Devanagari in both fields is counted once.

### 21.2 Anchor counts

| Split | S1 rows | Devanagari name | Devanagari address | Either field |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,206,821 | 0 | 0 | 0 |
| Test | 1,732,544 | 0 | 0 | 0 |

This is a strong structural fact: all observed Devanagari is on the target side. Devanagari matching is therefore primarily Latin-anchor-to-Devanagari-target retrieval rather than same-script matching.

### 21.3 Target counts by source

| Split/source | Rows | Name has Devanagari | Address has Devanagari | Either field | Both fields | Devanagari-only name | Devanagari + Latin name |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train S2 | 5,034,616 | 269,424 | 277,523 | 479,500 | 67,447 | 258,863 | 10,561 |
| Train S3 | 5,285,603 | 158,003 | 275,861 | 394,374 | 39,490 | 138,187 | 19,816 |
| Test S2 | 4,887,273 | 309,103 | 318,369 | 550,530 | 76,942 | 297,604 | 11,499 |
| Test S3 | 5,082,316 | 181,068 | 319,073 | 454,885 | 45,256 | 158,605 | 22,463 |

All of these records are in the India slice. US and France have zero Devanagari-bearing names and addresses under the codepoint definition.

### 21.4 Combined target counts

| Split | Target rows | Name has Devanagari | Address has Devanagari | Either field | Both fields | Devanagari-only name | Devanagari + Latin name |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train S2+S3 | 10,320,219 | 427,427 (4.1416%) | 553,384 (5.3621%) | 873,874 (8.4676%) | 106,937 (1.0362%) | 397,050 (3.8473%) | 30,377 (0.2943%) |
| Test S2+S3 | 9,969,589 | 490,171 (4.9167%) | 637,442 (6.3939%) | 1,005,415 (10.0848%) | 122,198 (1.2257%) | 456,209 (4.5760%) | 33,962 (0.3407%) |

Within India targets only:

- Training: 427,427 of 4,133,346 names contain Devanagari (10.3409%); 873,874 records contain it in either field (21.1420%).
- Test: 490,171 of 4,717,565 names contain Devanagari (10.3903%); 1,005,415 records contain it in either field (21.3122%).

The test distribution is very similar to training within India, so the cross-script problem is not an isolated anomaly. Note that these are all target records, not necessarily true matched targets. Positive-label-only Devanagari counts have not yet been computed.

## 22. Cross-script matching without an SLM

The intended operation is usually **transliteration**, not semantic translation. Transliteration maps how a name is written or pronounced across scripts; translation changes its meaning. For example:

```text
Latin anchor:       Jain Industries Private Limited
Devanagari target:  जैन इंडस्ट्रीज प्राइवेट लिमिटेड
Romanized view:     jain industries private limited
```

An SLM is not required for this. The safest architecture keeps the original string and adds one or more deterministic or training-derived comparison views. It never replaces the original field.

### 22.1 Preferred option: deterministic transliteration as a retrieval view

- Use a fixed Unicode/script-table transliterator if challenge rules permit deterministic linguistic normalization.
- Generate a romanized Devanagari view for target names and possibly addresses.
- Normalize both the Latin S1 name and the romanized target with the same punctuation, spacing, casefolding, and legal-token rules.
- Use romanized similarity to retrieve candidates. Do not accept a link solely because the transliteration is similar.
- Preserve alternate spellings where possible. Indian romanization is not one-to-one because of schwa deletion, long vowels, aspiration, and English borrowings. A permissive character n-gram score is more robust than demanding exact romanized equality.

Upside: high direct coverage of Latin-to-Devanagari names, deterministic, fast, explainable, and not dependent on a generative model.

Downside: romanization ambiguity can create collisions; transliterated legal words such as `private` and `limited` are frequent and weak evidence; rule quality may vary by language and English loanword spelling.

Before using any external transliteration package, confirm that the challenge permits the package and its static language tables. This is different from looking up businesses or calling an external model/service.

### 22.2 Learn cross-script correspondences only from supplied labels

If even a general transliteration table is disallowed, training labels can still teach correspondences without external business data:

1. Start from positive S1-target pairs with strong address/number evidence.
2. Tokenize Latin S1 names and Devanagari target names.
3. Learn token or character-span alignments from repeated positive co-occurrence.
4. Score a Latin token and Devanagari token using conditional probability, PMI, or an alignment model.
5. Downweight ubiquitous legal-form alignments such as `private`, `limited`, and their Devanagari spellings.
6. Use the learned score as one retrieval/model feature, not as proof of identity.

This is transductive/weakly supervised matching from the provided dataset, not external translation. It must be learned on training folds only during validation to avoid leakage.

### 22.3 Script-independent fallback routes

These remain essential even when transliteration is available:

- exact or near-exact house/building numbers;
- postal code, phone-like numeric fragments, plot/unit numbers;
- Latin locality, state, district, or street tokens remaining in a mixed-script address;
- rare address-token overlap;
- an S2/S3 sibling record that has a Latin name and a near-identical address to the Devanagari record;
- graph propagation from a very high-confidence Latin target assigned to the anchor;
- target-source and country constraints;
- conflict evidence when another anchor explains the target much better.

The exact audit shows 553,384 training and 637,442 test target addresses contain Devanagari, but many of those addresses remain partly Latin or numeric. A binary `address is Devanagari` flag should not discard the usable Latin and numeric fragments.

### 22.4 If all transliteration is forbidden

Use a dedicated fallback blocker that retrieves by numeric/address evidence and target-target sibling linkage. The sequence is:

1. Retrieve high-confidence Latin target matches for the S1 anchor.
2. Retrieve target records sharing distinctive address numbers/tokens with those matched targets.
3. Permit a Devanagari-name target into the candidate pool when its address/sibling evidence is strong.
4. Apply a stricter final threshold because no direct name evidence is available.
5. Measure recall specifically on the known script-mismatch validation slice.

This will not recover every empty-address Devanagari target. Those cases may be irreducibly ambiguous without transliteration or a corroborating sibling, and the precision-weighted F0.5 metric may favor abstention.

## 23. Detailed multi-view filtering and retrieval catalogue

“Filter” should not mean that every view must agree. Most name/address views are **candidate routes** whose results are unioned. Only genuinely safe constraints should be hard intersections. A robust sequence is:

```text
hard partition/gates
    -> union of exact, token, fuzzy, script, address, and sibling retrieval views
    -> cheap candidate cap/prefilter
    -> pairwise feature model
    -> global conflict and anchor-count decoder
```

### 23.1 Hard partitions and cautious gates

| View/filter | What it checks and catches | Upside | What it misses or risks | Stage |
| --- | --- | --- | --- | --- |
| Country equality | Anchor and target have the same supplied country | Enormous speedup and fewer absurd matches | Unsafe if country labels are corrupted; audit cross-country positives before making hard | Hard gate after audit |
| Target source | Build independent S2 and S3 indexes/calibration | Handles source-specific corruption and guarantees source quotas are not accidentally mixed | A single source-specific route can miss support from the other source | Partition, not exclusion |
| Script detector | Latin, Devanagari, other Indic, or mixed | Routes records to appropriate transliteration/character models | Script does not identify language or business; mixed fields retain useful Latin | Router/feature |
| Missing-field route | Detect empty/`null`/placeholder address or unusable name | Avoids penalizing absent evidence and selects name-only model | A missing field removes a major disambiguator and raises collision risk | Router/feature |

### 23.2 Name retrieval views

| View/filter | Checks/catches | Upside | Misses/risks |
| --- | --- | --- | --- |
| Exact NFKC + casefold name | Case and Unicode presentation differences | Very high precision, cheap hash lookup | Punctuation, typos, token reorder, legal-form changes, script changes |
| Punctuation/whitespace normalized name | `A.B.C.` vs `ABC`, repeated spaces and symbols | High precision for formatting noise | Can merge meaningful punctuation; does not solve spelling variation |
| `&`/`and` variant | Ampersand versus word | Recovers common business-name formatting | `and` is frequent and weak on its own |
| Legal-form-stripped core name | Removes `Pvt`, `Private`, `Ltd`, `Inc`, etc. | Large recall gain; exact core equality was much stronger than exact full-name equality | Short/generic cores collide; legal tokens sometimes distinguish entities; do not discard them permanently |
| Sorted token multiset/set | Token reorder and inserted legal tokens | Robust to ordering corruption | Loses order and duplicate information; generic token sets collide |
| Rare-name-token inverted index | At least one high-IDF shared core token | Excellent scalable blocking; explainable | Misses spelling/script changes and pairs sharing only common tokens; very common tokens create huge postings |
| BM25/token retrieval | Weighted partial token overlap | Naturally discounts common terms and handles partial names | Still needs shared token text; parameter and tokenization sensitive |
| Character n-gram TF-IDF | Similar substrings despite spelling/punctuation errors | Strong general fuzzy baseline; language-agnostic within a script | Weak across scripts; short names create false similarity; index can be large |
| Edit/Jaro-Winkler rerank | Local typos/transpositions/prefix agreement | Interpretable and useful on short candidate lists | Too expensive as all-pairs retrieval; raw edit distance mishandles long insertions and token reorder |
| Acronym/initialism | `ABC Technologies` versus `A B C Tech` | Recovers severe abbreviations | Extremely collision-prone for short initials; require address corroboration |
| URL/domain stem | `wilfordhancock.com` versus `Wilford Hancock` | Directly handles the 3–4% website-style target names seen in samples | Hosting/subdomain boilerplate and generic domains; domain may not resemble legal name |
| Accent-preserving plus accent-folded | `Buex` versus `búex` | Recovers diacritic corruption and supports France | Accent folding can merge genuinely distinct forms; keep both views |
| Conservative OCR/leetspeak variants | `E1m` versus `Elm`, `0/O`, `8/B` | Recovers synthetic corruption | Aggressive substitution creates many false matches; use only as feature/retrieval expansion |
| Deterministic transliteration | Latin anchor versus Devanagari/other Indic target | Direct cross-script evidence without an SLM | Ambiguous spellings and legal-token collisions; rule coverage varies |
| Phonetic skeleton | Sound-like spellings after romanization | Helps vowel/transliteration variants | Poor for multilingual names, acronyms, and short strings; collision-heavy |

### 23.3 Address retrieval views

| View/filter | Checks/catches | Upside | Misses/risks |
| --- | --- | --- | --- |
| Exact normalized address | Case/punctuation/abbreviation normalization | Very high precision when present | Only 11.543% exact canonical coverage in sampled positives; order/partial/corruption common |
| Address token-set equality/Jaccard | Reordered address components | Robust to order | Common locality/state tokens inflate scores; partial addresses reduce Jaccard |
| Directional containment | Short target address contained in long anchor address | Handles truncated targets better than symmetric Jaccard | Tiny generic snippets can look contained; length/IDF safeguards needed |
| Rare address-token index | Shared distinctive locality/street/building token | High recall in sampled positives; scalable | OCR/script changes and common locality names; spelling variants break exact postings |
| House/building/plot number | Exact numeric components | Often highly discriminative and script-independent | Same building hosts many businesses; number corruption; unit versus house ambiguity |
| Postal code | Exact/near postal agreement | Strong geographic filter | Missing or corrupted; formatting differs; do not infer externally if unavailable |
| Locality/state/district | Geographic component agreement | Helps disambiguate common names and supports cross-script cases | Common values are weak; parsing errors; abbreviations and multilingual spellings |
| Address character n-grams | Spelling errors and partial textual overlap | Useful fuzzy fallback | Expensive over millions; common address phrases; weak across scripts |
| Numeric-set compatibility | Shared/contradictory numbers across the whole address | Script-independent feature with both positive and negative evidence | Some numbers are dates/floors/roads; absence is not contradiction |
| Explicit number contradiction | Both sides have confident house/postal values that disagree | Powerful negative evidence | Parsing must identify comparable number roles; naive “any differing number” is wrong |

### 23.4 Relational and decision views

| View/filter | Checks/catches | Upside | Misses/risks |
| --- | --- | --- | --- |
| S2/S3 sibling graph | Targets with highly similar addresses/names support one another | Can rescue empty-address or cross-script records through a stronger sibling | Error propagation can merge businesses; require a high-confidence seed and conservative edges |
| Competing-anchor ownership | Same target candidate retrieved for multiple anchors | Training shows target-to-S1 exclusivity, so competition is strong evidence | Absolute calibration errors can hand a target to the wrong anchor |
| Best-vs-runner-up margin | Difference between top anchor scores for a target | Prevents ambiguous forced assignment | A large margin among two bad candidates is not sufficient; require absolute score too |
| Anchor top-score/count profile | Number and strength of predicted links | Enables explicit singleton and multi-record decisions | Hard count priors can suppress legitimate tails; use soft priors |
| Retrieval route/rank | Which view found the pair and at what rank | Lets the model learn reliability; improves diagnostics | Route availability changes when blocker changes, requiring retraining |

### 23.5 Recommended combination rule

- Hard-intersect only audited country and source partitions.
- Union exact-name, core-name, rare-token, fuzzy-name, exact-address, rare-address, numeric, transliteration, URL, and sibling routes.
- Give every route a per-anchor/per-source cap so generic tokens cannot explode the pool.
- Never require both name and address agreement during blocking: missing/corrupted fields would destroy recall.
- Let the supervised pair model combine agreements, contradictions, missingness, route, and rank.
- Let the decoder enforce target exclusivity and precision-oriented abstention.

## 24. Fuzzy search, retrieval, and top-K sweeps

### 24.1 Pipeline location

Fuzzy search belongs primarily to **candidate generation**, after normalization/index construction and before the expensive pair classifier. It turns roughly ten million possible targets into tens of plausible targets for each anchor.

It can appear again as pairwise similarity features, but that is reranking, not corpus retrieval. Computing edit distance from every anchor to every target is infeasible.

### 24.2 Practical retrieval sequence

For each country and target source:

1. Build normalized name and address views for targets.
2. Build a character 3–5-gram TF-IDF index and/or a BM25 token index.
3. Query each S1 anchor against the S2 name index, S2 address index, S3 name index, and S3 address index.
4. Retrieve only the top K from each fuzzy view.
5. Union these with uncapped high-confidence exact hits and capped rare-token/numeric/transliteration routes.
6. Deduplicate `(S1, target)` pairs while retaining every route, rank, and retrieval score.
7. Apply cheap sanity checks or a first-stage model to reduce the union to the exact candidate set sent to the full matcher.
8. Compute richer edit, token, number, and cross-source features only for that set.

Example with `K=20`: retrieve up to 20 S2-name, 20 S2-address, 20 S3-name, and 20 S3-address neighbors. The union is at most 80 before exact/rare/script routes and usually smaller because views overlap. This does not mean predicting 20 matches; it only means allowing the classifier to inspect them.

### 24.3 Meaning of “sweep top-K values”

Do not guess one K. Re-run the blocker on held-out anchor groups with values such as:

```text
K = 5, 10, 20, 50, 100
```

For each K, record:

- pair candidate recall: true target links retrieved / all true target links;
- any-hit anchor recall: anchors with at least one true target retrieved / non-singleton anchors;
- complete-set anchor recall: anchors for which every true target was retrieved;
- mean, median, P95, and P99 candidates per anchor;
- candidate-file size, retrieval time, feature-generation time, and model-scoring time;
- final macro F0.5 after retraining or consistently evaluating the downstream model;
- the same metrics by country, S2/S3, missing address, script mismatch, and match count.

Select the smallest K near the recall/compute elbow. Because routes differ, the final system may use different K values: for example, a larger name K for short/generic names, a smaller address K when strong numbers are present, and a dedicated larger cross-script fallback K.

### 24.4 Important distinctions

- K is normally per retrieval view and per target source, not a global predicted-match count.
- A candidate cap belongs before classification; a final output count is chosen after classification/decoding.
- Exact hash hits need not compete with fuzzy top-K, although enormous collision buckets should be capped or reranked.
- Recall must be measured using the real union of all routes. Evaluating only a fuzzy index understates the system and can encourage an unnecessarily large K.
- `candidate_pairs.tsv` must contain the final candidates actually passed to the expensive matching model, not every raw posting considered internally.

## 25. Realistic hard-negative mining

Random same-country pairs are usually trivial and teach the model little. A hard negative is a nonmatching pair that the production candidate generator considers plausible.

### 25.1 Leakage-safe mining procedure

1. Split data by S1 anchor, keeping every positive S2/S3 record for an anchor in the same fold.
2. Fit any learned IDF, token correspondence, transliteration alignment, or calibration on the training fold only.
3. Run the actual multi-view candidate generator for training anchors against the permitted training target pool.
4. Mark every retrieved ground-truth link as positive and every retrieved nonlink as a negative, subject to label-completeness assumptions.
5. Store route, rank, retrieval score, country, source, script, missingness, and collision-bucket metadata.
6. Sample negatives per anchor and per difficulty family so a few huge ambiguous buckets do not dominate.
7. Train the pair model, score a broader candidate pool, then add the highest-scoring false pairs for another training round.
8. Evaluate on untouched grouped validation anchors and mine validation negatives only for diagnostics, never to retrain the same fold.

### 25.2 Negative families to mine

| Negative family | Construction | What it teaches |
| --- | --- | --- |
| Top fuzzy-name impostor | Highest-ranked name candidate not in the anchor's truth set | Typos and substring similarity are insufficient |
| Exact/core-name collision | Same normalized/core name but different address/entity | Generic or reused business names need location evidence |
| Token-set collision | Same tokens in different order or partial shared tokens | Order-free matching can overmerge |
| Acronym collision | Same initials/abbreviation, different entity | Acronyms require corroboration |
| URL/domain collision | Similar site/domain stem, wrong entity | Web-style normalization is not proof |
| Address-neighbor impostor | High address similarity but different name | Shared buildings, malls, accountants, or registered offices host many firms |
| House-number/locality collision | Same building number and locality, different business | Numeric agreement is powerful but not unique |
| Name-near/address-conflict | Strong name similarity with confident conflicting house/postal evidence | Explicit contradictions should override fuzzy name evidence |
| Address-near/name-conflict | Strong address evidence but unrelated core names | Co-location alone should not merge businesses |
| Cross-anchor positive | Target is a known positive for anchor B but retrieved for anchor A | Strongest safe ownership negative because training labels show exclusivity |
| Same-family/legal-form variant | Similar core plus changed `Pvt/Ltd/LLP` structure but wrong entity | Stripping legal tokens can create collisions |
| Short-name collision | One- or two-token generic names | Short strings need stricter thresholds |
| Missing-address impostor | Name-only near match when target address is empty | Missing evidence is uncertainty, not agreement |
| Cross-script impostor | Transliteration/phonetic near match but wrong address/owner | Transliteration ambiguity and frequent legal words |
| Sibling-graph near miss | Target adjacent to a cluster but belonging elsewhere | Prevent graph propagation from snowballing |

### 25.3 Strongest negative source: competing anchors

Every training target ID is positive for at most one S1 anchor. Therefore, if target `T` is a labelled positive for anchor `B` but the blocker retrieves it for anchor `A`, `(A,T)` is an especially realistic and trustworthy negative. It directly trains the model for the global ownership conflicts that will occur at inference.

Mine both:

- absolute impostors: high score for the wrong anchor;
- small-margin conflicts: the correct and incorrect anchors have similar scores.

Include features comparing the target's best and second-best anchor evidence. Train/calibrate pair scores first, then enforce exclusivity in the decoder.

### 25.4 Iterative hard-negative mining

Round 0 can use blocker-ranked negatives. Then:

1. train the initial model;
2. score a broad training-fold candidate pool;
3. collect high-scoring false positives, especially those above or near the operating threshold;
4. collect wrong-anchor competitors with small score margins;
5. stratify by source, country, script, missingness, and retrieval route;
6. add them to the next training set and retrain;
7. stop when validation F0.5 and false-positive slices stop improving.

This aligns training with the model's current failure surface rather than with arbitrary random pairs.

### 25.5 Starting sampling recipe

Keep all positive pairs. As a starting point, retain roughly 10–20 hard negatives per anchor, not as a fixed truth but as a tunable budget:

- 3–5 top fuzzy-name nonmatches;
- 2–4 exact/core/token-name collisions;
- 2–4 address/numeric collisions;
- 1–3 cross-anchor positive conflicts;
- 1–2 cross-script/transliteration confusers where applicable;
- 1–2 missing-field, short-name, URL, acronym, or graph confusers.

Cap each family per anchor, but preserve additional very high-scoring model false positives. Balance S2/S3 and US/India; oversample rare but important script-mismatch and missing-address regimes. The exact ratios must be tuned using grouped validation.

### 25.6 Weighting, calibration, and false-negative cautions

- Weight or sample by anchor so entities with thousands of candidates do not dominate entities with ten.
- Preserve source and country representation or use explicit sample weights.
- Record the sampling probability or difficulty family; sampled training probabilities will not equal real candidate-pool priors.
- Calibrate on out-of-fold predictions drawn from the real blocker distribution, not on the artificially balanced training sample.
- Do not automatically treat an unlabeled pair as negative until ground-truth completeness is confirmed. Cross-anchor positives are safe negatives for the current anchor under the observed exclusivity rule.
- Never mine validation false positives back into the model being evaluated; use cross-fitting or a separate mining fold.
- Track performance by negative family. Aggregate AUC can improve while the precision-critical collision families remain bad.

## 26. Updated pretrained-model policy

The user supplied the following authoritative challenge-policy clarification:

- Pretrained open-weight models are allowed.
- Every model must use an MIT or Apache-2.0 license.
- Every individual model must have at most 8 billion parameters.
- Models must run completely offline; no hosted or live API calls are allowed.
- A pretrained model may be fine-tuned only on the provided challenge data.
- The parameter and license rule is checked independently for every model used, including embedders, rerankers, matchers, translation/transliteration models, and preprocessing models.
- Hosted Claude, Gemini, ChatGPT, and similar APIs are prohibited.

Before adopting any pretrained component, create a model registry containing exact model repository/release, parameter count, license file, local artifact checksum, task, whether weights were changed, and fine-tuning-data provenance. Do not rely only on a model-card tag: retain the actual license text and verify that the particular checkpoint, tokenizer, and any bundled code/assets satisfy the rule.

This update permits an offline multilingual embedder, cross-encoder reranker, or small transliteration/generative model as an optional view. It does not make such models automatically useful. The exact, token, numeric, address, and graph routes remain important because they are faster, more explainable, and often more precise.

## 27. Candidate-set architecture under the updated model policy

For anchor `a`, each independently useful retrieval view produces a set of target IDs:

```text
C_exact_name(a)
C_core_name(a)
C_rare_name(a)
C_fuzzy_name(a)
C_embedding_name(a)
C_translit_name(a)
C_exact_address(a)
C_rare_address(a)
C_fuzzy_address(a)
C_numeric_address(a)
C_embedding_address(a)
C_sibling_graph(a)
```

The raw candidate pool is the union, not the intersection:

```text
C_raw(a) = union over all enabled views C_view(a)
```

Deduplicate `(anchor_id, target_id)` while preserving, for every pair, the list of views that retrieved it, each view's rank, score, query variant, and target source. A pair retrieved by three independent views should carry stronger evidence than a pair retrieved only at rank 50 by one weak view.

### 27.1 Lexical name candidate sets

1. **Exact normalized full-name set**: Unicode normalization, casefolding, punctuation/spacing normalization. Hash lookup; usually high precision.
2. **Exact core-name set**: legal suffixes removed or downweighted. High recall but collision-prone for generic cores.
3. **Sorted-token/set set**: order-insensitive equality or containment. Handles reordering.
4. **Rare-name-token set**: inverted postings for high-IDF core tokens. Scalable and explainable.
5. **BM25 name set**: top-K token-weighted results. Handles partial overlap.
6. **Character n-gram name set**: top-K TF-IDF/cosine results over 3–5-grams. Handles typos, punctuation, and moderate spelling changes.
7. **Acronym set**: initialism-compatible targets, under a small cap and usually requiring later address support.
8. **URL/domain-stem set**: target domains compared to compacted anchor names.
9. **OCR/leetspeak set**: conservative alternate views such as `1/l/i` and `0/o`, used with a tight cap.

### 27.2 Cross-script and learned name candidate sets

1. **Deterministic transliteration set**: romanize Devanagari/other Indic target names, then use exact, BM25, and character n-gram retrieval.
2. **Offline model-assisted transliteration set**: if a compliant model is used, generate one or a few transliteration hypotheses offline. Cache them, retain the original text, and use the hypotheses only for retrieval/features—not as identity proof.
3. **Multilingual name-embedding set**: encode S1 and target names with a compliant offline multilingual embedder and query an approximate-nearest-neighbor index separately by country/source.
4. **Provided-data fine-tuned embedding set**: contrastively fine-tune the compliant embedder on supplied positive pairs plus mined hard negatives. Validation fine-tuning must remain fold-safe.
5. **Learned cross-script token set**: retrieve using Latin–Indic token/character correspondences learned only from supplied training positives.

Embedding retrieval should complement lexical retrieval. It can find cross-script or semantically/phonologically related strings with no shared characters, but it can also retrieve businesses with similar meanings rather than identical identities.

### 27.3 Address and numeric candidate sets

1. **Exact canonical-address set**.
2. **Rare address-token set**.
3. **BM25 address set**.
4. **Character n-gram address set**.
5. **Directional containment set** for truncated target addresses.
6. **House/building/plot number plus locality set**.
7. **Postal-code set**, where present.
8. **Numeric-signature set** using normalized number groups with role-aware comparisons where possible.
9. **Multilingual address-embedding set**, preferably at a smaller K because generic addresses can be semantically similar.

### 27.4 Relational candidate sets

1. **S2/S3 sibling set**: after a strong initial anchor-target match, retrieve targets similar to that target by name/address.
2. **Shared distinctive-address cluster set**: targets in a conservative address cluster become candidates for the same anchor.
3. **Model expansion set**: after a first scoring pass, add a tightly capped set of neighbors of high-confidence targets, then rescore all expanded pairs.

Graph expansion must be limited to one or a small fixed number of rounds and require high-confidence seed/edge scores to prevent error cascades.

### 27.5 Candidate-union controls

- Partition indexes by country and S2/S3 source where audited safe.
- Give each fuzzy/model view its own top-K and minimum score.
- Allow high-confidence exact hits outside the fuzzy K, but cap huge collision buckets.
- Deduplicate pairs before expensive features.
- If `C_raw` is too large, use a cheap learned pre-ranker to retain a final cap per anchor/source while protecting exact and rare-token hits.
- Measure recall of the final post-cap candidate set, because that is the upper bound for matching.
- Write `candidate_pairs.tsv` from the exact candidate pairs actually evaluated by the expensive final matcher.

## 28. Revised end-to-end pipeline

### Stage 0: Compliance and reproducibility

- Create the model/license registry.
- Pin exact local model artifacts and checksums.
- Disable network/API paths during training and inference.
- Record that all fine-tuning examples originate from supplied files.

### Stage 1: Ingestion and audit

- Stream all TSVs with explicit tab delimiters.
- Validate IDs, country/source domains, missing values, duplicates, and label consistency.
- Preserve raw text alongside derived views.

### Stage 2: Grouped validation

- Split by S1 anchor, keeping all its positive targets together.
- Stratify by country, target count, source composition, missing address, and script-mismatch status.
- Fit learned normalization, IDF, token alignment, embedders, rerankers, and calibration on training folds only.

### Stage 3: Multi-view representation

- Produce full/core/token/character/domain/acronym name views.
- Produce original, canonical, token, character, locality, number, and postal address views.
- Produce script and transliteration views.
- Optionally produce compliant offline name/address embeddings.

### Stage 4: Target indexes

- Exact hash maps and token postings.
- BM25 and character n-gram indexes.
- Approximate-nearest-neighbor indexes for embeddings.
- Numeric/locality indexes.
- Conservative target-target/sibling index.

### Stage 5: Multi-view candidate retrieval

- Query every enabled view independently for S2 and S3.
- Union and deduplicate candidate pairs.
- Preserve retrieval provenance, ranks, and scores.
- Sweep per-view K and thresholds on grouped validation.

### Stage 6: Candidate reduction

- Protect strong exact/rare/numeric hits.
- Use inexpensive lexical, embedding, and contradiction features to cap large candidate pools.
- Measure pair recall, any-hit anchor recall, and complete-set anchor recall after this cap.

### Stage 7: Rich pair features

- Exact/core/token/character/edit similarities.
- Transliteration and multilingual embedding similarities.
- Address containment, rare-token, locality, postal, and role-aware number agreement/contradiction.
- Missingness, length ratios, source, country, script, route, rank, and cross-source sibling evidence.
- Best-anchor/second-anchor competition features where available without leakage.

### Stage 8: Pair scoring and reranking

- Begin with a gradient-boosted tree over engineered features.
- Optionally add a compliant offline cross-encoder/reranker on only the reduced candidate set.
- The cross-encoder output is another feature or score; it should not bypass structured numeric/address contradictions.
- Mine realistic hard negatives from the actual candidate generator and model false positives.

### Stage 9: Calibration

- Calibrate S2/S3 and relevant country/script regimes using out-of-fold predictions.
- Correct for negative sampling because sampled training prevalence differs from inference prevalence.

### Stage 10: Global decoding

- Permit multiple S2/S3 targets per S1 anchor.
- Permit each target to belong to at most one anchor.
- Use absolute score plus best-versus-runner-up ownership margin.
- Apply precision-oriented abstention and an anchor-level singleton/count decision.
- Use sibling evidence conservatively after strong seed assignments.

### Stage 11: Validation and ablation

- Compute exact macro entity-level F0.5.
- Report candidate recall before matcher performance.
- Slice by country, source, missing address, script mismatch, name length, and truth-set size.
- Ablate each candidate view: remove it and measure unique true pairs lost, candidate volume saved, and downstream F0.5 change.
- Compare lexical-only, lexical+transliteration, lexical+embedding, and full systems.

### Stage 12: Full training and test inference

- Refit only selected components using all supplied training data.
- Generate candidates for every test anchor.
- Score, calibrate, decode, and retain full provenance for diagnostics.

### Stage 13: Output validation

- Produce `matching_results.tsv` and `candidate_pairs.tsv`.
- Ensure every final match occurs in the candidate file.
- Run the provided validator and retain configuration/model/license manifests with the submission artifacts.

## 29. Fuzzy matching logic in this pipeline

“Fuzzy logic” here normally means approximate string/retrieval logic, not a formal fuzzy-rule inference system. It has two distinct jobs:

1. **Fuzzy retrieval**: find plausible targets without comparing an anchor to all targets.
2. **Fuzzy pair evidence**: quantify how strongly a retrieved anchor-target pair agrees.

### 29.1 Retrieval scores

- Character n-gram cosine measures shared local character patterns and tolerates small edits.
- BM25 rewards shared rare tokens and discounts frequent ones.
- Edit similarity measures the minimum character transformation, but should run only on retrieved pairs.
- Jaro-Winkler emphasizes matching prefixes and nearby character order.
- Token-set/containment scores handle word reorder and partial names/addresses.
- Transliteration similarity compares Latin anchors with romanized Indic targets.
- Embedding cosine can retrieve cross-script or structurally different variants, but may confuse semantic similarity with identity.

Each retrieval view returns its own top-K. The K is a search breadth, not a predicted match count.

### 29.2 Combining fuzzy evidence

Do not rely on a hand-written average such as `0.5 * name + 0.5 * address` for the final decision. Store the component scores separately and let a supervised model learn interactions such as:

```text
high name similarity + exact house number                 -> strong
high name similarity + conflicting postal/house evidence -> weak/negative
medium transliteration similarity + rare address overlap -> strong
high embedding similarity alone for a short generic name -> risky
empty address                                             -> unknown, not agreement
```

A transparent baseline may use staged fuzzy rules for retrieval:

```text
retrieve if any of:
  exact core name
  rare shared name token
  name n-gram score in top K
  transliteration score in top K
  embedding score in top K
  exact distinctive address/number rule
  address n-gram/BM25 score in top K
```

The final classifier should then use every score plus contradictions and missingness.

### 29.3 Per-view K sweep

Sweep K separately or in controlled bundles. A practical progression is:

```text
name character/BM25 K:       5, 10, 20, 50, 100
address fuzzy K:             5, 10, 20, 50
embedding K:                 5, 10, 20, 50
cross-script fallback K:    10, 25, 50, 100
```

For each setting, measure unique positive links added by that view, total candidates added, post-union candidate recall, tail candidate counts, latency/storage, and final macro F0.5. Choose the smallest setting near the recall/compute elbow, not simply the largest K.

## 30. Terminology correction and concrete stage-by-stage data flow

### 30.1 Anchor, target, candidate, match, and sibling

- **Anchor**: one row from S1. This is the entity for which the system must return zero or more matching records.
- **Target**: any individual row from S2 or S3. Calling a row a target does not imply that it is a true match.
- **Candidate target**: an S2/S3 row retrieved for an S1 anchor and passed to later scoring.
- **Predicted match**: a candidate target accepted by the final matcher/decoder.
- **True match**: an S2/S3 row listed for an S1 anchor in training ground truth.
- **Sibling target**: informal shorthand for another S2/S3 row believed to describe the same underlying S1 entity. It should not be assumed to be a sibling until the evidence/model says so.

The primary candidate-generation operation is exactly:

```text
for each S1 anchor:
    retrieve similar/plausibly related S2 rows
    retrieve similar/plausibly related S3 rows
```

The earlier phrase “after finding a strong target, search for its siblings” describes an optional **second-pass candidate expansion**:

```text
S1 anchor A
    -> retrieves and strongly scores S2 row T2
    -> T2 is used as an additional query against other S2/S3 rows
    -> retrieves S3 row T3 that resembles T2 even if A does not directly resemble T3
    -> add (A,T3) as a candidate, then score it normally
```

This can rescue a cross-script or missing-field record, but it must occur only after initial S1-to-target retrieval/scoring, and the new candidate is not automatically accepted.

Example:

```text
S1 A:  Jain Industries Private Limited
        12 MG Road, Jaipur

S2 T2: Jain Industries Pvt Ltd
        Plot 12, M G Road, Jaipur

S3 T3: जैन इंडस्ट्रीज प्राइवेट लिमिटेड
        Plot 12, MG Road, Jaipur
```

Direct S1-to-S2 retrieval easily finds T2. Direct S1-to-S3 name retrieval may miss T3 without transliteration. Once T2 scores very strongly for A, target-to-target address retrieval from T2 can propose T3 for A. The final model still checks `(A,T3)`, and global decoding can reject it.

### 30.2 What “build target indexes” means

An index is a precomputed lookup/search structure over S2/S3 rows. It avoids comparing every S1 anchor with every target row. Index construction produces candidates; it does not decide matches.

Indexes should normally be separated by country and target source, for example:

```text
India-S2 name index
India-S3 name index
India-S2 address index
India-S3 address index
US-S2 ...
```

Concrete index types:

1. **Hash index**

```text
normalized value -> list of target IDs
```

Example:

```text
"jain industries" -> [S2-17, S2-81, S2-904]
```

Querying an anchor's exact normalized/core name returns the stored list immediately.

2. **Token postings / inverted index**

```text
token -> target IDs containing the token
```

Example:

```text
"jain"       -> [S2-17, S2-81, S2-400, ...]
"industries" -> [many IDs]
```

Store document frequency so rare tokens receive more weight and enormous common-token postings can be ignored or capped.

3. **BM25 index**

This is a weighted inverted token index. A multi-token S1 query returns the top-K S2/S3 documents based on rare/shared tokens and document-length normalization. It is useful when only part of a name/address overlaps.

4. **Character n-gram index**

Convert each name/address into overlapping 3–5-character fragments and usually TF-IDF-weight them. Search by cosine similarity to retrieve strings with typos, punctuation changes, joined words, or small spelling differences.

5. **ANN index**

ANN means approximate nearest neighbors. A compliant offline model converts each target name/address into a vector. An ANN data structure rapidly returns the target vectors nearest to the S1 query vector without checking every vector exactly. This is the embedding retrieval route.

6. **Numeric/component index**

```text
(country, postal_code)                  -> target IDs
(country, house_or_plot_number, city)   -> target IDs
(country, rare_locality_token, number)  -> target IDs
```

This supports script-independent address retrieval but must use compound keys/caps because a number alone is not unique.

7. **Transliteration index**

Romanize Indic target names offline, then place those romanized strings into exact, BM25, and/or character n-gram indexes. The underlying target ID is preserved.

### 30.3 Complete pipeline with inputs, operations, and outputs

#### Stage 0: Model compliance

Input: proposed pretrained models and supporting assets.

Operation: verify each model independently is MIT/Apache-2.0, at most 8B parameters, offline, and fine-tuned only on supplied data. Pin checkpoint/tokenizer/license/checksum.

Output: approved local-model registry. No data matching occurs here.

#### Stage 1: Data ingestion and audit

Input: train/test S1, S2, S3 TSV files and training ground truth.

Operation: parse tabs correctly; verify IDs, source prefixes, country values, missing fields, duplicates, positive-label ownership, and record counts.

Output: clean immutable raw tables plus an audit report. Raw text is never overwritten by normalization.

#### Stage 2: Grouped validation split

Input: training anchors, targets, and ground truth.

Operation: divide S1 anchors into train/validation folds; all true S2/S3 matches of one S1 stay with that S1. Stratify important regimes.

Output: fold membership for every S1 anchor. This prevents variants of one entity from leaking between training and validation.

#### Stage 3: Multi-view representation

Input: raw S1/S2/S3 names and addresses.

Operation: create, without deleting raw fields:

- normalized full and core names;
- token sets, sorted tokens, character n-grams, acronym/domain forms;
- canonical address tokens, locality/postal/numeric components;
- script labels and transliterated forms;
- optional compliant offline embeddings.

Output: several representations per record. This still does not create anchor-target pairs.

#### Stage 4: Target-index construction

Input: S2/S3 target representations from Stage 3.

Operation: build hash maps, token postings, BM25, character n-gram, ANN/vector, numeric, and transliteration indexes, normally partitioned by country/source.

Output: searchable data structures mapping an S1 query representation to target IDs. No pair is accepted as a match.

#### Stage 5: Initial S1-to-target candidate retrieval

Input: each S1 anchor representation plus the target indexes.

Operation: query S2 and S3 independently through every enabled view:

```text
S1 name -> S2 name indexes
S1 name -> S3 name indexes
S1 address -> S2 address/numeric indexes
S1 address -> S3 address/numeric indexes
S1 embedding -> S2/S3 ANN indexes
```

Output: multiple candidate sets such as `C_exact_name`, `C_BM25_name`, and `C_numeric_address`, each containing `(S1 ID, target ID, rank, score)`.

#### Stage 6: Candidate union, deduplication, and cheap reduction

Input: all candidate sets produced for one S1.

Operation: union rather than intersect them; deduplicate identical pairs; merge route/rank/score provenance; apply per-view K/minimum scores and a safe final cap. Protect high-precision exact/rare/numeric hits.

Output: `C_final(S1)`, the manageable pair set that receives expensive scoring. Candidate recall must be measured here.

#### Stage 7: Rich pair-feature computation

Input: each `(S1, candidate target)` pair in `C_final`.

Operation: compute exact/core/token/character/edit/transliteration/embedding similarities; address containment; rare-token and numeric agreement; explicit contradictions; missingness; source/script; retrieval route/rank.

Output: one feature row per candidate pair.

#### Stage 8: First-pass pair scoring

Input: feature rows and training labels.

Operation: a GBDT and optionally a compliant offline reranker estimate whether each candidate target describes the S1 entity. Mine hard negatives from plausible but false candidates.

Output: a raw match score/probability for every candidate pair. This score is not yet the final assignment.

#### Stage 8b: Optional target-to-target/sibling expansion

Input: only very high-confidence first-pass anchor-target pairs.

Operation: use a high-confidence target row as an additional query against conservative S2/S3 target indexes, seeking variants that the original S1 query missed. Add newly retrieved `(S1, target)` candidates, compute their Stage 7 features, and run Stage 8 scoring.

Output: a small expanded scored-candidate pool. Expansion is optional and must be validated through ablation.

#### Stage 9: Calibration

Input: out-of-fold raw pair scores and labels.

Operation: convert scores into comparable probabilities, possibly by S2/S3 and important regimes. Correct for artificially sampled hard-negative prevalence.

Output: calibrated pair probabilities/scores.

#### Stage 10: Global decoding and anchor-level selection

Input: calibrated candidate scores for all anchors.

Operation:

- allow an S1 to receive several targets;
- allow each target to be assigned to at most one S1;
- compare competing anchors using absolute score and score margin;
- choose zero/one/multiple accepted targets per S1;
- abstain on weak/ambiguous links because F0.5 is precision-oriented.

Output: final predicted target-ID list for every S1.

#### Stage 11: Validation and ablation

Input: validation predictions, validation ground truth, and intermediate candidates.

Operation: compute candidate recall and exact macro entity-level F0.5; report country/source/script/missingness/count slices; remove each view to measure its unique value.

Output: selected views, K values, thresholds, model, calibration, and decoder configuration.

#### Stage 12: Full training and test inference

Input: chosen configuration plus all provided training data and test S1/S2/S3.

Operation: refit selected trainable components on all permitted training data; rebuild full target indexes; run Stages 3–10 for test anchors.

Output: test candidate pairs, scores, and final assignments.

#### Stage 13: Output construction and validation

Input: final assignments and exact final candidate pairs.

Operation: write required TSV schemas, ensure every S1 appears, ensure every predicted match appears in the candidate file, remove duplicates, and run the supplied validator.

Output: valid `matching_results.tsv` and `candidate_pairs.tsv`.

## 31. Current scope decision: defer SLM/model-based transliteration

The user explicitly decided to remove SLM/model-based transliteration conversion from the current pipeline and revisit it later.

Current Phase 1 behavior:

- Do not run a generative or sequence-to-sequence model to romanize/translate target names or addresses.
- Do not make candidate recall depend on model-generated transliterations.
- Do not include a transliteration model in the active model registry or inference graph.
- Cross-script candidate retrieval initially relies on compliant multilingual embeddings, numeric/address evidence, rare surviving Latin address tokens, and conservative target-to-target expansion.
- Any deterministic rule/table transliteration is also nonessential for Phase 1 and should be left disabled unless separately approved, so the first ablation baseline is unambiguous.
- Later, model-based transliteration can be added as an isolated candidate view and measured by unique recall, candidate volume, precision impact, latency, and licensing compliance.

Earlier sections that describe model-assisted transliteration should therefore be read as deferred design options, not active Phase 1 components.

## 32. Role-aware address-number extraction

### 32.1 Why one regex is insufficient

An address can contain many numbers with different meanings:

```text
Flat 4B, Plot 27, 3rd Floor, Sector 18, Road 5, PIN 560001
```

The numbers mean unit, plot, floor, sector, road, and postal code respectively. A regex can locate candidate spans such as `4B`, `27`, `3`, `18`, `5`, and `560001`, but it cannot reliably determine the role without context.

The extractor must be a staged parser:

```text
normalize
  -> segment/tokenize with offsets
  -> extract numeric/alphanumeric spans
  -> inspect contextual cue words and format
  -> assign a role plus confidence
  -> retain unclassified numbers rather than forcing a role
```

### 32.2 Normalization without destroying structure

Before extracting:

- Unicode-normalize text and convert non-ASCII decimal digits to ASCII equivalents.
- Casefold cue words.
- Normalize spacing around `-`, `/`, and `#`, but preserve those separators inside identifiers.
- Normalize common cue variants, for example `h.no`, `h no`, and `house no`, without removing their token positions.
- Preserve comma/semicolon segments and original character offsets.
- Preserve alphanumeric forms such as `5A1`, `B-78/1`, `G-3/571`, `12A`, and `110001`.

Do not apply a general punctuation remover before number parsing; it would turn `B-78/1` into ambiguous disconnected numbers.

### 32.3 Candidate-span extraction

Use regex only to propose spans, including:

- integers: `27`;
- alphanumeric identifiers: `4B`, `5A1`, `B12`;
- hyphen/slash compounds: `B-78/1`, `G-3/571`, `12/3A`;
- ordinal/floor forms: `3rd`, `2nd`;
- hash-number forms: `#74`;
- postal-looking fixed-length digit sequences.

Keep raw form and canonical forms. For example, `B 78/1`, `B-78/1`, and `B78/1` may share a compact comparison form while retaining the original text.

### 32.4 Contextual role assignment

Assign roles using tokens within the same address segment and a small window around the numeric span.

High-confidence cue families:

| Role | Example cues | Example |
| --- | --- | --- |
| House/door | `house`, `h no`, `hno`, `door`, `premises`, `bungalow` | `H.NO 13` |
| Plot | `plot`, `plt` | `Plot No B-78/1` |
| Flat/unit | `flat`, `unit`, `apt`, `apartment`, `suite`, `shop`, `office`, `room` | `Flat 4B` |
| Floor | `floor`, `fl`, ordinal adjacent to floor | `3rd Floor` |
| Sector/block/phase | `sector`, `sec`, `block`, `phase` | `Sector 18` |
| Survey/khasra | `survey`, `s no`, `sy no`, `khasra`, `kh no` | `KH NO 570/13` |
| Road/highway | `road`, `rd`, `street`, `highway`, `nh`, route cues | `Road 5` |
| Postal | `pin`, `pincode`, `postal`, `zip` plus country-valid format | `PIN 560001` |

Use cue precedence and distance. `Plot No B-78/1` is a clear plot identifier. A bare leading `302` in `302, Andheri West` is ambiguous and should be stored as `leading_property_or_unit` with lower confidence, not asserted to be a house number.

### 32.5 Country-aware postal recognition

Use the supplied country as context. Prefer an explicit postal cue. A bare fixed-length value can be marked as a postal candidate only when its format and address position are compatible with that country.

Examples of common formats useful as starting rules:

- India: six digits, often following `PIN`/`pincode`.
- US: five digits or five-plus-four form, often near the end.
- France: five digits, often before a locality.

A format match alone is not proof. A six-digit company registration, phone fragment, or survey number must not be promoted to postal code merely because it has six digits.

Negative contexts for postal/house classification include `phone`, `mobile`, `tel`, `fax`, registration/tax identifiers, years/dates, and explicit survey/sector/phase cues.

### 32.6 Multiple roles and ambiguity

One address can legitimately have several typed numbers:

```text
43 Sukh Samruddhi, Bungalow No 5, S.No. 5
```

Possible parse:

```text
43 -> ambiguous leading property/building number, medium confidence
5  -> bungalow/house number, high confidence
5  -> survey number, high confidence
```

Do not collapse these to the set `{43,5}` and assume every `5` has the same meaning.

Store a structured representation such as:

```text
house_numbers
plot_numbers
unit_numbers
floor_numbers
sector_numbers
survey_numbers
road_numbers
postal_codes
ambiguous_property_numbers
other_numbers
```

Every extracted item should include raw span, canonical value, role, confidence, cue, segment index, and character offsets.

### 32.7 Using extracted numbers for matching

Number evidence must be role-aware:

- Exact high-confidence same-role equality is positive evidence.
- Plot `B-78/1` matching plot `B78/1` is strong.
- House `13` matching sector `13` is not a same-role match.
- Different floor/unit numbers do not necessarily contradict entity identity if the business occupies multiple units or the source is corrupted.
- Conflicting high-confidence postal codes are strong negative evidence.
- Conflicting high-confidence plot/house identifiers can be negative evidence when locality also matches and both parses are reliable.
- Shared unclassified numbers are weak positive evidence.
- Different unclassified numbers are not negative evidence.

For candidate generation, avoid indexing a number alone. Prefer compound keys:

```text
(country, postal_code)
(country, plot_number, locality_token)
(country, house_number, rare_street_or_locality_token)
(country, survey_number, district_or_state)
```

Unit and floor numbers are generally better as pair features than primary blocking keys because many records omit or alter them.

### 32.8 Extraction implementation order

Phase 1 should use deterministic high-precision rules:

1. segment/tokenize addresses;
2. identify explicitly cued postal, plot, house, unit, floor, sector, survey, and road identifiers;
3. identify country-format postal candidates with lower confidence;
4. keep remaining spans in ambiguous/other buckets;
5. use only high-confidence compound keys for candidate retrieval;
6. expose all typed and ambiguous comparisons as matcher features.

Only after measuring errors should a learned token classifier be considered. If added, it must be trained/fine-tuned only on provided-data-derived annotations and compared with the deterministic parser.

### 32.9 Required validation for the parser

Create a manually inspected audit sample stratified by country, source, cue family, and confidence. Measure:

- extraction precision per role;
- fraction of addresses with each role;
- positive-pair same-role agreement;
- apparent contradiction rate among true pairs;
- false-candidate reduction and candidate recall from each compound key;
- performance when address is partial, mixed-script, or contains `null`.

The true-pair contradiction audit is crucial. If many true pairs disagree on a role, that role must not be used as a hard rejection rule.

## 33. Simplified Phase 1 numeric-signature design

The user proposed a simpler alternative to role-classifying house/plot/unit/floor numbers: extract all number-bearing address components, remove postal code into its own field, concatenate the remaining components into a numeric signature, and retrieve/compare records with similar signatures.

This is approved as the preferred Phase 1 baseline. It avoids pretending that a regex can reliably distinguish house, plot, unit, survey, sector, and road roles. Role-aware parsing remains a possible later enhancement.

### 33.1 Extraction output

For every address, produce:

```text
postal_code_candidates
numeric_tokens_in_order
numeric_multiset_signature
numeric_set
numeric_token_count
```

Example:

```text
Flat 4B, Plot 27, 3rd Floor, Sector 18, Road 5, PIN 560001
```

becomes approximately:

```text
postal_code          = 560001
ordered tokens       = [4B, 27, 3, 18, 5]
ordered signature    = 4B|27|3|18|5
sorted multiset      = 3|4B|5|18|27
set                  = {3, 4B, 5, 18, 27}
```

Postal code is removed from the general numeric signature after extraction so it is not counted twice.

### 33.2 Number-bearing token extraction

Extract every address token containing at least one decimal digit while preserving meaningful internal letters and separators:

```text
13
4B
5A1
B-78/1
G-3/571
12/3A
#74
3rd
```

Canonicalization should:

- uppercase/casefold letters consistently;
- convert Unicode decimal digits to ASCII;
- remove superficial spaces and optional `#`/`No` decoration;
- preserve internal `/` and `-` for the raw canonical view;
- also generate a compact comparison form, for example `B-78/1` -> `B78/1`;
- normalize ordinal suffixes where appropriate, for example `3rd` -> `3` while retaining the raw token;
- preserve token order and duplicate values.

Do not reduce the address to one unstructured digit string such as `4B273185`; boundaries are necessary to distinguish `[4B,27,3,18,5]` from other segmentations.

### 33.3 Postal-code extraction

Postal codes get a separate field because they are both geographically meaningful and relatively structured. Prefer explicit cues such as `PIN`, `pincode`, `ZIP`, and `postal code`. A bare country-compatible fixed-length value near the locality/end of the address can be a lower-confidence postal candidate.

Maintain:

```text
postal_code_value
postal_code_confidence
postal_code_cue
```

Do not assume every six-digit India-address number is a PIN or every five-digit US/France-address number is postal. Survey, registration, phone, route, and account values can share those lengths. When ambiguous, keep the value in the general numeric signature as well or mark the postal extraction provisional rather than deleting it irreversibly.

### 33.4 Excluding non-address identifiers

Use contextual exclusions or separate buckets for clearly labelled:

- phone/mobile/telephone/fax numbers;
- GST/CIN/tax/registration/account identifiers;
- dates and years when explicitly marked;
- latitude/longitude or measurement values if present.

If classification is uncertain, retain the token with low confidence. False deletion can hurt recall more than leaving a weak token for the model.

### 33.5 Multiple signature views

Do not rely only on one concatenated string. Keep at least:

1. **Ordered signature**: preserves address order and duplicates.
2. **Sorted multiset signature**: ignores address reordering but preserves duplicate counts.
3. **Set representation**: supports overlap/Jaccard.
4. **Token-count and rare-token features**.
5. **Postal field**, separately.

The ordered and sorted exact signatures are useful hash-index keys. Set/multiset similarity supports partial and corrupted addresses.

### 33.6 Retrieval and bucketing

Safe retrieval routes include:

```text
exact postal code + at least one rare address/locality/name token
exact ordered numeric signature, when it contains at least two useful tokens
exact sorted multiset signature, when it contains at least two useful tokens
shared rare alphanumeric identifier such as B78/1
high numeric-set overlap plus compatible locality/name evidence
```

Avoid creating a candidate bucket from a single common number such as `1`, `2`, `10`, or `12`. Cap buckets by frequency and use the rarest numeric/alphanumeric token first.

### 33.7 Pair features

For every retrieved pair, compute:

- exact postal-code agreement and confident contradiction;
- exact ordered-signature equality;
- exact sorted-multiset equality;
- numeric token-set Jaccard;
- overlap count and overlap coefficient;
- multiset intersection size;
- ordered longest-common-subsequence ratio;
- rarest shared numeric-token frequency;
- number of anchor/target numeric tokens;
- anchor-only and target-only token counts;
- whether one signature is contained in the other.

Use numeric mismatch cautiously. Addresses are often partial, so different or missing signatures are generally not hard negative evidence. A confident postal-code contradiction is more reliable than a generic numeric-signature mismatch.

### 33.8 Expected advantages and limitations

Advantages:

- simple and deterministic;
- script-independent;
- avoids unreliable semantic role assignment;
- preserves compound identifiers such as `B-78/1`;
- handles addresses that contain several numbers;
- easy to hash, index, and diagnose.

Limitations:

- the same signature can describe unrelated addresses;
- reordered/partial addresses require multiset/overlap views;
- one changed unit/floor number can lower similarity even for a true pair;
- phone/registration/postal misclassification can contaminate signatures;
- number-only matching cannot establish business identity and must be combined with name, locality, textual address, or embedding evidence.

### 33.9 Validation experiment

On grouped validation, compare:

```text
no numeric route
exact ordered-signature route
exact ordered + sorted routes
exact + numeric-overlap route
numeric routes + separate postal features
```

For each, measure unique positive links recovered, candidate volume, collision-bucket sizes, cross-script recall, and final macro F0.5. Inspect the most frequent signatures and false-positive buckets before enabling them at full scale.

## 34. Final Phase 1 candidate, index, and end-to-end plan

Date: 2026-09-25. Status: final proposed implementation plan, not yet implemented or validated.

This section is the operative Phase 1 specification and supersedes earlier proposed architectures wherever they conflict. Preserve the earlier sections as design history. In particular:

- no SLM/model-based or deterministic transliteration route is active in Phase 1;
- simplified numeric signatures plus a separate postal field supersede role-specific house/plot/unit parsing for the initial implementation;
- direct S1-to-S2/S3 retrieval is mandatory;
- target-to-target/sibling expansion is optional and occurs only after a validated direct baseline;
- multilingual embeddings are the initial cross-script name-retrieval route, subject to the per-model license/size/offline rules.

### 34.1 Phase 1 record representations

For every S1, S2, and S3 row, retain raw fields and create the following derived fields.

Name representations:

```text
name_norm_full
name_norm_accent_folded
name_tokens_full
name_tokens_core
name_core_ordered
name_core_sorted_multiset
name_compact_alnum
name_domain_stem
name_acronym
name_char_ngrams
name_script_bucket
name_embedding                  # compliant offline multilingual model
```

Address representations:

```text
address_norm_full
address_tokens_full
address_tokens_useful
address_sorted_multiset
address_char_ngrams
address_numeric_tokens_ordered
address_numeric_signature_ordered
address_numeric_signature_sorted
address_numeric_set
postal_code_value
postal_code_confidence
postal_code_cue
address_embedding               # optional after name-embedding baseline
```

Do not overwrite raw text. Treat empty strings and literal `null` as missing/noisy fields in derived views.

### 34.2 Target-index layout

Build target indexes over S2 and S3. Partition by observed country and source after confirming the complete training labels contain no cross-country positives. Country labels remain open-set strings, so the same index-building code must handle France and future countries.

For every `(country, source)` partition, build:

Exact/hash indexes:

```text
name_norm_full                    -> target IDs
name_core_ordered                 -> target IDs
name_core_sorted_multiset         -> target IDs
name_domain_stem                  -> target IDs
address_norm_full                 -> target IDs
address_numeric_signature_ordered -> target IDs
address_numeric_signature_sorted  -> target IDs
postal_code_value                 -> target IDs plus confidence metadata
```

Inverted postings with document frequencies:

```text
core name token   -> target IDs
useful address token -> target IDs
numeric/alphanumeric address token -> target IDs
```

Fuzzy lexical indexes:

```text
BM25 over core/full name tokens
BM25 over useful address tokens
character 3–5-gram TF-IDF/cosine index for names
character 3–5-gram TF-IDF/cosine index for addresses
```

Embedding index:

```text
ANN index over multilingual name embeddings
```

An address-embedding ANN index is a later ablation, not required for the first runnable system.

Large exact/posting buckets must store frequency and support a cap/rerank path. A common token, postal code, or single common number must not produce an unbounded candidate set.

### 34.3 Direct candidate routes for each S1 anchor

Query S2 and S3 separately within the anchor's supplied country. Every route returns `(anchor_id, target_id, route, rank, retrieval_score, source)`.

Mandatory high-precision/exact routes:

```text
C_name_full_exact
C_name_core_exact
C_name_sorted_exact
C_domain_exact
C_address_exact
C_numeric_ordered_exact
C_numeric_sorted_exact
```

Use numeric exact-signature routes only when the signature has at least two useful tokens or includes a sufficiently rare compound alphanumeric token. Postal equality alone is not an unrestricted route.

Mandatory token routes:

```text
C_rare_name_token
C_rare_address_token
C_rare_numeric_token
```

Rank postings by token IDF and combined supporting evidence. Ignore or tightly cap ubiquitous postings.

Mandatory fuzzy routes:

```text
C_name_BM25_topK
C_name_char_ngram_topK
C_address_BM25_topK
C_address_char_ngram_topK
C_numeric_overlap_topK
```

For numeric overlap, use rare numeric-token postings to create a small pool, then rank by set overlap, overlap coefficient, multiset intersection, containment, and ordered-subsequence similarity. Do not compare every numeric set with every target.

Mandatory cross-script/model route:

```text
C_name_embedding_ANN_topK
```

The embedding route uses a compliant offline multilingual model directly on original names. It does not generate a translated or transliterated string.

Compound postal route:

```text
C_postal_supported
```

Require postal agreement plus at least one other signal such as a rare/core name token, useful address token, numeric-signature overlap, or sufficiently high name fuzzy/embedding score. A postal code by itself is a feature/bucket aid, not identity proof.

Deferred routes:

```text
all transliteration routes
address-embedding ANN
target-to-target/sibling expansion
cross-encoder retrieval/reranking
role-specific house/plot/unit extraction
```

Each deferred route may be added later only as an ablation after the direct baseline is measured.

### 34.4 Candidate union and provenance

For anchor `a`:

```text
C_raw(a) = union of every enabled direct candidate route
```

Deduplicate `(anchor_id, target_id)` pairs while retaining:

```text
all retrieving routes
rank within every route
raw and normalized retrieval scores
query variant used
target source
bucket/posting frequency
rare-token frequencies
postal/numeric extraction confidence
```

A pair appearing in several independent routes remains one candidate pair but carries multi-route evidence.

### 34.5 Initial retrieval sweeps

The following are starting sweep grids, not validated final settings:

```text
name BM25 K:             10, 20, 50
name character K:        10, 20, 50
address BM25 K:           5, 10, 20
address character K:      5, 10, 20
name embedding K:        10, 20, 50
numeric-overlap K:        5, 10, 20
raw union cap/source:    50, 100, 200
```

Measure after the union and any cap:

- positive-pair candidate recall;
- non-singleton anchors with at least one truth retrieved;
- anchors with their complete truth set retrieved;
- unique recall contribution of every route;
- mean, median, P95, and P99 candidates per anchor;
- collision-bucket sizes, runtime, memory, and disk volume;
- slice recall for S2/S3, US/India, missing address, script mismatch, and truth-set size.

Choose the smallest settings near the recall/cost elbow. Do not select K only from final model accuracy because lost true candidates can never be recovered downstream.

### 34.6 Cheap candidate reduction

If `C_raw` exceeds the selected budget, run an inexpensive pre-ranker using:

```text
exact-route flags
rare-token IDF evidence
BM25 and character retrieval scores/ranks
embedding cosine/rank
postal agreement/confidence
numeric exact/overlap/containment scores
basic missingness and length ratios
```

Protect strong exact-core, exact-address, rare compound-numeric, and multiply supported candidates. The post-reduction set is `C_final` and is the exact set written to `candidate_pairs.tsv` and passed to the expensive matching model.

### 34.7 Grouped validation and leakage control

Split by S1 anchor. Keep every positive S2/S3 target of an anchor with that anchor's fold. Stratify by country, total true links, S2/S3 composition, singleton, missing target address, and script mismatch.

Indexes may contain the target records required to simulate the real retrieval pool, but labels for validation anchors must not be used to learn thresholds, hard-negative selection, embedding fine-tuning, reranking, calibration, or decoding. Any learned model is trained on training-fold labels only. Report results using untouched validation-anchor labels.

### 34.8 Rich pair features

For every pair in `C_final`, compute:

Name features:

```text
exact full/core/sorted equality
token Jaccard and containment
IDF-weighted shared tokens
character n-gram cosine/Dice
edit and Jaro-Winkler similarities
accent-folded similarity
compact-alphanumeric and domain similarity
acronym compatibility
multilingual embedding cosine
name lengths and script buckets
```

Address features:

```text
exact normalized equality
token Jaccard and directional containment
IDF-weighted shared useful tokens
character n-gram similarity
missing/null indicators
```

Numeric/postal features:

```text
exact ordered numeric signature
exact sorted numeric signature
numeric set Jaccard
numeric overlap coefficient
multiset intersection
ordered longest-common-subsequence ratio
signature containment
rarest shared numeric-token frequency
anchor-only/target-only numeric counts
postal equality and confident contradiction
postal extraction confidences
```

Retrieval/context features:

```text
source S2/S3
country as open-set-compatible metadata
all route flags
route ranks and scores
number of independent supporting routes
bucket/posting sizes
best and second-best competing-anchor evidence when available
```

Generic numeric mismatches are neutral or weak; they are not hard rejection rules. Confident postal contradiction is stronger negative evidence.

### 34.9 Training examples and hard negatives

Use all available positive links or an anchor-balanced positive sample if memory requires it. Generate negatives from the real train-fold candidate generator, not mainly random pairs.

Priority negative families:

```text
top fuzzy-name nonmatches
exact/core-name collisions
rare-token collisions
address-near but name-different records
numeric-signature/postal bucket collisions
embedding-near semantic impostors
missing-address name impostors
short/generic-name collisions
targets known positive for a competing S1 anchor
```

Cap negatives per anchor/family, balance country and S2/S3, and iteratively add high-scoring model false positives. Cross-anchor positives retrieved for the wrong anchor are particularly reliable negatives because exact training analysis found no target assigned to two anchors.

### 34.10 Pair matcher

The first production matcher is a gradient-boosted decision-tree model over the structured pair features. Use source as a feature and compare:

```text
one joint S2/S3 model with source feature
separate S2 and S3 models
joint model plus source-specific calibration
```

Select using grouped validation macro F0.5, not only AUC. A compliant offline cross-encoder is deferred until the GBDT baseline and error analysis justify its cost.

### 34.11 Calibration and decoding

Generate out-of-fold pair scores and calibrate them, at minimum evaluating source-specific calibration. Correct for the artificial negative-sampling distribution.

Final decoding must:

- allow zero, one, or several targets for each S1;
- assign each S2/S3 target to at most one S1;
- consider calibrated absolute probability and best-versus-second-best anchor margin;
- allow targets to remain unassigned;
- choose anchor-level accepted-link counts/thresholds for validation macro F0.5;
- apply precision-oriented abstention, especially for short names, embedding-only candidates, and missing-address cases.

Do not impose the observed maximum of five S2/six S3 links as a hard test constraint; use count distributions only as soft diagnostics/priors.

### 34.12 Optional second-pass expansion

After the direct baseline is validated, test a tightly controlled experiment:

1. take only very high-confidence direct anchor-target links;
2. use the strong target as a query against conservative target name/address/numeric indexes;
3. add a small number of previously unseen targets to the original anchor;
4. compute full pair features and score normally;
5. enforce target ownership and compare with the no-expansion baseline.

Keep the expansion only if it adds unique true-pair recall and improves macro F0.5 without unacceptable false propagation. It is not part of the minimum viable pipeline.

### 34.13 End-to-end execution order

```text
0. verify model/license/offline compliance
1. ingest TSVs and run integrity audit
2. create grouped anchor validation folds
3. build multi-view representations and numeric/postal signatures
4. build per-country/per-source S2/S3 target indexes
5. query every S1 through all direct candidate routes
6. union, deduplicate, preserve provenance, and cheaply cap candidates
7. measure post-cap candidate recall and volume
8. create rich pair features
9. train GBDT using production-blocker hard negatives
10. iteratively mine high-scoring false positives and retrain
11. create out-of-fold scores and calibrate
12. globally decode target ownership and anchor match sets
13. evaluate exact grouped macro F0.5 and all required slices
14. sweep/ablate candidate routes, K values, features, models, and thresholds
15. refit chosen components on all provided training data
16. build test target indexes and generate `C_final` for every test S1
17. score, calibrate, and globally decode test candidates
18. write `candidate_pairs.tsv` from exact `C_final`
19. write one `matching_results.tsv` row for every test S1
20. run `utils/validate_submission.py` and preserve versioned results/configuration
```

### 34.14 Implementation milestones

Milestone A — deterministic blocker and measurement:

- normalization and numeric/postal extraction with unit tests;
- exact/hash and rare-token/posting indexes;
- grouped candidate-recall evaluator;
- exact macro F0.5 scorer.

Milestone B — fuzzy blocker:

- name/address BM25;
- name/address character n-gram retrieval;
- K/cap sweeps and route ablation.

Milestone C — cross-script retrieval:

- compliant multilingual name embeddings;
- ANN index;
- embedding hard-negative analysis and cross-script recall slice.

Milestone D — learned matcher:

- pair-feature table;
- GBDT;
- realistic and iterative hard-negative mining;
- out-of-fold calibration.

Milestone E — final decisions and packaging:

- target exclusivity/global decoding;
- singleton and link-count decision;
- optional sibling expansion ablation;
- test inference, both output files, and validator run.

## 35. Steps 1–6 implementation status

Date: 2026-09-25. Status: implemented and unit-tested; full grouped folds generated; a deterministic real-data index shard built; full ten-million-target index not yet built.

### 35.1 Implementation resource and package layout

The implementation now lives under the required submission-oriented directory:

```text
code/business_entity_resolution/
├── IMPLEMENTATION_PLAN.md
├── README.md
├── requirements.txt
├── src/business_entity_resolution/
│   ├── __init__.py
│   ├── cli.py
│   ├── folds.py
│   ├── indexing.py
│   ├── io_utils.py
│   ├── metrics.py
│   └── normalization.py
└── tests/
    ├── test_folds.py
    ├── test_indexing.py
    ├── test_metrics.py
    └── test_normalization.py
```

`IMPLEMENTATION_PLAN.md` records the scope, acceptance criteria, commands, and full-data execution order for finalized Phase 1 steps 1–6. `requirements.txt` currently has no third-party requirements; these steps use Python 3.9+ standard-library modules, including `csv`, `sqlite3`, `unicodedata`, `hashlib`, and `tempfile`.

### 35.2 Exact macro entity-level F0.5 scorer

Implemented in `metrics.py` with CLI command:

```bash
PYTHONPATH=src python3 -m business_entity_resolution.cli score \
  --truth PATH_TO_TRUTH.tsv \
  --predictions PATH_TO_PREDICTIONS.tsv
```

Properties:

- explicit tab-delimited parsing;
- rejects duplicate S1 rows and duplicate IDs within one match list;
- strict equality between truth and prediction anchor sets;
- exact per-anchor set precision/recall and F-beta;
- true-empty/predicted-empty singleton score 1.0;
- true-empty/predicted-nonempty singleton score 0.0;
- macro average across all truth anchors;
- temporary SQLite tables join truth and predictions on disk, avoiding simultaneous in-memory dictionaries containing millions of anchors and target IDs;
- JSON result includes beta, macro score, anchor count, singleton count, correctly empty singletons, and anchors with at least one true positive.

The README example with two true IDs and three predicted IDs, two correct, is unit-tested to produce `5/7 = 0.7142857...`.

### 35.3 Deterministic grouped folds

Implemented in `folds.py` with CLI command `make-folds`.

Method:

1. Stream `train_source1.tsv` into a temporary SQLite mapping from S1 ID to country.
2. Stream `train_ground_truth.tsv` in 800-row batches and join to country by indexed S1 ID. The two source files are not assumed to have the same row order.
3. Derive match count, S2 count, S3 count, composition (`none`, `s2_only`, `s3_only`, `both`), and singleton flag.
4. Define the stratum as `country|match_count|source_composition`.
5. Compute fold as the first eight bytes of BLAKE2b over `seed`, stratum, and S1 ID, reduced modulo the fold count.
6. Use a SQLite primary-key table to verify every ground-truth S1 ID occurs once.

Full-data command executed:

```bash
cd code/business_entity_resolution
PYTHONPATH=src python3 -m business_entity_resolution.cli make-folds \
  --source1 ../../dataset/train/train_source1.tsv \
  --ground-truth ../../dataset/train/train_ground_truth.tsv \
  --output artifacts/folds/train_folds.tsv \
  --folds 5 \
  --seed amazon-ml-2026-v1
```

Exact output results over all 2,206,821 training anchors:

| Fold | Anchors |
| ---: | ---: |
| 0 | 442,084 |
| 1 | 440,941 |
| 2 | 440,312 |
| 3 | 442,384 |
| 4 | 441,100 |

The output has 2,206,822 physical lines including the header and is approximately 81 MiB. It is stored at `code/business_entity_resolution/artifacts/folds/train_folds.tsv`.

Country counts by fold:

| Fold | India | US |
| ---: | ---: | ---: |
| 0 | 177,049 | 265,035 |
| 1 | 175,903 | 265,038 |
| 2 | 176,701 | 263,611 |
| 3 | 177,160 | 265,224 |
| 4 | 176,375 | 264,725 |

Composition counts by fold:

| Fold | Both | None | S2 only | S3 only |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 355,371 | 24,713 | 28,974 | 33,026 |
| 1 | 354,859 | 24,608 | 28,633 | 32,841 |
| 2 | 354,582 | 24,631 | 28,251 | 32,848 |
| 3 | 355,998 | 24,776 | 28,754 | 32,856 |
| 4 | 355,237 | 24,519 | 28,417 | 32,927 |

These are exact measurements from the generated fold file. Hash stratification produces approximate, not perfectly equal, fold sizes.

### 35.4 Production normalization and numeric/postal extraction

Implemented in `normalization.py`.

Name views:

- normalized full name;
- accent-folded name;
- full and legal-token-stripped core tokens;
- ordered and sorted core keys;
- compact alphanumeric view;
- domain stem and acronym;
- character 3–5-grams.

Address views:

- normalized full address with literal `null` removed from the derived normalized field;
- full and useful tokens;
- sorted useful-token multiset;
- character n-grams;
- ordered numeric tokens;
- ordered and sorted numeric signatures;
- numeric set;
- postal value, confidence, cue, and explicit/fallback indicator.

Numeric extraction preserves compound identifiers and converts Unicode decimal digits to ASCII. Unit tests cover `4B`, `5A1`, `B-78/1`, `G-3/571`, ordinals, and Devanagari digits.

Explicitly cued postal codes are removed from the general numeric signature. Bare country-format values near the end are lower-confidence candidates and remain available when ambiguity exists. Phone/tax/registration/account contexts are excluded from general address-number extraction. Survey/sector/phase contexts prevent postal classification but the numbers remain in the general numeric signature, consistent with the simplified all-address-numbers design.

### 35.5 Production Unicode-tokenization correction

A deterministic real-data shard exposed a problem with Python regex word tokenization: `[^\W_]`-style tokenization can split Devanagari combining marks from their base characters and fragment words. The earlier EDA normalization percentages remain defined by and valid for the explicit basic EDA regex; they have not been recomputed.

Production normalization was corrected to use a Unicode-category tokenizer that retains combining-mark categories with the current letter/number token. Regression example:

```text
राम मार्केटिंग प्राइवेट लिमिटेड
```

now remains four intact normalized words rather than consonant fragments. Compact alphanumeric views also retain combining marks. A regression test protects this behavior.

### 35.6 SQLite exact and rare-token indexes

Implemented in `indexing.py` with CLI command `build-index`.

Exact posting schema partitions on `(country, source, view, key, entity_id)` and covers:

- full name;
- core name;
- sorted core name;
- domain stem;
- full normalized address;
- ordered/sorted numeric signatures when there are at least two numeric tokens or one compound alphanumeric identifier;
- postal code only at confidence at least 0.9.

Token postings partition on `(country, source, field, token, entity_id)` for core-name, useful-address, and numeric fields. A materialized `token_stats` table stores document frequency by country/source/field/token. Rare-token lookup filters by maximum document frequency and ranks by rarest supporting token, then number of supporting tokens.

The index builder streams target TSVs, batches postings, uses disk-backed SQLite temporary operations for document-frequency aggregation, refuses to silently reuse an existing database unless `--overwrite` is supplied, and supports `--max-rows-per-file` for deterministic prefix shards.

### 35.7 Deterministic real-data index shard

Executed against the first 10,000 physical data rows of each training target file, for 20,000 total target records:

```bash
PYTHONPATH=src python3 -m business_entity_resolution.cli build-index \
  --database artifacts/indexes/train_targets_10k_each.sqlite \
  --targets ../../dataset/train/train_source2.tsv ../../dataset/train/train_source3.tsv \
  --max-rows-per-file 10000 \
  --overwrite
```

This is a deterministic prefix shard, not a random or representative sample.

Shard results after the Unicode-tokenization correction:

| Quantity | Count |
| --- | ---: |
| Target rows | 20,000 |
| Exact postings | 95,642 |
| Token postings | 212,323 |
| Token-stat rows | 70,899 |

The SQLite file is approximately 20 MiB. A direct India/S2 exact full-name query for `राम मार्केटिंग प्राइवेट लिमिटेड` returned exactly `S2-166376419`, verifying intact Devanagari indexing and country/source isolation on real data.

Do not extrapolate the 20 MiB shard linearly as a reliable full-index size estimate because token vocabulary, SQLite B-tree fill, and posting-frequency distributions change with scale. The full 10,320,219-target training index has not been built.

### 35.8 Verification results

The final command used an isolated bytecode cache because the managed macOS environment denied Python's default user-cache write location:

```bash
PYTHONPYCACHEPREFIX=/tmp/ber_pycache PYTHONPATH=src \
python3 -m compileall -q src tests

PYTHONPYCACHEPREFIX=/tmp/ber_pycache PYTHONPATH=src \
python3 -m unittest discover -s tests -v
```

Final result: 15 tests passed in the latest run. Covered behaviors include:

- grouped F0.5 and singleton handling;
- disk-backed file scoring with out-of-order anchors;
- deterministic fold hashing and out-of-order S1/ground-truth joining;
- full/core name normalization;
- intact Devanagari combining marks;
- explicit postal removal and survey/postal disambiguation;
- compound identifiers and Unicode digits;
- literal `null` handling;
- SQLite exact lookup, rare-token ranking, and country/source isolation.

CLI help was also executed successfully for `score`, `make-folds`, and `build-index`.

### 35.9 Full-scale training target indexes

Built 2026-09-25. Both indexes use `--skip-stats` during posting insertion, followed by a separate `materialize-stats` step using WAL journal mode for safety.

| Index file | Source | Target rows | Token-stats rows | File size |
| --- | --- | ---: | ---: | ---: |
| `artifacts/indexes/train_s2_full.sqlite` | S2 | 4,978,266 (98.9% of S2) | 1,935,170 | ~4.0 GB |
| `artifacts/indexes/train_s3_full.sqlite` | S3 | 5,285,603 (100% of S3) | 1,983,173 | ~4.1 GB |

S2 index recovered 98.9% of records after a database corruption incident (build killed during token_stats GROUP BY with `journal_mode=MEMORY` and `synchronous=OFF`). Recovery used `sqlite3 .recover`. The missing 1.1% (~56K records) are from the unrecovered portion.

S3 index was rebuilt from scratch with `--resume` after a similar corruption and recovered 100% of records.

**Lesson learned**: Always use `--skip-stats` for large builds and run `materialize-stats` separately with WAL journaling. Never use `journal_mode=MEMORY` for long-running writes that cannot be cheaply restarted.

### 35.10 BM25 and numeric retrieval routes

Implemented 2026-09-25 in `indexing.py` and `recall.py`.

**BM25 candidates** (`TargetIndex.bm25_candidates`): Uses existing `token_postings` and `token_stats` tables. Computes IDF from stored document frequencies, skips tokens with df > `max_token_df`, and accumulates scores per entity. Critical performance finding: `max_token_df` must be set to ~5,000 (not the initial 500,000) to avoid scanning posting lists with hundreds of thousands of entries per query.

**Numeric overlap candidates** (`TargetIndex.numeric_overlap_candidates`): Finds targets sharing at least 2 address numbers plus a locality token with the anchor. Also includes postal code exact lookup.

CLI flags added: `--enable-bm25`, `--bm25-name-limit`, `--bm25-address-limit`, `--bm25-max-token-df`, `--enable-numeric`, `--numeric-limit`.

### 35.11 Character n-gram indexes

Implemented 2026-09-25 in `indexing.py`. Two-pass approach:
1. Pass 1: count document frequencies for all n-grams across targets.
2. Pass 2: only store postings for n-grams with df <= `max_ngram_df` (default 1000).

This limits disk usage by excluding common n-grams that provide little discriminative value. CLI command: `build-ngram-index`.

| Index | Eligible n-grams | N-gram postings | Build time |
| --- | ---: | ---: | --- |
| S2 (train_s2_full.sqlite) | 4,003,877 | 62,733,129 | ~16 min |
| S3 (train_s3_full.sqlite) | 4,105,398 | 65,067,029 | ~20 min |

Built 2026-09-25 with `--max-ngram-df 1000 --ngram-min 4 --ngram-max 5`.

### 35.12 Blocking recall evaluation — full-scale results

Evaluated 2026-09-25 on fold 0, sample_modulo=500 (913 non-singleton anchors), using both S2 and S3 full indexes with exact + rare + BM25 + numeric routes.

Configuration: `max_document_frequency=5000`, `rare_route_limit=50`, `exact_bucket_cap=500`, `bm25_name_limit=20`, `bm25_address_limit=10`, `bm25_max_token_df=5000`, `numeric_limit=50`.

| Metric | Value |
| --- | ---: |
| Truth pairs | 3,175 |
| Retrieved truth pairs | 2,951 |
| **Pair recall** | **92.9%** |
| Non-singleton anchors | 853 |
| **Any-hit anchor recall** | **98.9%** |
| **Complete-set anchor recall** | **80.7%** |
| Singleton anchors | 60 |
| Mean candidates/anchor | 272 |
| Median candidates/anchor | 246 |
| P95 candidates/anchor | 438 |
| P99 candidates/anchor | 854 |
| Maximum candidates/anchor | 1,108 |

**Route unique positive hits** (pairs found ONLY by that route):

| Route | Unique hits |
| --- | ---: |
| bm25_address | 31 |
| rare_address | 31 |
| rare_numeric | 30 |
| rare_name | 16 |
| exact_name_sorted | 8 |
| exact_address | 2 |
| bm25_name | 1 |
| numeric_overlap | 1 |

**Country-level pair recall**:
- US: 93.5% (1,703/1,822)
- India: 92.2% (1,248/1,353)

**Source-level pair recall**:
- S2: 93.2% (1,430/1,534)
- S3: 92.7% (1,521/1,641)

**Comparison with earlier partial baseline** (superseded):
The earlier partial-index baseline (S2 98.9% + S3 67%, exact+rare only, same 913 anchors) showed 64.3% pair recall, 93.9% any-hit. The full indexes plus BM25+numeric routes raised pair recall by +28.6pp to 92.9% while mean candidates only increased from 264 to 272.

### 35.12b Blocking recall with all 5 routes (including character n-gram)

Evaluated 2026-09-25 on the same fold 0 / sample_modulo=500 / 913 anchors. All routes enabled: exact + rare + BM25 + numeric + character n-gram.

Configuration: same as 35.12 plus `--enable-ngram --ngram-name-limit 20`.

| Metric | Exact+Rare (partial) | + BM25+Numeric | + N-gram (all 5) |
| --- | ---: | ---: | ---: |
| **Pair recall** | 64.3% | 92.9% | **94.6%** |
| **Any-hit anchor recall** | 93.9% | 98.9% | **99.2%** |
| **Complete-set anchor recall** | 23.1% | 80.7% | **85.0%** |
| Mean candidates/anchor | 264 | 272 | 288 |
| Median candidates/anchor | — | 246 | 260 |
| P95 candidates/anchor | — | 438 | 459 |
| Maximum candidates/anchor | — | 1,108 | 1,108 |

**Route unique positive hits** (all 5 routes):

| Route | Unique hits |
| --- | ---: |
| **ngram_name** | **53** |
| rare_numeric | 27 |
| bm25_address | 26 |
| rare_address | 19 |
| exact_name_sorted | 8 |
| rare_name | 6 |
| exact_address | 1 |
| bm25_name | 1 |
| numeric_overlap | 1 |

The n-gram route contributed the most unique positive hits of any single route (53), demonstrating its value for fuzzy name matching that other routes miss.

**Country-level pair recall (all 5 routes)**:
- US: 95.4% (1,739/1,822)
- India: 93.5% (1,265/1,353)

**Source-level pair recall (all 5 routes)**:
- S2: 95.0% (1,457/1,534)
- S3: 94.3% (1,547/1,641)

### 35.13 Current limitations

- S2 index is 98.9% complete (missing ~56K records from corruption recovery). Impact on recall is likely small but should be noted.
- Cross-script/transliteration retrieval not implemented. India recall (93.5%) is lower than US (95.4%), partly due to Devanagari targets that share no Latin character overlap.
- No embedding-based retrieval yet (would help cross-script and semantic matching).
- The fold generator uses approximate deterministic hash stratification rather than an exactly balanced allocation within each stratum.
- Postal fallback rules are conservative heuristics and need a manual/error audit on true pairs before being treated as contradiction evidence.
- The remaining ~5.4% missed pairs likely include cross-script matches, very short/generic names, and records with empty addresses — these may need transliteration, sibling graph, or embedding retrieval.

### 35.14 Pairwise feature extraction (2026-09-26)

New module `features.py` with 31 pairwise features for (anchor, target) pair scoring:

**Name features (14):** name_exact_full, name_exact_core, name_exact_sorted, name_token_jaccard, name_core_token_jaccard, name_char_trigram_dice, name_char_4gram_dice, name_edit_ratio, name_length_ratio, name_shared_core_count, name_core_token_count_anchor, name_core_token_count_target, name_acronym_match.

**Address features (8):** addr_exact_full, addr_token_jaccard, addr_useful_token_jaccard, addr_char_trigram_dice, addr_length_ratio, addr_shared_useful_count, addr_useful_token_count_anchor, addr_useful_token_count_target.

**Numeric/postal features (5):** num_shared_count, num_anchor_count, num_target_count, num_jaccard, postal_match, postal_either_present.

**Metadata features (4):** target_addr_missing, is_s2, script_mismatch, route_count.

Edit distance uses full O(nm) Levenshtein on core-ordered name strings. Character n-gram features use trigrams (separate from the 4-5 gram index n-grams). Script mismatch detects Latin vs Devanagari disagreement.

### 35.15 Matching model pipeline (2026-09-26)

New module `matching.py` — candidate pair generation + HistGradientBoostingClassifier training.

**Pair generation workflow:**
1. Load fold anchors from `train_folds.tsv` (filtered by fold number and optional sample modulo)
2. Load S1 anchor records (name, address) by streaming `train_source1.tsv`
3. Load ground truth match sets from `train_ground_truth.tsv`
4. For each anchor: run 5-route blocker against S2+S3 SQLite indexes → candidate set
5. Batch-fetch candidate target records from `raw_records` table in SQLite (NOT from a 4GB in-memory dict — see 35.16)
6. Label: positive if target_id in ground truth, negative otherwise
7. Sample up to 20 hard negatives per anchor (ranked by route_count — highest first, i.e., hardest to distinguish from true matches)
8. Extract 31 pairwise features, write to TSV

**Model training:** HistGradientBoostingClassifier (scikit-learn 1.4.1), 500 iterations, max_depth=6, min_samples_leaf=50. Supports train/val split via separate pair files.

**Smoke test result (fold 0, sample modulo 10000 = 49 anchors):**
- Train AUC: 0.9995, precision: 0.9936, recall: 0.9451, F1: 0.9688

**Preliminary model v1 (fold 0, sample modulo 500 = 913 anchors):**
- 21,264 pairs (3,004 positive, 18,260 negative)
- Train AUC: 1.0, precision: 0.999, recall: 0.996, F1: 0.998
- Note: train-only metrics, no validation yet. Near-perfect train AUC expected — model can separate matches from hard negatives on seen data.

CLI commands: `generate-training-pairs`, `train-matcher`.

### 35.16 SQLite raw_records optimization (2026-09-26)

**Problem:** Original pair generation loaded all 10.3M target records into a Python dict (~4GB RAM). With multiple concurrent processes on 16GB machine, memory pressure caused SQLite page cache thrashing → 18s/anchor average.

**Fix:** Added `raw_records` table to existing SQLite indexes (entity_id, business_name, business_address, country). Target text is now fetched via `get_records_batch()` — batched `SELECT ... WHERE entity_id IN (...)` query, ~0.3ms per batch of ~20 records.

**Populate:** `populate-raw-records` CLI command. Streams TSV, inserts in 50K batches. S2: 5,034,616 rows in ~90s. S3: 5,285,603 rows in ~90s.

**Benchmark (76 anchors, no contention):**

| Component | Time |
|---|---:|
| Blocker (5 routes × 2 sources) | 780ms |
| SQLite batch lookup (~20 records) | 0.3ms |
| Feature extraction (~20 pairs) | 10ms |
| **Total per anchor** | **791ms** |

**Before vs after:**

| Metric | Before (4GB dict) | After (SQLite) |
|---|---|---|
| Startup time | 50s | 0s |
| RAM per process | ~4GB | ~17MB |
| Per-anchor time (no contention) | ~2s | ~0.8s |
| Per-anchor time (3 concurrent) | ~18s (thrashing) | ~0.8s (no thrashing) |
| 900 anchors (1 fold sample) | ~4.5 hours | ~12 minutes |

**Key insight:** The blocker itself is fast (~800ms). The old slowness was entirely caused by OS memory pressure from the 4GB dict evicting SQLite page cache. With raw_records in SQLite, the OS file cache is shared across processes.

### 35.17 Training pair generation status (2026-09-26)

Currently running: all 5 folds in parallel (sample modulo 500, ~880-913 anchors per fold).

| Fold | Purpose | Anchors (approx) |
|---|---|---|
| 0 | Training | 884 |
| 1 | Training | 882 |
| 2 | Training | 881 |
| 3 | Training | 885 |
| 4 | Validation | 882 |

Plan: concatenate folds 0-3 for training (~3,500 anchors, ~12K pos, ~70K neg), fold 4 for held-out validation (~3K pos, ~18K neg). Train HistGradientBoostingClassifier, evaluate pairwise metrics + entity-level macro F0.5.

### 35.18 Next steps

1. Wait for fold pair generation to complete (~20-25 min with 5 concurrent processes)
2. Concatenate folds 0-3 into training set, fold 4 as validation
3. Train matcher v2 with train/val split
4. Evaluate entity-level macro F0.5 on fold 4 (the competition metric)
5. Threshold tuning — optimize for F0.5 (precision-weighted)
6. Global decoder — enforce target-to-anchor exclusivity
7. Source-specific calibration (S2 vs S3 corruption patterns differ)
8. Cross-script/transliteration retrieval for Devanagari targets
9. France zero-shot transfer
