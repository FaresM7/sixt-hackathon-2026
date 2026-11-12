# Project 3: Judge the Judge — a recipe for building LLM judges you can trust

**Project owner:** Jigyasa Singhal

**Questions before the hackathon:** email the project owner directly: jigyasa.singhal@sixt.com. On the day, ask on site.

## Problem
AI judges check the replies a customer-support chatbot writes, and their scores decide whether a change to the chatbot
counts as better or worse. But judges miss small changes in meaning:

> **Customer:** "Can I keep the car a few days longer?"
> **A:** "You can usually extend your rental in the app or by phone, **as long as the car is not reserved for another customer**."
> **B:** "You can usually extend your rental in the app or by phone."

Our starting judge says both are fine. But B makes a promise the company can't always keep. Recent research (2026)
shows the same pattern: judges ignore harmless wording changes, which is good, but miss most edits that change the
meaning, which is not. There is no standard recipe for building a judge that does better.

## Your mission
Don't tune one judge. **Build a recipe that makes AI judges better, even ones you haven't seen.**

1. **Find** where the judge fails: which edits, which languages?
2. **Split:** simple code for facts like numbers, the AI for the rest.
3. **Improve** the AI judge: better instructions, examples, smaller questions, voting.
4. **Check** it on cases you didn't tune on.

**The surprise:** in the afternoon you get a new judging task (a new criterion for the same replies and policy, plus a
few labelled examples). Run your recipe on it **unchanged**: only the inputs change, not your code.

## Data
Everything is in [`starter_pack/`](starter_pack/). All of it is made up: no company or customer data.

- **226 test cases** in English, German, French, Spanish and Italian, plus a 119-case subset for quick tests.
  Each case is a good reply and a copy of it with one small change:
  - **harmless** (the judge should accept both): polite opener, filler sentence, reordering, translation;
  - **harmful** (the judge should reject the change): dropped condition, "usually" → "always", a wrong number or policy detail.
- A **made-up company policy** (`policy.md`), the only thing replies are judged against.
- **A starting judge, already run:** its verdicts and report card are included, so you see its mistakes right away.
  It scores 0.70 and catches only 7% of dropped conditions.
- **A starter kit:** a judge template laid out in the four steps (`my_judge.py`), a runner, a scorer, a diagnosis
  tool that shows where a judge fails, and the one-page recipe template (`RECIPE.md`).

Full details (fields, sizes, limitations): [`starter_pack/DATACARD.md`](starter_pack/DATACARD.md).
Your code is also scored on cases you have not seen, including edit types that are not in this data, so solve the idea,
not the list.

## How results are judged
- **Score on the unseen test cases, for both judges** (the starting criterion and the surprise one): bad replies caught
  without rejecting good ones (`score.py` "combined"), compared with the starting judge.
- **Consistency:** your judge is run twice; how often it gives the same verdict (`score.py --repeat`).
- **Agreement** with the labels (Cohen's kappa), per edit type and language.
- **Reusability:** is your recipe clear enough that someone else could follow it?

## Model (required): `claude-haiku-4.5`
Every team uses the same model, so results show the quality of your recipe, not the size of your model. Your final
judge must run on `claude-haiku-4.5`, and your submitted results must be produced with it. We check this by rerunning
a sample of your results ourselves.

## Rules
- Your judge sees **one reply at a time**, never the pair, the edit type or the label, and must not keep state between calls.
- Nothing criterion-specific may be hard-coded: everything must come from the criterion text, the policy and the examples.
- The unseen test cases are not in this repo. Trying to obtain or train on them means disqualification.

## Setup

<img src="../assets/time-quick-setup.svg" alt="Under 5 minutes" width="360">

**Quick setup (under 5 min):** `bash setup.sh` in this folder. It makes a virtual environment, creates `.env` from `.env.example` for your LLM key and scores the starting judge. Python is our starting point; any language is fine.
Do this **before the hackathon day**.

- `git` and **Python 3.11+**, with a virtual environment (`python3 -m venv .venv` or `uv venv`). The starter pack itself
  needs only the standard library; add your own dependencies to a `requirements.txt` in this folder.
- Your own LLM coding agent, so you can work with the tool you are most comfortable with.
- **LLM API access** for the judge: SIXT provides LLM API keys for this task. The judge talks to any
  OpenAI-compatible endpoint, configured with three environment variables: `JUDGE_LLM_BASE_URL`, `JUDGE_LLM_API_KEY`,
  and `JUDGE_LLM_MODEL=claude-haiku-4.5` (see "Model" above). Keep keys in `.env` (git-ignored), never in code.
- Check that the kit runs, no LLM needed:
  ```bash
  cd Project_3/starter_pack
  python3 score.py baseline_dev_verdicts.csv dev_pairs.jsonl
  python3 diagnose.py baseline_dev_verdicts.csv dev_pairs.jsonl
  ```
- Skim the background reading below.

## Submission
**Timeline on the day**
- **About 14:30, surprise task:** you get a new criterion (`CRITERION.md`) and a few labelled examples
  (`dev_pairs.jsonl`). Run your judge on them unchanged: only `--criterion` and `--dev` change.
- **16:00, code freeze:** commit and push your final judge as `starter_pack/final_judge.py`. This commit is your judge;
  later changes to it don't count.
- **16:00, test cases released:** you get `hidden_cases.zip`, the unseen test cases for both criteria, **without labels**.
- **16:00-16:45:** run your frozen judge on both test sets **twice** (for consistency) with `claude-haiku-4.5`, then
  push the result files:

  | File | Run |
  |---|---|
  | `submission/criterion1_run1.csv`, `submission/criterion1_run2.csv` | original criterion |
  | `submission/criterion2_run1.csv`, `submission/criterion2_run2.csv` | surprise criterion |

**We check:** we rerun a random sample of your test cases with your 16:00 code on `claude-haiku-4.5`. Results that don't
match your code are disqualified, so changing the judge after 16:00, using another model or editing results by hand
all show up.

**Budget:** at most **3 LLM calls per reply**.

**By 16:45, also push:**
- `SUBMISSION.md` (see the root `SUBMISSION_TEMPLATE.md`), including your filled-in recipe from `RECIPE.md` and your
  before-and-after numbers on both criteria.
- **Either** a 1-minute demo video (`submission/demo.mp4`, under 50 MB) **or** up to 5 slides (`submission/slides.pdf`).

## Background reading
- *A Judge Should Know What Changed* (arXiv:2608.24419): judges miss meaning changes, especially over-promising
- *Reliability without Validity* (arXiv:2606.19544): agreement and consistency can look good while the judge is still wrong
- *Counterfactual Fairness of Contact-Center QA* (Observe.AI, ACL Findings 2026)
- *Rubric Artifacts in LLM-as-a-Judge* (arXiv:2609.02942)
