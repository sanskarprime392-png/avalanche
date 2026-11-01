# NHESS submission form — text to paste

Fields the upload form asks for that are *not* part of the manuscript file. Having them written
in advance avoids composing them in a web form at submission time.

---

## Short summary

NHESS requires a non-technical summary of **500 characters including spaces**, in paragraph form,
with no lists, and all abbreviations spelled out. It is written for a reader outside the field.

> Maps that predict where avalanches and landslides are likely routinely claim over 90 per cent
> accuracy. These maps are built from records of places where events happened, so researchers must
> invent the "safe" locations themselves. We show that this invented choice, not the quality of the
> model, sets the accuracy score: a model that knows nothing can score a perfect result. We show
> what such a score can and cannot mean, and release a tool that measures the problem.

<!-- 468 characters including spaces. Verify with: python -c "print(len(open('x.txt').read()))" -->

---

## Abstract

Paste the 199-word abstract from the manuscript. NHESS limits research-article abstracts to
**100–200 words**, so do not substitute a longer version.

---

## Before you submit — items that need a real value

| Item | Where | Status |
|---|---|---|
| Supervisor's name | Acknowledgements | `[SUPERVISOR NAME]` placeholder |
| Zenodo DOI for the code | Code and data availability | `[ZENODO DOI]` placeholder |
| Funding | Financial support | states none — confirm this is true |
| Full first names | Title page | required, not initials |
| Corresponding author email | Title page | required |

### Minting the Zenodo DOI

NHESS requires GitHub-hosted code to carry a DOI, because a repository can be force-pushed or
deleted and a DOI cannot. Zenodo issues one from a GitHub release:

1. Sign in to zenodo.org with the GitHub account.
2. Under **GitHub** in the account settings, switch the `avalanche` repository on.
3. On GitHub, draft a release and tag it (for example `v1.0-submission`).
4. Zenodo archives that tag and mints the DOI within a few minutes.

Use the **version** DOI, not the concept DOI, so the citation resolves to the exact code that
produced the reported numbers.

---

## Suggested reviewers

Copernicus asks for suggestions. Reasonable choices are authors of the avalanche-detection and
pseudo-absence literature already cited in the manuscript. Do not suggest anyone who has
co-authored with the author, and do not suggest the authors of the protocol this paper evaluates —
an editor may invite them independently, which is appropriate, but suggesting them is not.
