# The **ti**me series **d**iscover**y** **bench**mark (tidybench)

This repository holds implementations of the following four algorithms for causal structure learning for time series,

* `QRBS` (Quantiles of Ridge regressed Bootstrap Samples),
* `SLARAC` (Subsampled Linear Auto-Regression Absolute Coefficients),
* `LASAR` (LASso Auto-Regression),
* `SELVAR` (Selective auto-regressive model),

which came in first in 18 and close second in 13 out of the 34 competition categories in the [Causality 4 Climate competition](https://causeme.uv.es/neurips2019/) at the Conference on Neural Information Processing Systems 2019 (NeurIPS). For details on the competition tasks and the outcomes you may watch the [recording of the NeurIPS session](https://slideslive.com/38922052/competition-track-day-21) or consult [the result slides](https://causeme.uv.es/neurips2019/static/img/Runge_NeurIPS_compressed.pdf).

Feel free to use our algorithms (AGPL-3.0 license). In fact, we encourage their use as baseline benchmarks and guidance of future algorithmic and methodological developments for structure learning from time series.

*Note*: We are currently in the progress of migrating and polishing the algorithms `QRBS`, `SLARAC`, `LASAR`, `SELVAR` from their original versions that we used in the competition to a standalone version in this repository.

## What you get

Input: time series data (and some method-specific parameters)
Output: score matrix indicating which structural links are inferred likely to exist

All four algorithms take as input multivariate time series data in form of a T x d matrix of T time samples of d variables and output a d x d score/adjacency matrix A. The (i,j)th entry corresponds to an edge from the i-th to the j-th time series component, where higher values correspond to edges that are inferred to be more likely to exist, given the observed data.

More details can be found in a soon-to-appear preprint and the respective well-documented code files.


## Example

At the moment, only a [toy example](examples/toy.py) is provided.


## Requirements

`QRBS`, `SLARAC`, and `LASAR` require numpy and sklearn. These requirements are listed in the [requirements.txt](requirements.txt) and can be installed via `pip install -r requirements.txt`.

`SELVAR` requires numpy, scipy, and compilation of [selvarF.f](tidybench/selvarF.f).

## Who we are

We are a team of PhD students and Postdocs that formed at the [Copenhagen Causality Lab (CoCaLa)](https://math.ku.dk/cocala) of the University of Copenhagen ([Martin E Jakobsen](https://www.math.ku.dk/english/research/spt/cocala/?pure=en/persons/410383), [Phillip B Mogensen](https://www.math.ku.dk/english/staff/?pure=en/persons/467826), [Lasse Petersen](https://www.math.ku.dk/english/research/spt/cocala/?pure=en/persons/433485), [Nikolaj Thams](https://nikolajthams.github.io/), [Gherardo Varando](https://gherardovarando.github.io/), [Sebastian Weichwald](https://sweichwald.de)) to participate in the C4C competition.
