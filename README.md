# *TI*me series *D*iscover*Y* *BENCH*mark (tidybench)

This repository holds implementations of the following four algorithms for causal structure learning for time series,

* `SLARAC` (Subsampled Linear Auto-Regression Absolute Coefficients),
* `QRBS` (Quantiles of Ridge regressed Bootstrap Samples),
* `LASAR` (LASso Auto-Regression),
* `SELVAR` (Selective auto-regressive model),

which came in first in 18 and close second in 13 out of the 34 competition categories in the [Causality 4 Climate competition](https://causeme.uv.es/neurips2019/) at the Conference on Neural Information Processing Systems 2019 (NeurIPS). For details on the competition tasks and the outcomes you may watch the [recording of the NeurIPS session](https://slideslive.com/38922052/competition-track-day-21) or consult [the result slides](https://causeme.uv.es/neurips2019/static/img/Runge_NeurIPS_compressed.pdf).
(Algorithm names map as follows between `tidybench` and our competition implementations: `tidybench.slarac` was varvar, `tidybench.qrbs` was ridge, `tidybench.lasar` was varvar(lasso=True), and `tidybench.selvar` was selvar.)

More details can be found in our [accompanying paper](http://proceedings.mlr.press/v123/weichwald20a.html) and the respective well-documented code files.

Feel free to use our algorithms (AGPL-3.0 license). In fact, we encourage their use as baseline benchmarks and guidance of future algorithmic and methodological developments for structure learning from time series.

We kindly ask you to cite our [accompanying paper](http://proceedings.mlr.press/v123/weichwald20a.html) in case you find our code useful:
```
@InProceedings{weichwald2020causal,
  title = {{Causal structure learning from time series: Large regression coefficients may predict causal links better in practice than small p-values}},
  author = {Weichwald, Sebastian and Jakobsen, Martin E. and Mogensen, Phillip B. and Petersen, Lasse and Thams, Nikolaj and Varando, Gherardo},
  publisher = {PMLR},
  series = {Proceedings of the NeurIPS 2019 Competition and Demonstration Track, Proceedings of Machine Learning Research},
  volume = {123},
  pages = {27--36},
  year = {2020},
  editor = {Hugo Jair Escalante and Raia Hadsell},
  pdf = {http://proceedings.mlr.press/v123/weichwald20a/weichwald20a.pdf},
  url = {http://proceedings.mlr.press/v123/weichwald20a.html},
}
```



## What you get

**Input**: time series data (and some method-specific parameters)

**Output**: score matrix indicating which structural links are inferred likely to exist

All four algorithms take as input multivariate time series data in form of a T x d matrix of T time samples of d variables and output a d x d score/adjacency matrix A. The (i,j)th entry corresponds to an edge from the i-th to the j-th time series component, where higher values correspond to edges that are inferred to be more likely to exist, given the observed data.


## Example

At the moment, only a [toy example](examples/toy.py) is provided.


## Requirements

`SLARAC`, `QRBS`, and `LASAR` require numpy and sklearn. These requirements are listed in the [requirements.txt](requirements.txt) and can be installed via `pip install -r requirements.txt`.

`SELVAR` requires lapack/blas installed and the compilation of
[selvarF.f](tidybench/selvarF.f) with [f2py](https://docs.scipy.org/doc/numpy/f2py/)
(e.g. `f2py -llapack -c -m selvarF selvarF.f`).

## Who we are

We are a team of PhD students and Postdocs that formed at the [Copenhagen Causality Lab (CoCaLa)](https://math.ku.dk/cocala) of the University of Copenhagen ([Martin E Jakobsen](https://www.math.ku.dk/english/research/spt/cocala/?pure=en/persons/410383), [Phillip B Mogensen](https://www.math.ku.dk/english/staff/?pure=en/persons/467826), [Lasse Petersen](https://www.math.ku.dk/english/research/spt/cocala/?pure=en/persons/433485), [Nikolaj Thams](https://nikolajthams.github.io/), [Gherardo Varando](https://gherardovarando.github.io/), [Sebastian Weichwald](https://sweichwald.de)) to participate in the C4C competition.
