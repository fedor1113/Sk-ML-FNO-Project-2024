# Fourier Neural Operator for Solving PDEs

This repository is created as the the final project for [Machine Learning MA060018](http://files.skoltech.ru/data/edu/syllabuses/2023/MA060018.pdf?v=f4hls9) course. It is mainly based on the previous work described in this repository:

- [Fourier Neural Operator](https://github.com/khassibi/fourier-neural-operator/blob/main/README.md)

We have trained a model with FNO algorithm to solve 1D generalized Burgers-Fisher equation. 

$$
    \partial_t u + \alpha u^\delta \partial_x u = \partial_x^2 u + \beta u \left( 1 - u^\delta \right),
$$

where $\alpha \neq 0$, $\beta$ and $\delta > 0$.

- The training data were generated with the file [gen-burges-fisher-gen.ipynb](gen-burgers-fisher-gen.ipynb)
    - [Generated Burgers-Fisher datasets](https://kaggle.com/datasets/14b1e5b86bc461380277887164c2b9e88716c56510b9ded8d064731f68c4fba5)
    - [Skoltech picture dataset](https://kaggle.com/datasets/6228b6270a9f707b18f4a2b04fe085787577780a9b48c122a3b501af4c6e9b11)
- The mode was trained with this file [fno1d-notebook.ipynb](fno1d-notebook.ipynb)
- Learned parameters was extracted into the files
    - [4 FNO layers with 1001 points mesh](results%20fno%201d/fno-4layers-1001.pth)
    - [4 FNO layers with 2001 points mesh](results%20fno%201d/fno-4layers-2001.pth)
    - [4 FNO layers with 251 points mesh](results%20fno%201d/fno-4layers-251.pth)
    - [4 FNO layers with 5001 points mesh](results%20fno%201d/fno-4layers-5001.pth)

- Testing and data analysis
    - [L2 error for 1001 points mesh](results%20fno%201d/l2-1001.png), etc.
    - [MSE error for 1001 points mesh](results%20fno%201d/mse-1001.png), etc.
    - [Testing animation](results%20fno%201d/nn.gif)


We have trained a model to solve 2D heat equation with FNO algorithm.

$$
    \partial_t u = k_c \left( \partial_x^2 u + \partial_y^2 u \right)
$$

- The model was trained with this file [fno2d.ipynb](fno2d.ipynb)
- [Result](results%20fno%201d/panic%20(1).gif) 

## Citations

```
@misc{li2020fourier,
      title={Fourier Neural Operator for Parametric Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2010.08895},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@misc{li2020neural,
      title={Neural Operator: Graph Kernel Network for Partial Differential Equations}, 
      author={Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar},
      year={2020},
      eprint={2003.03485},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}


@article{Brunton2015,
	author = {Steven L. Brunton and Joshua L. Proctor and J. Nathan Kutz},
	journal = {Proceedings of the National Academy of Sciences},
	pages = {3932--3937},
	title = {Discovering governing equations from data by sparse identification of nonlinear dynamical systems},
	url = {https://api.semanticscholar.org/CorpusID:1594001},
	volume = {113},
	year = {2015}
}

@article{Forootani2023ARS,
	author = {Ali Forootani and Pawan Goyal and Peter Benner},
	journal = {ArXiv},
	title = {A Robust SINDy Approach by Combining Neural Networks and an Integral Form},
	url = {https://api.semanticscholar.org/CorpusID:261823341},
	volume = {abs/2309.07193},
	year = {2023}
}

@article{Brunton2013CompressiveSA,
	author = {Steven L. Brunton and Jonathan H. Tu and Ido Bright and J. Nathan Kutz},
	journal = {SIAM J. Appl. Dyn. Syst.},
	pages = {1716--1732},
	title = {Compressive Sensing and Low-Rank Libraries for Classification of Bifurcation Regimes in Nonlinear Dynamical Systems},
	url = {https://api.semanticscholar.org/CorpusID:15950036},
	volume = {13},
	year = {2013}
}

@article{Kiser2023ExactIO,
	author = {Shawn L. Kiser and Mikhail Alexandrovich Guskov and Marc R'ebillat and Nicolas Ranc},
	journal = {ArXiv},
	title = {Exact identification of nonlinear dynamical systems by Trimmed Lasso},
	url = {https://api.semanticscholar.org/CorpusID:260438377},
	volume = {abs/2308.01891},
	year = {2023}
}

@article{Li2020FourierNO,
	author = {Zong-Yi Li and Nikola B. Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew M. Stuart and Anima Anandkumar},
	journal = {ArXiv},
	title = {Fourier Neural Operator for Parametric Partial Differential Equations},
	url = {https://api.semanticscholar.org/CorpusID:224705257},
	volume = {abs/2010.08895},
	year = {2020}
}

@article{LuLu2021DeepOnet,
	author = {Lu Lu and Pengzhan Jin and Guofei Pang and Zhongqiang Zhang and George Karniadakis},
	doi = {10.1038/s42256-021-00302-5},
	journal = {Nature Machine Intelligence},
	month = {03},
	pages = {218--229},
	title = {Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators},
	volume = {3},
	year = {2021}
}

@article{Marvian2014ExtendingNT,
	author = {Iman Marvian and Robert W. Spekkens},
	journal = {Nature Communications},
	title = {Extending Noether{\rq}s theorem by quantifying the asymmetry of quantum states},
	url = {https://api.semanticscholar.org/CorpusID:17343800},
	volume = {5},
	year = {2014}
}

@article{Raissi2017PhysicsID,
	author = {Maziar Raissi and Paris Perdikaris and George Em Karniadakis},
	journal = {ArXiv},
	title = {Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
	url = {https://api.semanticscholar.org/CorpusID:394392},
	volume = {abs/1711.10561},
	year = {2017}
}

@article{HORNIK1989359,
	abstract = {This paper rigorously establishes that standard multilayer feedforward networks with as few as one hidden layer using arbitrary squashing functions are capable of approximating any Borel measurable function from one finite dimensional space to another to any desired degree of accuracy, provided sufficiently many hidden units are available. In this sense, multilayer feedforward networks are a class of universal approximators.},
	author = {Kurt Hornik and Maxwell Stinchcombe and Halbert White},
	doi = {10.1016/0893-6080(89)90020-8},
	issn = {0893-6080},
	journal = {Neural Networks},
	keywords = {Feedforward networks, Universal approximation, Mapping networks, Network representation capability, Stone-Weierstrass Theorem, Squashing functions, Sigma-Pi networks, Back-propagation networks},
	number = {5},
	pages = {359--366},
	title = {Multilayer feedforward networks are universal approximators},
	url = {https://www.sciencedirect.com/science/article/pii/0893608089900208; https://www.sciencedirect.com/science/article/pii/0893608089900208},
	volume = {2},
	year = {1989}
}

@inproceedings{Zhao2022IncrementalSL,
	author = {Jiawei Zhao and Robert Joseph George and Zong-Yi Li and Anima Anandkumar},
	booktitle = {NeurIPS 2022 AI for Science Workshop},
	title = {Incremental Spectral Learning in Fourier Neural Operator},
	url = {https://api.semanticscholar.org/CorpusID:256627806},
	year = {2022}
}

@book{murray2002introduction,
	author = {James Dickson Murray},
	publisher = {Springer New York, NY},
	subtitle = {I. An introduction},
	title = {Mathematical Biology},
	url = {https://doi.org/10.1007/b98868},
	year = {2002}
}

@article{Wang_1990,
	abstract = {Exact solitary wave solutions of the generalised Burgers-Huxley equation delta u/ delta t- alpha udelta  delta u/ delta chi - delta 2u/ delta chi 2= beta u(1-udelta )(udelta - gamma ) are obtained by using the relevant nonlinear transformations. The results obtained are the generalisation of former work. The method in this paper can also be applied to the Burgers-Fisher equation.},
	author = {X Y Wang and Z S Zhu and Y K Lu},
	doi = {10.1088/0305-4470/23/3/011},
	journal = {Journal of Physics A: Mathematical and General},
	month = {feb},
	number = {3},
	pages = {271},
	publisher = {},
	title = {Solitary wave solutions of the generalised Burgers-Huxley equation},
	url = {https://dx.doi.org/10.1088/0305-4470/23/3/011},
	volume = {23},
	year = {1990}
}

@article{ISMAIL2004203,
	abstract = {Solving generalized Fisher and Burger--Fisher equations by the finite difference technique yields difficult nonlinear system of equations. In this paper linearization and restrictive Pad{\'e} approximation is considered. It yields more accurate and faster results. Also the stability analysis is discussed. Numerical results are treated.},
	author = {Hassan N.A. Ismail and Aziza A.Abd Rabboh},
	doi = {10.1016/S0096-3003(03)00703-3},
	issn = {0096-3003},
	journal = {Applied Mathematics and Computation},
	number = {1},
	pages = {203--210},
	title = {A restrictive Pad{\'e} approximation for the solution of the generalized Fisher and Burger--Fisher equations},
	url = {https://www.sciencedirect.com/science/article/pii/S0096300303007033},
	volume = {154},
	year = {2004}
}


```
