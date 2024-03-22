# Fourier Neural Operator for Solving PDEs

This repository is created as the the final project for [Machine Learning MA060018](http://files.skoltech.ru/data/edu/syllabuses/2023/MA060018.pdf?v=f4hls9) course. It is mainly based on the previous work described in this repository:

- [Fourier Neural Operator](https://github.com/khassibi/fourier-neural-operator/blob/main/README.md)

We have trained a model with FNO algorithm to sove 1D generalized classical Burger's equation . 

$$
    \partial_t u + \alpha u^\delta \partial_x u = \partial_x^2 u + \beta u \left( 1 - u^\delta \right),
$$

where $\alpha \neq 0$, $\beta$ and $\delta > 0$.

- The training data were generated with the file [gen-burges-fisher-gen.ipynb](gen-burgers-fisher-gen.ipynb)
    - [Generated datasets](https://kaggle.com/datasets/14b1e5b86bc461380277887164c2b9e88716c56510b9ded8d064731f68c4fba5)
- The mode was trained with this file [fno1d-notebook.ipynb](fno1d-notebook.ipynb)
- Learned parameters was extracted into the files
    - [4 FNO layers with 1001 points mesh](results%20fno%201d/fno-4layers-1001.pth)
    - [4 FNO layers with 2001 points mesh](results%20fno%201d/fno-4layers-2001.pth)
    - [4 FNO layers with 251 points mesh](results%20fno%201d/fno-4layers-251.pth)
    - [4 FNO layers with 5001 points mesh](results%20fno%201d/fno-4layers-5001.pth)

- Testing and data analysis
    - [Animation of the training and testing](results%20fno%201d/animation.npy)
    - [L2 error for 1001 points mesh](results%20fno%201d/l2-1001.png), etc.
    - [MSE error for 1001 points mesh](results%20fno%201d/mse-1001.png), etc.
    - [Testing animation](results%20fno%201d/nn.gif)



perform experiments on Burgers' equation, Darcy flow, and the Navier-Stokes equation (including the turbulent regime). 
Our Fourier neural operator shows state-of-the-art performance compared to existing neural network methodologies and it is up to three orders of magnitude faster compared to traditional PDE solvers.

It follows from the previous works:
- [(GKN) Neural Operator: Graph Kernel Network for Partial Differential Equations](https://arxiv.org/abs/2003.03485)
- [(MGKN) Multipole Graph Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2006.09535)


## Requirements
- We have updated the files to support [PyTorch 1.8.0](https://pytorch.org/). 
Pytorch 1.8.0 starts to support complex numbers and it has a new implementation of FFT. 
As a result the code is about 30% faster.
- Previous version for [PyTorch 1.6.0](https://pytorch.org/) is avaiable at `FNO-torch.1.6`.

## Files
The code is in the form of simple scripts. Each script shall be stand-alone and directly runnable.

- `fourier_1d.py` is the Fourier Neural Operator for 1D problem such as the (time-independent) Burgers equation discussed in Section 5.1 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
- `fourier_2d.py` is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
- `fourier_2d_time.py` is the Fourier Neural Operator for 2D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf), 
which uses a recurrent structure to propagates in time.
- `fourier_3d.py` is the Fourier Neural Operator for 3D problem such as the Navier-Stokes equation discussed in Section 5.3 in the [paper](https://arxiv.org/pdf/2010.08895.pdf),
which takes the 2D spatial + 1D temporal equation directly as a 3D problem
- The lowrank methods are similar. These scripts are the Lowrank neural operators for the corresponding settings.
- `data_generation` are the conventional solvers we used to generate the datasets for the Burgers equation, Darcy flow, and Navier-Stokes equation.

## Datasets
We provide the Burgers equation, Darcy flow, and Navier-Stokes equation datasets we used in the paper. 
The data generation configuration can be found in the paper.
- [PDE datasets](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-?usp=sharing)

The datasets are given in the form of matlab file. They can be loaded with the scripts provided in utilities.py. 
Each data file is loaded as a tensor. The first index is the samples; the rest of indices are the discretization.
For example, 
- `Burgers_R10.mat` contains the dataset for the Burgers equation. It is of the shape [1000, 8192], 
meaning it has 1000 training samples on a grid of 8192.
- `NavierStokes_V1e-3_N5000_T50.mat` contains the dataset for the 2D Navier-Stokes equation. It is of the shape [5000, 64, 64, 50], 
meaning it has 5000 training samples on a grid of (64, 64) with 50 time steps.

We also provide the data generation scripts at `data_generation`.

## Models
Here are the pre-trained models. It can be evaluated using _eval.py_ or _super_resolution.py_.
- [models](https://drive.google.com/drive/folders/1swLA6yKR1f3PKdYSKhLqK4zfNjS9pt_U?usp=sharing)

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
```
