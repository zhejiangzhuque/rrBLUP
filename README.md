# rrBLUP

rrBLUP is a R package which used for genomic prediction with the rrBLUP linear mixed model ([Endelman 2011](https://acsess.onlinelibrary.wiley.com/doi/full/10.3835/plantgenome2011.08.0024)). We transform it into functions in Python.

## Requirement

Code for rrBLUP was prepared using Python 3.8.15. The major requirements are listed below, and the full list can be found in `requirement.txt`:

- numpy==1.24.0
- scikit-learn==1.2.0
- pandas==1.5.2
- scipy==1.9.3
- rpy2==3.5.5

We suggest creating a new virtual environment for a clean installation of all the relevant requirements by following commands:

```
conda create -n rrblup python=3.8.15
conda activate rrblup
conda install --yes --file requirement.txt
```

Use jupyter notebook in the new virtual environment by following commands:

```
conda install ipykernel
python -m ipykernel install --user --name rrblup --display-name='rrblup'
```

## Data

We use the genomic dataset with protein as the trait from SoyNAM ([soybase.org/SoyNAM/index.php](https://soybase.org/SoyNAM/index.php)), found in `data` folder. The default coding format in SoyNAM is : **-1** for the missing value, **0** for 0/0 genotype, **1** for 0/1 genotype and **2** for 1/1 genotype. However, rrBLUP R package requires input genotype matrix coded as **-1**,**0**,**1** for 0/0, 0/1, 1/1 genotype respectively. Therefore, we convert the data coding format in `data/data_convertion.ipynb`.

## Usage

There are two functions `A_mat` and `mixed_solve` in `rrBLUP.py`, which are same as the functions `A.mat` and `mixed.solve` in R package rrBLUP. The notebook includes how to use rrBLUP in Python within `demo.ipynb`. Meanwhile we also provide a R script about how to use these functions in R within `demo.R`.  Detail usages about these two functions are shown below.

```python
A_mat(X, min_MAF = None, max_missing = None, impute_method = "mean", tol = 0.02,
      shrink = False, n_qtl = 100, n_iter = 5, return_imputed = False)
    
    '''
    Parameters:
    -----------
    X [array]:
        Matrix of unphased genotypes for n lines and m biallelic markers, coded as {-1,0,1}
        
    min_MAF [float, default = None]:
        Minimum minor allele frequency, default removes monomorphic markers
        
    max_missing [float, default = None]:
        Maximum proportion of missing data, default removes completely missing markers
        
    impute_method [str("mean" or "EM"), default = "mean"]:
        Method of genotype imputation, there are only two options, "mean" imputes with the mean of each marker and
        "EM" imputes with an EM algorithm
        
    tol [float, default = 0.02]:
        Convergence criterion for the EM algorithm
        
    shrink [Union[bool, str("EJ" or "REG")], default = False]:
        Method of shrinkage estimation, default disable shrinkage estimation; If string, there are only two options,
        "EJ" uses EJ algorithm described in Endelman and Jannink (2012) and "REG" uses REG algorithm described in
        Muller et al. (2015); If True, uses EJ algorithm
    
    n_qtl [int, default = 100]:
        Number of simulated QTL for the REG algorithm
    
    n_iter [int, default = 5]:
        Number of iterations for the REG algorithm
    
    return_imputed [bool, default = False]:
        Whether to return the imputed marker matrix
    
    
    Returns:
    --------
    A [array]:
        Additive genomic relationship matrix (n * n)
    
    (When return_imputed = True)
    imputed [array]:
        Imputed X matrix
    '''


mixed_solve(y, Z = None, K = None, X = None, method = "REML",
            bounds = [1e-09, 1e+09], SE = False, return_Hinv = False)
    
    '''
    Parameters:
    -----------
    y [array]:
        Vector of observations for n lines and 1 observation
    
    Z [array, default = None]:
        Design matrix of the random effects for n lines and m random effects, default to be the identity matrix
    
    K [array, default = None]:
        Covariance matrix of the random effects, if not passed, assumed to be the identity matrix
    
    X [array, default = None]:
        Design matrix of the fixed effects for n lines and p fixed effects, which should be full column rank,
        default to be a vector of 1's
    
    method [str("ML" or "REML"), default = "REML"]:
        Method of maximum-likelihood used in algorithm, there are only two options, "ML" uses full maximum-likelihood
        method and "REML" uses restricted maximum-likelihood method
    
    bounds [list, default = [1e-09, 1e+09]]:
        Lower and upper bound for the ridge parameter
    
    SE [bool, default = False]:
        whether to calculate and return standard errors
    
    return.Hinv [bool, default = False]:
        whether to return the inverse of H = Z*K*Z' + \lambda*I, which is useful for GWAS
    
    
    Returns:
    --------
    Vu [float]:
        Estimator for the marker variance \sigma^2_u
    
    Ve [float]:
        Estimator for the residual variance \sigma^2_e
    
    beta [array]:
        BLUE for the fixed effects \beta
    
    u [array]:
        BLUP for the random effects u
    
    LL [float]:
        maximized log-likelihood
    
    (When SE = True)
    beta.SE [float]:
        Standard error for the fixed effects \beta
    
    u.SE [float]:
        Standard error for the random effects u
    
    (When return_Hinv = True)
    Hinv [array]:
        Inverse of H = Z*K*Z' + \lambda*I
    '''
```

## License

Our code is released under Apache License (see LICENSE file for details).

