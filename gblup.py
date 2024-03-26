import pandas as pd
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def run_asreml_G_gblup(G_matrix, blue_df):
    pandas2ri.activate()
    print(f'G_matrix:\n{G_matrix.shape}, blue_df:\n{blue_df.shape}')
    ro.r('''Sys.setlocale("LC_CTYPE", "en_US.UTF-8")''')
    ro.globalenv['G_matrix'] = pandas2ri.py2rpy(G_matrix)
    # Verify by printing the first few rows in R
    # Ensure that the dimnames (row and column names) are correctly set in R
    ro.r('''
    dimnames(G_matrix) <- list(rownames(G_matrix), colnames(G_matrix))
    ''')

    ro.globalenv['blue'] = pandas2ri.py2rpy(blue_df)
    # ro.r('rownames(blue)')
    ro.r('''
    library(asreml)
    library(ASRgenomics)
    library(tidyverse)
    G_matrix <- as.matrix(G_matrix)
    Gmat = G.matrix(G_matrix)$G
    rownames(Gmat) <- rownames(G_matrix)
    colnames(Gmat) <- rownames(G_matrix)
    diag(Gmat) = diag(Gmat) + 0.01
    ginv = G.inverse(Gmat,sparseform = T)
    ginv = ginv$Ginv.sparse

    attr(ginv,"rowNames") %>% head

    blue$Name <- as.factor(blue$Name)


    asr2 <- asreml(Value ~ 1,
                   random = ~ vm(Name,ginv), 
                   residual = ~idv(units),
                   data = blue,workspace = 20480000 * 12)
    summary(asr2)$varcomp

    ''')
    g = ro.r('coef(asr2)$random')

    g_row_names = ro.r('rownames(coef(asr2)$random)')

    g_df = pd.DataFrame(g)

    # Convert the row names to a Python list
    g_row_names_py = list(g_row_names)

    # Assign the row names to the pandas DataFrame index
    g_df.index = g_row_names_py
    g_df['Name'] = ['_'.join(name.split('_')[1:]) for name in g_df.index]
    g_df.columns = ['Value', 'Name']
    g_df = g_df[['Name', 'Value']]
    g_df.to_csv('output/gblup.csv',index=True)
    print(g_df.head(10))
    return g_df

def run_asreml_ped_gblup(ped_df, blue_df):
    # Ensure automatic conversion between pandas dataframes and R data frames
    pandas2ri.activate()

    # Convert pandas dataframes to R data frames and put them into the R environment
    ro.globalenv['ped'] = pandas2ri.py2rpy(ped_df)
    ro.globalenv['blue'] = pandas2ri.py2rpy(blue_df)

    # Run the ASReml model in R
    ro.r('''
    library(asreml)
    ainv <- asreml::ainverse(ped)
    blue$Name <- as.factor(blue$Name)
    blue <- merge(blue, ped, by = "Name")
    asr2 <- asreml::asreml(predicted.value ~ 1,
                           random = ~ vm(Name, ainv),
                           residual = ~ idv(units),
                           data = blue)
    ''')

    # Extract results from the asr2 model in R to Python
    g = ro.r('coef(asr2)$random')

    g_row_names = ro.r('rownames(coef(asr2)$random)')

    g_df = pd.DataFrame(g)

    # Convert the row names to a Python list
    g_row_names_py = list(g_row_names)

    # Assign the row names to the pandas DataFrame index
    g_df.index = g_row_names_py
    g_df['Name'] = [name.split('_')[1] for name in g_df.index]

    g_df.columns = ['Value', 'Name']
    g_df = g_df[['Name', 'Value']]

    return g_df


# Example usage
# ped_df and blue_df should be defined as pandas DataFrames before calling this function
# results_df = run_asreml_and_extract_results(ped_df, blue_df)
# print(results


def solve_mixed_gblup_model(n_obs, obs_ids, y_values, a):
    import numpy as np
    ainv = np.linalg.inv(np.array(a))


    # Fixed effects: Assuming only a mean effect for simplicity
    X = np.ones((n_obs, 1))

    # Random effects design matrix Z based on observation IDs
    Z = np.zeros(
        (n_obs, ainv.shape[0]))  # Z should have a row for each observation and a column for each individual in ainv
    for i, obs_id in enumerate(obs_ids):
        Z[i, obs_id] = 1  # Assuming obs_ids are 0-indexed and match the order in ainv

    # Observations
    y = np.array(y_values).reshape(-1, 1)

    # Matrix operations
    XX = X.T @ X
    XZ = X.T @ Z
    ZX = Z.T @ X
    ZZ = Z.T @ Z
    ZZA = ZZ + ainv * 2  # Adjust based on your specific model
    Xy = X.T @ y
    Zy = Z.T @ y

    # Assemble mixed model equations (MME)
    mme = np.vstack([
        np.hstack([XX, XZ]),
        np.hstack([ZX, ZZA])
    ])

    # Solve the MME for solution vector
    sol = np.linalg.solve(mme, np.vstack([Xy, Zy]))

    return sol