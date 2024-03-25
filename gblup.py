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