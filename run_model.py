from simulate_model import ToyMarketSimulator
from diffusion import diffusion
from state_space import JDKFStateSpace
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


def main():

    xp = np
    # Generate market data
    sim = ToyMarketSimulator()
    X, Y = sim.X, sim.Y
    Z = xp.concatenate((X, Y), axis=1)

    m, n = X.shape[1], Y.shape[1]
    T = X.shape[0]

    # Estimate the B matrix from the factor model
    B_init = xp.linalg.lstsq(X, Y, rcond=None)[0]

    # Get the diffusion matrix and diffusion coordinate
    # In the factor model, we only use X data to do this
    diff = diffusion(X)
    P, evals, evecs = diff.P, diff.eval, diff.evec

    # Use three eigenbasis
    l = 3
    evals = evals[1:l+1]
    evecs = evecs[:, 1:l+1]


    state = JDKFStateSpace(X, evals, evecs, P=P, epsilon=diff.epsilon)
    Hx, A, Q = state.Hx, state.A, state.Q
    # Using a different method to find Hx
    # We want X ~= Psi @ H.T, so solve argminH||Z- Psi @ H.T||^2
    Hx = xp.linalg.lstsq(evecs, X, rcond=None)[0].T

    # Create Hy = BHx
    Hy = B_init.T @ Hx
    Hz = xp.vstack([Hx, Hy])

    # log likelihood
    k_max = 10
    epsilon = 0.01
    k = 0
    total_loglike = xp.zeros(k_max+1)
    total_loglike[0] = -xp.inf

    while k < k_max:

        # Build residual covariance matrix
        Z_hat = evecs @ Hz.T
        obs_resid = Z_hat - Z
        R = xp.cov(obs_resid.T)
        #R = xp.diag(xp.var(obs_resid, axis=0))


        # ===========================#
        ### Kalman filtering steps ###
        # ===========================#
        # Results storage
        loglike = xp.zeros(T)
        # Where we store psi_t_t and P_t_t
        psi_filt = xp.zeros(shape=evecs.shape)
        P_filt = xp.zeros(shape=(T, l, l))

        # Where we store psi_t_t-1 and P_t_t-1
        # psi_pred[t] = psi_t_t-1
        psi_pred = xp.zeros(shape=evecs.shape)
        P_pred = xp.zeros(shape=(T, l, l))

        psi_tm1_tm1 = evecs[0]
        P_tm1_tm1 = xp.eye(evecs.shape[1]) * 0.1

        psi_filt[0] = psi_tm1_tm1
        P_filt[0] = P_tm1_tm1
        psi_pred[0] = psi_filt[0]
        P_pred[0] = P_filt[0]

        for t in range(1,T):
            # Prediction step
            psi_t_tm1 = A@psi_tm1_tm1
            P_t_tm1 = A@P_tm1_tm1@A.T + Q

            # Innovation
            et = Z[t] - Hz@psi_t_tm1
            St = Hz @ P_t_tm1 @ Hz.T + R

            # Kalman Gain
            Kt = P_t_tm1 @ Hz.T @ xp.linalg.inv(St)

            # Update step
            psi_t_t = psi_t_tm1 + Kt@et
            P_t_t = (xp.eye(len(psi_t_tm1)) - Kt@Hz) @ P_t_tm1

            # Log likelyhood
            _, logdet = xp.linalg.slogdet(St)
            loglike[t] = -0.5 * (logdet + et.T @ xp.linalg.solve(St, et) + (m+n)*xp.log(2*xp.pi))

            # Storing results
            psi_pred[t] = psi_t_tm1
            P_pred[t] = P_t_tm1

            psi_filt[t] = psi_t_t
            P_filt[t] = P_t_t

            # For next iteration
            psi_tm1_tm1 = psi_t_t
            P_tm1_tm1 = P_t_t

        total_loglike[k+1] = xp.sum(loglike)

        if k == 0:
            pass

        elif xp.abs(total_loglike[k+1] - total_loglike[k]) / xp.abs(total_loglike[k]) < epsilon:
            print("Tolerance reached, breaking loop")
            break
        else:
            print("Continuing algorithm, log liklihood tolerance not reached")
            print(total_loglike[k+1])
            print(xp.abs(total_loglike[k+1] - total_loglike[k]) / xp.abs(total_loglike[k]))


        # ================ #
        ### RTS Smoother ###
        # ================ #
        # Initialise variables

        # Place to store results
        psi_smooth = xp.zeros(shape=(T, l))
        P_smooth = xp.zeros(shape=(T, l, l))

        psi_smooth[-1] = psi_filt[-1]
        P_smooth[-1] = P_filt[-1]

        for t in range(T-2, -1, -1):

            # Smoother gain
            Jt = P_filt[t] @ A.T @ xp.linalg.inv(P_pred[t+1])

            # Smoothed state estimate
            psi_t_T = psi_filt[t] + Jt @ (psi_smooth[t+1] - psi_pred[t+1])

            # Smoothed covariance
            P_t_T = P_filt[t] + Jt @ (P_smooth[t+1] - P_pred[t+1]) @ Jt.T

            # Store results
            psi_smooth[t] = psi_t_T
            P_smooth[t] = P_t_T


        # ============================= #
        ### M-Step: Parameter Updates ###
        # ============================= #

        xtH = X - psi_smooth @ Hx.T
        ytH = Y - psi_smooth @ Hy.T

        Rx = 1/T * (xtH.T @ xtH)
        Ry = 1/T * (ytH.T @ ytH)
        Rxy = 1/T * (xtH.T @ ytH)

        R = xp.block([[Rx, Rxy] , [Rxy.T, Ry]])

        # Updating B for factor model
        X_lift = psi_smooth @ Hx.T

        # Not how this works yet. Coudln't figure out
        # how to implemenet the formula directly
        B_new = np.linalg.lstsq(X_lift, Y, rcond=None)[0]
        Hy = B_new.T @ Hx
        Hz = xp.vstack([Hx, Hy])

        k += 1

            
    predictions = psi_filt @ Hz.T
    smoothed = psi_smooth @ Hz.T

    # Persist the key outputs in one place so the run can be resumed or inspected later.
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "run_model_results.npz",
        X=X,
        Y=Y,
        Z=Z,
        evals=evals,
        evecs=evecs,
        Hx=Hx,
        Hy=Hy,
        Hz=Hz,
        A=A,
        Q=Q,
        B_init=B_init,
        B_new=B_new,
        psi_pred=psi_pred,
        P_pred=P_pred,
        psi_filt=psi_filt,
        P_filt=P_filt,
        psi_smooth=psi_smooth,
        P_smooth=P_smooth,
        predictions=predictions,
        smoothed=smoothed,
        total_loglike=total_loglike,
    )

    metadata = {
        "T": int(T),
        "m": int(m),
        "n": int(n),
        "latent_dim": int(l),
        "k_max": int(k_max),
        "epsilon": float(epsilon),
        "final_iteration": int(k),
        "final_loglike": float(total_loglike[k]),
    }

    with (output_dir / "run_model_metadata.json").open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    plt.plot(predictions[:, 0], label="filtered prediction X0")
    plt.plot(smoothed[:, 0], label="smoothed prediction X0")
    plt.plot(X[:, 0], label="true X0")
    plt.legend()
    plt.show()
if __name__ == "__main__":
    main()