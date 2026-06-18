import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import procrustes
from scipy.linalg import orthogonal_procrustes, subspace_angles

LATENT_SEED, FEATURE_SEED, NOISE_SEED = 0, 1, 2
SIGMA_ETAS, DS = [0.0, 0.05, 0.1, 0.2], [50, 100]

class synthetic_data_generation:
    def __init__(self, mu=np.array([0., 0., 1.]), kappa=0.5, sigma=0.2, dt=0.01, T=100, d_latent=2, rand_seed=LATENT_SEED):
        self.simulate_ou_on_sphere(mu=mu, kappa=kappa, sigma=sigma, dt=dt, T=T, d_latent=d_latent, rand_seed=rand_seed)

    def simulate_ou_on_sphere(self, mu=np.array([0., 0., 1.]), kappa=0.5, sigma=0.2,
                            dt=0.01, T=100, d_latent=2, rand_seed=LATENT_SEED):
        self.mu = np.asarray(mu, dtype=float)
        self.kappa = kappa
        self.sigma = sigma
        self.dt = dt
        self.T = T
        self.d_latent = d_latent

        rng = np.random.default_rng(rand_seed)
        N = int(T / dt) + 1
        self.Y = np.zeros((N, d_latent))
        stationary_sd = sigma / np.sqrt(2 * kappa)
        self.Y[0] = rng.normal(0.0, stationary_sd, size=d_latent)

        a = np.exp(-kappa * dt)
        ou_std = np.sqrt((sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * dt)))
        for t in range(1, N):
            self.Y[t] = a * self.Y[t - 1] + ou_std * rng.normal(0.0, 1.0, size=d_latent)

        mu = self.mu / np.linalg.norm(self.mu)
        ref = np.array([0.0, 0.0, 1.0]) if abs(mu[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u1 = np.cross(ref, mu)
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(mu, u1)

        Yn = np.linalg.norm(self.Y, axis=1)
        sinc = np.divide(np.sin(Yn), Yn, where=Yn != 0, out=np.ones_like(Yn))
        tangent = self.Y[:, 0][:, None] * u1[None, :] + self.Y[:, 1][:, None] * u2[None, :]
        self.X = np.cos(Yn)[:, None] * mu[None, :] + sinc[:, None] * tangent

        assert Yn.max() < np.pi, "exp_mu not injective: ||Y|| reached pi"
        assert np.allclose(np.linalg.norm(self.X, axis=1), 1.0), "X left S^2"

        return {"X": self.X, "Y": self.Y}

    def plot_trajectory_on_sphere(self):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        ax.plot_surface(
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
            alpha=0.15,
            color='lightblue',
            linewidth=0,
        )
        sc = ax.scatter(self.X[:, 0], self.X[:, 1], self.X[:, 2], c=np.arange(len(self.X)), cmap='viridis', s=8)
        fig.colorbar(sc, ax=ax, label='Time')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(r'Trajectory $X_t$ on $S^2$')
        plt.show()

    def thin_trajectories(self, gap=10, burnin=500):
        self.gap = gap
        self.burnin = burnin
        self.dt_eff = gap * self.dt
        self.X_geom = self.X[burnin::gap]
        self.Y_geom = self.Y[burnin::gap]
        self.n_geom = self.X_geom.shape[0]


class observation_map:
    def __init__(self, X, D=100, feature_seed=FEATURE_SEED, d_manifold=3):
        self.X = np.asarray(X)
        self.D = D
        self.feature_seed = feature_seed
        self.d_manifold = d_manifold
        self.draw_feature_map()
        self.apply_feature_map()
        self.eta = np.random.default_rng(NOISE_SEED).standard_normal(self.Gclean.shape)
        self.Zobs = None

    def draw_feature_map(self):
        rng = np.random.default_rng(self.feature_seed)
        self.W = rng.standard_normal((self.D, self.d_manifold))
        self.b = rng.uniform(0, 2 * np.pi, size=self.D)
        return self.W, self.b

    def apply_feature_map(self):
        self.Gclean = np.cos(self.X @ self.W.T + self.b)
        return self.Gclean

    def add_noise(self, sigma_eta):
        self.sigma_eta = sigma_eta
        self.Zobs = self.Gclean + self.sigma_eta * self.eta
        return self.Zobs

    def clean_state(self):
        return {"Gclean": self.Gclean, "W": self.W, "b": self.b}

    def sample_observation(self, sigma_eta):
        return self.add_noise(sigma_eta=sigma_eta)

    def generate(self, sigma_eta=None):
        if sigma_eta is None:
            return self.clean_state()
        self.sample_observation(sigma_eta=sigma_eta)
        return {"Gclean": self.Gclean, "Zobs": self.Zobs, "W": self.W, "b": self.b}


class embedding:
    def kernel_mass_slope(self, data, npts=70, lo=-1.5, hi=2.0):
        Dsq = squareform(pdist(data)**2)
        Dnd = Dsq.copy(); np.fill_diagonal(Dnd, np.inf)
        d_min = np.median(Dnd.min(1)); d_med = np.median(Dsq[np.triu_indices_from(Dsq, 1)])
        self.grid = np.logspace(np.log10(d_min)+lo, np.log10(d_med)+hi, npts)
        upper = Dsq[np.triu_indices_from(Dsq, 1)]
        self.T = np.array([2*np.exp(-upper/e).sum() + Dsq.shape[0] for e in self.grid])
        self.slope = np.gradient(np.log(self.T), np.log(self.grid))

    def select_epsilon(self, data):
        self.kernel_mass_slope(data)
        i = int(np.argmax(self.slope))
        lo = max(i-5, 0); hi = min(i+5, len(self.grid)-1)
        self.eps = float(np.sqrt(self.grid[lo]*self.grid[hi]))

    def diffusion_map(self, data, eps, alpha=1, k=4):
        Dsq = squareform(pdist(data)**2)
        Wm = np.exp(-Dsq/eps); q = Wm.sum(1)
        Wa = Wm/np.outer(q**alpha, q**alpha)
        da = Wa.sum(1); Dis = 1.0/np.sqrt(da)
        S = Dis[:, None]*Wa*Dis[None, :]                
        self.w, v = np.linalg.eigh(S)
        idx = np.argsort(self.w)[::-1]; self.w, v = self.w[idx], v[:, idx]
        phi = Dis[:, None]*v                             
        self.Psi = phi[:, 1:k+1]*self.w[1:k+1]                     
        
    
    def get_embedding(self, data, alpha=1, k=4):
        self.alpha = alpha; self.k = k
        self.select_epsilon(data)
        self.diffusion_map(data, self.eps, alpha, k)

        return self.w, self.Psi
    
class MLE:
   
    def ou_mle(self, Z, dt):
        Z0, Z1 = Z[:-1], Z[1:]
        M = Z0.size                                       
        a = (Z1*Z0).sum()/(Z0*Z0).sum()
        a = min(max(a, 1e-9), 1 - 1e-12)
        kappa = -np.log(a)/dt                             
        resid = Z1 - a*Z0
        s2_innov = (resid**2).sum()/M
        sigma = np.sqrt(s2_innov*2*kappa/(1 - a**2))      
        I_a = (Z0*Z0).sum()/s2_innov                      
        se_kappa = np.sqrt(1.0/I_a)/(a*dt)                
        return kappa, sigma, a, (kappa - 1.96*se_kappa, kappa + 1.96*se_kappa)


if __name__ == "__main__":
    latent = synthetic_data_generation(rand_seed=LATENT_SEED)
    latent.thin_trajectories(gap=10, burnin=500)

    obs = observation_map(latent.X_geom, D=100, feature_seed=FEATURE_SEED)
    clean = obs.clean_state()

    noise_levels = [0.0, 0.05, 0.1, 0.2]
    embeddings = {}

    for sigma_eta in noise_levels:
        Zobs = obs.sample_observation(sigma_eta)
        dm = embedding()
        w, Psi = dm.get_embedding(Zobs, alpha=1, k=4)
        embeddings[sigma_eta] = {
            "Zobs": Zobs,
            "eigenvalues": w,
            "diffusion_coordinates": Psi,
            "epsilon": dm.eps,
        }

    print("latent X_geom shape:", latent.X_geom.shape)
    print("clean G shape:", clean["Gclean"].shape)
    for sigma_eta, result in embeddings.items():
        print(f"sigma_eta={sigma_eta}: Zobs shape={result['Zobs'].shape}, epsilon={result['epsilon']:.4g}")

