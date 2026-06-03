import numpy as np
import matplotlib.pyplot as plt



class ToyMarketSimulator:
    def __init__(self):
        self.m = 5
        self.n = 20

        self.rng = np.random.default_rng(21)

        # Simulation parameters
        self.T = 1000          # number of timesteps
        self.dt = 0.01         # timestep size

        # Scale for latent space noise
        self.latent_noise_scale = 0.01
        # Noise scale for factor model
        self.returns_noise_scale = 0.01

        self.simulate_latent_space()
        self.generate_covariates()
        self.generate_returns()

    def simulate_latent_space(self):
        # OU parameters
        alpha = np.array([0.3, 0.1]) # mean reversion rates
        sigma = np.array([0.2, 0.1]) # noise strengths

        # Latent factors
        self.theta = np.zeros((self.T, 2))

        # Initial condition
        self.theta[0] = np.array([1.0, -1.0])

        # Euler-Maruyama simulation
        for t in range(self.T - 1):
            dW = np.sqrt(self.dt) * self.rng.normal(size=2)

            drift = -alpha * self.theta[t]
            diffusion = sigma * dW

            self.theta[t + 1] = self.theta[t] + drift * self.dt + diffusion

    def generate_covariates(self):
        self.X = np.zeros((self.T, 5))

        self.X[:,0] = np.sin(self.theta[:,0])

        self.X[:,1] = np.cos(2*self.theta[:,1])

        self.X[:,2] = self.theta[:,0] * self.theta[:,1]

        self.X[:,3] = self.theta[:,0]**2 - self.theta[:,1]**2

        self.X[:,4] = np.exp(0.3*self.theta[:,0])

        self.X += self.rng.normal(size=self.X.shape) * self.latent_noise_scale
        self.X = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)

    def generate_returns(self):

        # generate the coefficient matrix for factor model
        B = np.zeros((self.m, self.n))

        # market/general exposure: most assets positive
        B[0, :] = self.rng.normal(loc=1.0, scale=0.25, size=self.n)

        # macro/rate exposure: mixed signs
        if self.m > 1:
            B[1, :] = self.rng.normal(loc=0.0, scale=0.4, size=self.n)

        # growth/sentiment exposure: mostly positive
        if self.m > 2:
            B[2, :] = self.rng.normal(loc=0.5, scale=0.3, size=self.n)

        # volatility/risk exposure: mostly negative
        if self.m > 3:
            B[3, :] = self.rng.normal(loc=-0.4, scale=0.25, size=self.n)

        # remaining factors: smaller idiosyncratic factor loadings
        for j in range(4, self.m):
            B[j, :] = self.rng.normal(loc=0.0, scale=0.15, size=self.n)

        self.B = B

        noise = self.rng.normal(size=(self.T, self.n))
        self.Y = self.X @ B + noise * self.returns_noise_scale



    def plot_ou(self):
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        axes[0].plot(self.theta[:, 0], label=r'$\theta_1$')
        axes[0].plot(self.theta[:, 1], label=r'$\theta_2$')
        axes[0].set_ylabel("Factor value")
        axes[0].set_title("Latent Ornstein-Uhlenbeck Process")
        axes[0].legend()

        axes[1].plot(self.X[:, 0], label=r'$X_1$')
        axes[1].plot(self.X[:, 1], label=r'$X_2$')
        axes[1].plot(self.X[:, 2], label=r'$X_3$')
        axes[1].plot(self.X[:, 3], label=r'$X_4$')
        axes[1].set_ylabel("Covariate value")
        axes[1].set_title("Generated Covariates")
        axes[1].legend()

        axes[2].plot(self.Y[:, 0], label=r'$Y_1$')
        axes[2].plot(self.Y[:, 1], label=r'$Y_2$')
        axes[2].plot(self.Y[:, 2], label=r'$Y_3$')
        axes[2].set_xlabel("Time step")
        axes[2].set_ylabel("Return value")
        axes[2].set_title("Generated Returns")
        axes[2].legend()

        fig.tight_layout()
        plt.show()



def main():
    sim = ToyMarketSimulator()
    sim.plot_ou()


if __name__ == "__main__":
    main()


