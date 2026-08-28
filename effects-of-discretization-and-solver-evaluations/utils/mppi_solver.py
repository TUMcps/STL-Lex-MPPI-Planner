import math
import time
import numpy as np


class MPPISolver:
    """
    MPPI Solver
    """

    def __init__(
        self,
        cost_functions: list,
        x0,
        K,
        system,
        n_iterations: int = 10,
        n_samples_constant: int = 500,
        n_samples_cosine: int = 950,
        n_samples_cosine_min: int = 50,
        cov=np.array([]),
        lamb: float = 2.0,
        gamma: float = 0.6,
        beta_shrinking_method: str = "exponential",  # "cosine"
        beta_min: float = 1e-6,
        samples_shrinking_method: str = "constant",  # "cosine"
        input_clipping: bool = False,
        random_seed=None,
        verbose: bool = True,
        pregenerated_eps=None,
    ):

        self.cost_functions = cost_functions  # List of cost functions
        self.x0 = x0  # Initial state
        self.K = K  # Time horizon
        self.sys = system  # System

        # Parameters
        self.n_iterations = n_iterations  # number of iterations
        self.n_samples_constant = n_samples_constant
        self.n_samples_cosine = n_samples_cosine
        self.n_samples_cosine_min = n_samples_cosine_min
        self.cov = cov  # Initial covariance
        self.lamb = lamb  # Initial temperature
        self.gamma = gamma  # For exponential decay rule
        self.beta_shrinking_method = beta_shrinking_method  # Method for shrinking beta
        self.beta_min = beta_min  # Minimum beta value for shrinking
        self.samples_shrinking_method = samples_shrinking_method  # Shrinking method
        self.input_clipping = input_clipping  # If True, clip inputs to system limits
        self.random_seed = random_seed  # Random seed for reproducibility
        self.verbose = verbose
        self.pregenerated_eps = pregenerated_eps

        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(random_seed)

        # Initialize best sample tracking across all iterations
        self.best_overall_lex_cost = float("inf")
        self.best_overall_u = None

        # Precompute power sums
        self.power_sums = self.precompute_power_sums()

        if self.verbose:
            self.print_total_samples_planned()

    def print_total_samples_planned(self):
        total_planned_samples = sum(
            self.calculate_number_of_samples(i) for i in range(self.n_iterations)
        )
        print(f"Total planned samples: {total_planned_samples}")

    def precompute_power_sums(self):
        bits = [math.ceil(math.log2(c.nof_intervals + 1)) for c in self.cost_functions]
        return [1 << int(sum(bits[i + 1 :])) for i in range(len(bits))]

    def calculate_beta(self, iteration):
        if self.beta_shrinking_method == "exponential":
            return math.sqrt(self.gamma**iteration)
        elif self.beta_shrinking_method == "cosine":
            if self.n_iterations == 0:
                return self.beta_min
            ratio = 0.5 * (1 - math.cos(math.pi * iteration / (self.n_iterations - 1)))
            return 1.0 - (1.0 - self.beta_min) * ratio
        else:
            raise ValueError(
                f"Unknown beta shrinking method: {self.beta_shrinking_method}"
            )

    def calculate_number_of_samples(self, iteration):
        if self.samples_shrinking_method == "constant":
            return self.n_samples_constant
        elif self.samples_shrinking_method == "cosine":
            if self.n_iterations == 0:
                return self.n_samples_cosine
            ratio = 0.5 * (1 - math.cos(math.pi * iteration / (self.n_iterations - 1)))
            return round(
                self.n_samples_cosine
                - (self.n_samples_cosine - self.n_samples_cosine_min) * ratio
            )
        else:
            raise ValueError(
                f"Unknown samples shrinking method: {self.samples_shrinking_method}"
            )

    def solve(self):
        # Initialize lists for recordings
        record_y_opt, record_y, record_best_sample_idx, record_cost = [], [], [], []
        record_beta, record_n_samples = [], []

        # Initialize input trajectory
        u = np.zeros([self.sys.m, self.K])

        # Start timer
        t_start = time.time()

        # Perform the optimization for n_iterations
        for i in range(self.n_iterations):
            if self.verbose:
                print(
                    f"Iteration: {i + 1}/{self.n_iterations} --------------------------------------------"
                )

            # Calculate parameters for the current iteration
            beta = self.calculate_beta(i)
            n_samples_current = self.calculate_number_of_samples(i)
            cov = beta * self.cov
            lamb = beta**2 * self.lamb

            # Execute step of the MPPI algorithm
            u, y_opt, y, best_sample_idx, cost = self.mppi_step(
                i, u, cov, lamb, n_samples_current
            )

            # Record intermediate solutions
            record_n_samples.append(n_samples_current)
            record_beta.append(beta)
            record_y_opt.append(y_opt)
            record_y.append(y)
            record_best_sample_idx.append(best_sample_idx)
            record_cost.append(cost)

        # End timer
        solve_time = time.time() - t_start

        # Get overall best sample
        best_overall_x, _ = self.forward_rollout(self.best_overall_u)

        # Outputs
        final_u = u
        final_x, final_y = self.forward_rollout(final_u)
        cost = self.evaluate(final_y)

        return (
            final_x,
            final_u,
            cost,
            solve_time,
            record_y_opt,
            record_y,
            record_best_sample_idx,
            record_cost,
            best_overall_x,
            self.best_overall_lex_cost,
            record_n_samples,
            record_beta,
        )

    def mppi_step(self, iteration_idx, u, cov, lamb, n_samples):
        x_dim = self.sys.n
        y_dim = self.sys.p
        u_dim = self.sys.m

        eps = np.zeros([n_samples, u_dim, self.K])  # initialize epsilons
        u_samples = np.zeros([n_samples, u_dim, self.K])  # initialize u_samples
        x = np.zeros([n_samples, x_dim, self.K])  # initialize state trajectories
        y = np.zeros([n_samples, y_dim, self.K])  # initialize output trajectories
        cost = np.zeros(n_samples)  # initialize path costs
        lex_cost = np.zeros(n_samples)  # initialize path costs

        cov_inv = np.linalg.inv(cov)  # Invert covariance

        # Generate new samples
        for n in range(n_samples):
            # Retrieve pre-generated epsilons (already drawn from N(0, beta_i * cov))
            # or draw fresh samples from the current scaled covariance
            if self.pregenerated_eps is not None:
                eps[n, ...] = self.pregenerated_eps[iteration_idx, n, :, :]
            else:
                eps[n, ...] = np.random.multivariate_normal(
                    mean=np.zeros(u_dim), cov=cov, size=self.K
                ).T

            # Apply input constraints
            if self.input_clipping:
                u_sample = np.clip(u + eps[n, ...], self.sys.u_min, self.sys.u_max)
                eps[n, ...] = u_sample - u  # Adjust eps after clipping

            u_sample = u + eps[n, ...]
            u_samples[n, ...] = u_sample

            x[n, ...], y[n, ...] = self.forward_rollout(u_sample)

            # Calculate cost
            cost[n], lex_cost[n] = self.calculate_path_cost(
                x[n, ...], y[n, ...], eps[n, ...], u, lamb, cov_inv
            )

        # get trajectory with the lowest costs
        best_sample = np.argmin(cost)
        psi = cost[best_sample]

        # Track the best overall sample across all iterations
        best_lex_sample = np.argmin(lex_cost)
        if lex_cost[best_lex_sample] < self.best_overall_lex_cost:
            self.best_overall_lex_cost = lex_cost[best_lex_sample]
            self.best_overall_u = u_samples[best_lex_sample].copy()

        # calculate weightings for each sample
        eta = np.sum(np.exp(-1 / lamb * (cost - psi)))
        omega = 1 / eta * np.exp(-1 / lamb * (cost - psi))

        for k in range(self.K):
            u[:, k] += omega @ eps[:, :, k]

        _, y_opt = self.forward_rollout(u)

        return u, y_opt, y, best_sample, cost

    def forward_rollout(self, u):
        """Performs an integration step of the system dynamics"""
        x = np.empty((self.sys.n, self.K))
        x[:, 0] = self.x0

        f = self.sys.f
        for k in range(self.K - 1):
            x[:, k + 1] = f(x[:, k], u[:, k])

        return x, x  # This assumes y = x

    def calculate_path_cost(self, x, y, w, u, lamb, cov_inv):
        """Calculate the path costs"""
        mppi_cost = 0
        for k in range(self.K - 1):
            mppi_cost += lamb * np.sum(u[:, k] @ cov_inv @ w[:, k])  # State cost
        lex_cost = self.evaluate(y)  # Lex cost
        return mppi_cost + lex_cost, lex_cost

    def evaluate(self, y):
        rule_costs = []
        for cost_func in self.cost_functions:
            rule_costs.append(int(cost_func.evaluate_trajectory(y)))

        c_bar = 0
        for i in range(len(rule_costs)):
            c_bar += rule_costs[i] * self.power_sums[i]

        return c_bar
