"""
1D Simulation Environment for Lexicographic Scalar Optimization

This module provides the core components for 1D simulation scenarios including:
- Reachable interval computation
- Distance metrics
- Cost functions
- System dynamics
"""

from typing import List
import numpy as np
from interval import interval  # type: ignore

# Numerical tolerance for interval operations
INTERVAL_TOLERANCE = 1e-9


def safe_intersection(
    x1: interval, x2: interval, tol: float = INTERVAL_TOLERANCE
) -> interval:
    """
    Compute intersection of two intervals with numerical tolerance.

    If intervals don't intersect but are within tolerance, returns the overlap region.

    Args:
        x1: First interval.
        x2: Second interval.
        tol: Numerical tolerance.

    Returns:
        Intersection interval.

    Raises:
        ValueError: If intervals are truly disjoint (beyond tolerance).
    """
    result = x1 & x2

    if len(result) != 0:
        return result

    # Check if intervals are within tolerance of each other
    lb1, ub1 = x1[0][0], x1[0][1]
    lb2, ub2 = x2[0][0], x2[0][1]

    # Compute potential intersection bounds
    lb = max(lb1, lb2)
    ub = min(ub1, ub2)

    # If the gap is within tolerance, create intersection
    if lb - ub <= tol:
        return interval[lb, ub] if lb <= ub else interval[(lb + ub) / 2]

    raise ValueError(f"Intervals {x1} and {x2} are disjoint beyond tolerance")


def compute_reachable_intervals(
    x_0: interval, cost_functions: List, system, use_discrete_cost: bool = False
) -> List[interval]:
    """
    Computes the reachable interval vector by forward and backward propagation.

    Args:
        x_0: Initial interval.
        cost_functions: List of CostFunction instances.
        system: An object that implements propagate_forward() and propagate_backward().
        use_discrete_cost: If True, uses evaluate_disc_cost(), otherwise evaluate_cont_cost().

    Returns:
        A list of refined reachable intervals (x_opt_bar), same length as x_opt.
    """

    # --- Forward propagation ---
    x_opt = [x_0]
    for cost_fct in cost_functions:
        x_next = system.propagate_forward(x_opt[-1])

        if use_discrete_cost:
            x_next_opt = cost_fct.evaluate_disc_cost(x_next)
        else:
            x_next_opt = cost_fct.evaluate_cont_cost(x_next)

        x_opt.append(x_next_opt)

    # --- Backward refinement ---
    x_opt_bar = x_opt.copy()  # hard copy

    for i in reversed(range(len(x_opt) - 1)):
        x_prev = system.propagate_backward(x_opt_bar[i + 1])

        try:
            x_prev_refined = safe_intersection(
                x_prev, x_opt[i]
            )  # Necessary because of numeric issues with "&"" operator
        except ValueError as e:
            raise ValueError(
                f"Empty intersection at index {i} during backward propagation! "
                f"x_prev={x_prev}, x_opt[{i}]={x_opt[i]}"
            ) from e

        x_opt_bar[i] = x_prev_refined

    return x_opt_bar


def is_cont_subset_of_disc(
    x_opt_cont: List[interval],
    x_opt_disc: List[interval],
    tol: float = INTERVAL_TOLERANCE,
) -> bool:
    """
    Checks whether x_opt_cont is a subset of x_opt_disc at every time step.

    For each index i, checks: x_opt_disc[i] ⊇ x_opt_cont[i], i.e.
        x_opt_disc[i].lb <= x_opt_cont[i].lb  AND
        x_opt_disc[i].ub >= x_opt_cont[i].ub

    A numerical tolerance is applied to handle floating-point rounding, so
    small violations within `tol` are treated as equality.

    Args:
        x_opt_cont: Reachable intervals from continuous optimization.
        x_opt_disc: Reachable intervals from discrete optimization.
        tol: Absolute tolerance for bound comparisons.

    Returns:
        True if x_opt_cont ⊆ x_opt_disc at every time step, False otherwise.
    """
    if len(x_opt_cont) != len(x_opt_disc):
        raise ValueError("Interval lists must have the same length.")

    for iv_cont, iv_disc in zip(x_opt_cont, x_opt_disc):
        cont_lb, cont_ub = iv_cont[0][0], iv_cont[0][1]
        disc_lb, disc_ub = iv_disc[0][0], iv_disc[0][1]

        if disc_lb > cont_lb + tol:
            return False
        if disc_ub < cont_ub - tol:
            return False

    return True


def average_directed_cost_excess(
    x_opt_cont: List[interval],
    x_opt_disc: List[interval],
    cost_functions: List,
) -> float:
    """
    Computes the average directed cost excess of x_opt_disc over x_opt_cont.

    For each time step t, the scalar worst-case violation cost is evaluated for
    both intervals using the corresponding cost function:
      - 'pos' direction: cost = max(0, interval.upper_bound - lb)
      - 'neg' direction: cost = max(0, ub - interval.lower_bound)

    Only the positive excess of x_opt_disc's cost over x_opt_cont's cost is
    accumulated (i.e., only when the discrete solution incurs more violation
    than the continuous one). The result is the average over all time steps.

    Args:
        x_opt_cont: Reachable intervals from continuous optimization.
        x_opt_disc: Reachable intervals from discrete optimization.
        cost_functions: List of CostFunction instances (length = len(x_opt_cont) - 1).

    Returns:
        Average directed cost excess (>= 0).
    """
    if len(x_opt_cont) != len(x_opt_disc):
        raise ValueError("Interval lists must have the same length.")
    if len(cost_functions) != len(x_opt_cont) - 1:
        raise ValueError("cost_functions length must be len(x_opt_cont) - 1.")

    total = 0.0
    n = len(cost_functions)

    for idx, cost_fct in enumerate(cost_functions):
        iv_cont = x_opt_cont[idx + 1]
        iv_disc = x_opt_disc[idx + 1]

        if cost_fct.direction == "pos":
            cost_cont = max(0.0, iv_cont[0][1] - cost_fct.lb)
            cost_disc = max(0.0, iv_disc[0][1] - cost_fct.lb)
        elif cost_fct.direction == "neg":
            cost_cont = max(0.0, cost_fct.ub - iv_cont[0][0])
            cost_disc = max(0.0, cost_fct.ub - iv_disc[0][0])
        else:
            raise ValueError("Direction must be either 'pos' or 'neg'")

        total += max(0.0, cost_disc - cost_cont)

    return total / n


def calculate_rule_violations(x_opt: List[interval], cost_functions: List) -> List[int]:
    violations = []
    for idx, cost_fct in enumerate(cost_functions):
        if cost_fct.direction == "pos":
            violations.append(min(0, cost_fct.lb - x_opt[idx + 1][0][1]))
        elif cost_fct.direction == "neg":
            violations.append(min(0, x_opt[idx + 1][0][0] - cost_fct.ub))
        else:
            raise ValueError("Direction must be either 'pos' or 'neg'")
    return violations


def mean_absolute_discrete_cost_difference(
    x_opt_1: List[interval], x_opt_2: List[interval], cost_functions: List
) -> float:
    """Computes the mean absolute difference per rule between the discrete cost idx of two solutions"""

    total_diff = 0.0
    for idx, cost_fct in enumerate(cost_functions):
        x_1 = x_opt_1[idx + 1]
        x_2 = x_opt_2[idx + 1]

        def get_discrete_cost_idx(x: interval, fct) -> int:
            for i, interv in enumerate(fct.intervals):
                if len(interv & x) != 0:
                    return i
            raise ValueError("No interval intersection found")

        cost_idx_1 = get_discrete_cost_idx(x_1, cost_fct)
        cost_idx_2 = get_discrete_cost_idx(x_2, cost_fct)

        total_diff += abs(cost_idx_1 - cost_idx_2)

    return total_diff  # / len(cost_functions) # TODO MAYBE DONT USE MEAN


def averaged_hausdorff_distance(
    a_intervals: List[interval], b_intervals: List[interval]
) -> float:
    """
    Computes the averaged Hausdorff distance between two lists of intervals.

    Args:
        a_intervals: First list of intervals.
        b_intervals: Second list of intervals.

    Returns:
        The averaged Hausdorff distance.
    """
    if len(a_intervals) != len(b_intervals):
        raise ValueError("Both interval lists must have the same length.")

    total_distance = 0.0
    for a_interval, b_interval in zip(a_intervals, b_intervals):
        d = max(
            abs(a_interval[0][0] - b_interval[0][0]),
            abs(a_interval[0][1] - b_interval[0][1]),
        )
        total_distance += d

    return total_distance / len(a_intervals)


class CostFunction:
    """
    Lexicographic cost function with interval-based discretization.

    Args:
        lb: Lower bound of the cost interval.
        ub: Upper bound of the cost interval.
        direction: 'pos' (minimize) or 'neg' (maximize).
        nof_intervals: Number of discretization intervals.
        range_limit: Range limit for infinity approximation (default: 1000).
        time: Time step at which this cost function is applied (default: 0).
    """

    def __init__(
        self,
        lb: float,
        ub: float,
        direction: str,
        nof_intervals: int = 1,
        range_limit: float = 1000,
        time: int = 0,
    ):
        self.lb = lb
        self.ub = ub
        self.direction = direction  # 'pos' or 'neg'
        self.nof_intervals = nof_intervals
        self.range_limit = range_limit
        self.intervals = self.set_intervals()
        self.time = time  # Time step at which this cost function is applied

    def set_intervals(self) -> List[interval]:
        """
        Generate discretized intervals based on direction.

        Returns:
            List of interval objects representing the discretization.
        """
        intervals = []

        if self.direction == "pos":
            # 1. Satisfied interval: (-inf to lb)
            intervals.append(interval[-self.range_limit, self.lb])

            if self.nof_intervals == 1:
                # One single violation interval covering the rest
                intervals.append(interval[self.lb, self.range_limit])
            else:
                # Multiple violation intervals
                # Discretize [lb, ub] into (nof_intervals - 1) segments
                n_between = self.nof_intervals - 1
                step = (self.ub - self.lb) / n_between

                for i in range(n_between):
                    a = self.lb + i * step
                    b = self.lb + (i + 1) * step
                    intervals.append(interval[a, b])

                # Final violation interval: (ub to +inf)
                intervals.append(interval[self.ub, self.range_limit])

        elif self.direction == "neg":
            # 1. Satisfied interval: (ub to +inf)
            intervals.append(interval[self.ub, self.range_limit])

            if self.nof_intervals == 1:
                # One single violation interval covering the rest
                intervals.append(interval[-self.range_limit, self.ub])
            else:
                # Multiple violation intervals
                # Discretize [lb, ub] into (nof_intervals - 1) segments
                n_between = self.nof_intervals - 1
                step = (self.ub - self.lb) / n_between

                # Iterate backwards from ub towards lb
                for i in range(n_between):
                    a = self.ub - (i + 1) * step
                    b = self.ub - i * step
                    intervals.append(interval[a, b])

                # Final violation interval: (-inf to lb)
                intervals.append(interval[-self.range_limit, self.lb])

        else:
            raise ValueError("Direction must be either 'pos' or 'neg'")

        return intervals

    def evaluate_cont_cost(self, x: interval) -> interval:
        """
        Evaluate continuous cost (optimal interval without discretization).

        Args:
            x: Input interval.

        Returns:
            Optimal interval according to the cost direction.
        """
        if self.direction == "pos":
            inter = interval[-self.range_limit, self.lb] & x
            return inter if len(inter) != 0 else interval[x[0][0]]  # lb
        elif self.direction == "neg":
            inter = interval[self.ub, self.range_limit] & x
            return inter if len(inter) != 0 else interval[x[0][1]]  # ub
        else:
            raise ValueError("Direction must be either 'pos' or 'neg'")

    def evaluate_disc_cost(self, x: interval) -> interval:
        """
        Evaluate discretized cost (first matching interval in lexicographic order).

        Args:
            x: Input interval.

        Returns:
            First interval that intersects with x.
        """
        for interv in self.intervals:
            inter = interv & x
            if len(inter) != 0:
                return inter
        else:
            raise ValueError("No interval intersection found")

    def evaluate_trajectory(self, y: np.ndarray) -> int:
        """
        Evaluate which interval a trajectory point falls into.

        Args:
            y: Trajectory array where y[0, self.time] is the state at this time step.

        Returns:
            Index of the interval containing the state.
        """
        x_t = y[0, self.time]

        # Both directions use the same logic - find containing interval
        for i, interv in enumerate(self.intervals):
            if interv[0][0] <= x_t <= interv[0][1]:
                return i

        # If x_t doesn't fall in any interval, raise an error
        raise ValueError(
            f"Value x_t={x_t} at time {self.time} does not fall within any interval"
        )

    def update_nof_intervals(self, nof_intervals: int):
        """
        Update the number of discretization intervals and regenerate intervals.

        Args:
            nof_intervals: New number of intervals.
        """
        self.nof_intervals = nof_intervals
        self.intervals = self.set_intervals()


class System:
    """
    1D linear system with bounded control input.

    The system evolves as: x[k+1] = x[k] + dt * u[k]
    where u[k] ∈ [u_min, u_max]

    Args:
        dt: Time step.
        u_min: Minimum control input.
        u_max: Maximum control input.
    """

    def __init__(self, dt: float, u_min: float, u_max: float):
        self.dt = dt
        self.u = interval[u_min, u_max]
        self.dt_u = self.dt * self.u

        self.u_min = u_min
        self.u_max = u_max

        # State-space representation for compatibility with the MPPI solver
        self.A = np.array([[1]])
        self.B = np.array([[dt]])
        self.C = np.array([[1], [0]])
        self.D = np.array([[0], [1]])

        self.m = 1  # state dimension
        self.n = 1  # input dimension
        self.p = 2  # output dimension

    def propagate_forward(self, x: interval) -> interval:
        """
        Forward propagation: x[k+1] = x[k] + dt * u

        Args:
            x: Current state interval.

        Returns:
            Next state interval.
        """
        x_next = x + self.dt_u
        return x_next

    def propagate_backward(self, x: interval) -> interval:
        """
        Backward propagation: x[k-1] = x[k] - dt * u

        Args:
            x: Current state interval.

        Returns:
            Previous state interval.
        """
        x_prev = x - self.dt_u
        return x_prev

    def f(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        State transition function: x[k+1] = A*x[k] + B*u[k]

        Args:
            x: Current state.
            u: Control input.

        Returns:
            Next state.
        """
        return self.A @ x + self.B @ u

    def g(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Output function: y[k] = C*x[k] + D*u[k]

        Args:
            x: Current state.
            u: Control input.

        Returns:
            Output.
        """
        return self.C @ x + self.D @ u
