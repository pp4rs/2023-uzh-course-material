import numpy as np


class Grid:

    def __init__(self, n_y, rho, sd_log_y, n_a, min_a, max_a):
        self.n_y = n_y
        self.rho = rho
        self.sd_log_y = sd_log_y
        self.n_a = n_a
        self.min_a = min_a
        self.max_a = max_a
        self.Pi, self.grid_y = self.rouwenhorst(n_y, rho, sd_log_y)
        self.pi_ss = self.stationary_dist(self.Pi)[0,:]
        self.grid_y = self.normalize_y(self.grid_y, self.pi_ss)
        self.grid_a = self.discretize_assets(min_a, max_a, n_a)

    # sigma is the sd of the error, e_t
    def rouwenhorst(self, n, rho, sd_log_y):
        
        # the grid    
        e = np.arange(n) # sd of e on this grid with Pi is sqrt(n-1)/2
        e = e / ( (n-1)**0.5 /2 ) # now its unit sd
        e = e * sd_log_y # now it's the sd of the cross section of log_y

        # the transition matrix
        p = (1+rho)/2
        Pi = np.array([[p, 1-p], [1-p, p]])
        
        while Pi.shape[0] < n:
            Pi_next = np.zeros((1+Pi.shape[0], 1+Pi.shape[1]))
            Pi_next[0:Pi.shape[0], 0:Pi.shape[1]] += Pi * p
            Pi_next[0:Pi.shape[0], -Pi.shape[1]:] += Pi * (1-p)
            Pi_next[-Pi.shape[0]:, -Pi.shape[1]:] += Pi * p
            Pi_next[-Pi.shape[0]:, 0:Pi.shape[1]] += Pi * (1-p)
            Pi_next[1:-1, :] /= 2
            Pi = Pi_next

        return Pi, e

    def stationary_dist(self, Pi):
        Pi_stationary = Pi.copy()
        eps = 1
        while eps > 10E-12:
            Pi_old = Pi_stationary.copy()
            Pi_stationary = Pi_stationary @ Pi_stationary
            eps = np.max(np.abs(Pi_stationary - Pi_old))

        if np.max(
                np.abs( 
                    np.sum(Pi_stationary - Pi_stationary,axis = 0) / Pi_stationary.shape[0]
                )
            ) < 10E-10:
            print("the steady state is unique.")

        return Pi_stationary

    def normalize_y(self, log_y, pi_ss): # make y have unit mean
        y = np.exp(log_y)
        y = y / np.vdot(y, pi_ss)
        return y


    # write a function which discretizes the asset space
    def discretize_assets(self, amin, amax, n_a):
        # find ubar 
        ubar = np.log(np.log(amax - amin + 1)+1)
        # make linar grid for the u's
        grid_u = np.linspace(0,ubar, n_a)
        # transform back to a
        grid_a = amin + np.exp(np.exp(grid_u)-1)-1
        return grid_a


from numba import njit

@njit
def backward_iteration(V_a, beta, eis, r, grid_a, grid_y, Pi):
    W = beta * Pi @ V_a

    c_endog = W ** (-1/eis)
    a_endog = (1 + r)**(-1) * (grid_a[np.newaxis, :] - grid_y[:, np.newaxis] + c_endog)
    a_prime = np.empty((grid_y.shape[0], grid_a.shape[0]))

    for i_y in range(grid_y.shape[0]):
        a_prime[i_y, :] = np.interp(grid_a, a_endog[i_y, :], grid_a)

    c = (1 + r) * (grid_a[np.newaxis, :] + grid_y[:, np.newaxis]) - a_prime

    return c, a_prime

def get_lotteries(policy, grid_a):

    indexes = np.searchsorted(grid_a, policy) # indexes corresponding to a'(y, a)+  (function returns i with a[i-1] < v <= a[i])
    q = (policy - grid_a[indexes - 1]) / (grid_a[indexes] - grid_a[indexes - 1]) # lotteries
    return indexes, q

# forward iteration
@njit
def forward_iteration(indexes, q, Pi, D):
    n_y, n_a = D.shape
    D_new = np.zeros((n_y, n_a))
    for y in range(n_y):
        for a in range(n_a):
            
            D_new[y, indexes[y, a]]   += q[y, a] * D[y, a]
            D_new[y, indexes[y, a]-1]     += (1 - q[y, a]) * D[y, a]

    # D_new is D_tilde right now. Now we need to update D_tilde using Pi
    D_new = Pi @ D_new

    return D_new

class SteadyStateHH:

    def __init__(self, model_params, grid_params, tol = 1e-6, max_iter = 1_000):
        self.model_params = model_params
        self.grid_params = grid_params
        self.Grids = Grid(n_y = grid_params['n_y'], rho = model_params['rho'],  sd_log_y = model_params['sd_log_y'], n_a = grid_params['n_a'], min_a = grid_params['min_a'], max_a = grid_params['max_a'])
        self.tol = tol
        self.max_iter = max_iter
        self.c = None
        self.a_prime = None
        self.V_a = None
        self.D = None

    # adding the model_params as an argument allows solving for different parameterizations
    def solve_ss(self, model_params):

        # update grid if necessary
        if (self.model_params['rho'], self.model_params['sd_log_y']) != (model_params['rho'], model_params['sd_log_y']):
            self.Grids = Grid(n_y = self.grid_params['n_y'], rho = model_params['rho'],  sd_log_y = model_params['sd_log_y'], n_a = self.grid_params['n_a'], min_a = self.grid_params['min_a'], max_a = self.grid_params['max_a'])
        # update model_params if necessary
        if self.model_params != model_params:
            self.model_params = model_params

        # initialize value function derivative with guess
        if self.V_a is None:
            V_a = np.ones((self.grid_params['n_y'], self.grid_params['n_a']))
        else:
            V_a = self.V_a

        for i in range(self.max_iter):
            c, a_prime = backward_iteration(V_a, model_params['beta'], model_params['eis'], model_params['r'], self.Grids.grid_a, self.Grids.grid_y, self.Grids.Pi)
            V_a_new = (1 + model_params['r']) * c **(-1/model_params['eis'])

            if np.max(np.abs(V_a_new - V_a)) < self.tol:
                break

            V_a = V_a_new

        self.c = c
        self.a_prime = a_prime
        self.V_a = V_a

        return c, a_prime
    
    def distribution_ss(self, maxiter=10_000, tol=1E-10, verbose=False):
        
        assert self.a_prime is not None, "solve_ss must be called first"

        Pi = self.Grids.Pi
        grid_a = self.Grids.grid_a
        policy = self.a_prime

        indexes, q = get_lotteries(policy, grid_a)

        # initialize distribution
        D = np.ones_like(policy)/np.size(policy)

        count, error = 0, 1
        while error > tol and count < maxiter:
            D_new = forward_iteration(indexes, q, Pi, D)
            error = np.max(np.abs(D - D_new))
            D = D_new.copy()
            count += 1
            
        
        if verbose : 
            print("max |D_t - D_t+1| = ", error, "\nnum iterations:", count)

        self.D = D

        return D

    def plot_policy(self, bound_grid = 0.4):
        """ 
        Plot the policy function for the first 4 income states
        bound_grid: float, fraction of the grid to plot
        """
        rng_asset_grid = int(grid_params['n_a']*bound_grid)
        fig, ax = plt.subplots()
        for i_y, y in enumerate(self.Grids.grid_y[0:4]):
            ax.plot(self.Grids.grid_a[0:rng_asset_grid], self.c[i_y, 0:rng_asset_grid], label = f'y = {y:.2f}')
        ax.set(xlabel = r'$a$', ylabel = r'$c(y,a)$', title = 'Steady State Policy Function')
        plt.legend(fontsize = 'small')
        plt.show()