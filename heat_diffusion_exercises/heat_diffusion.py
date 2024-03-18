import numpy as np
import matplotlib.pyplot as plt

class heat_diff_model:
    def __init__(self,L,N,dx,Q):
        # Define parameters
        self.L = L # length of the rod
        self.N = N  # number of grid points
        self.dx = self.L / (self.N - 1)  # grid spacing
        self.Q = Q  # heat generation rate

    def bc_setup(self,T_left,T_right):
        # Define the grid
        self.x = np.linspace(0, self.L, self.N)

        # Initialize the solution vector
        self.T = np.zeros(self.N)

        # Set the initial guess for the solution
        self.T[0] = T_left
        self.T[-1] = T_right

        # Define the coefficient matrix
        self.A = np.zeros((self.N, self.N))
        self.A[0, 0] = 1.0
        self.A[-1, -1] = 1.0

    def heat_solve(self,k):
        for i in range(1, self.N - 1):
            self.A[i, i - 1] = -k / self.dx**2
            self.A[i, i] = 2.0 * k / self.dx**2 + self.Q
            self.A[i, i + 1] = -k / self.dx**2

        # Solve the linear system of equations
        self.T = np.linalg.solve(self.A, self.T)
        
        return self.T

    def return_T_in_x_position(self,n):
        return self.T[n]

# Plot the solution
# Define parameters
# L = 1.0  # length of the rod
# N = 101  # number of grid points
# dx = L / (N - 1)  # grid spacing
# Q = 1.0  # heat generation rate
# k = 0.01

# # Set the boundary conditions
# T_left = 0.0
# T_right = 1.0

# model = heat_diff_model(L,N,dx,Q)
# model.bc_setup(T_left,T_right)

# plt.plot(model.x, model.heat_solve(k))
# plt.xlabel('Distance')
# plt.ylabel('Temperature')
# plt.title('Steady Heat Diffusion Equation')
# plt.show()