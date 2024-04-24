import numpy as np
from numpy import linalg as LA
from sympy import *
from sympy import latex, symbols, diff, IndexedBase
from sympy import hessian, Function
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy.integrate import solve_ivp
import sys
import time

# If True, report some intermediate results
verbose = False
# The minimum cost, determines a loop around 0 in the stable manifold
γ0 = 1e-5
# The maximum cost
γ1 = 2
# solve_ivp method parameter 
# method = 'LSODA'
method = 'RK45'

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    start = time.time()
    def show(j):
        x = int(size*j/count)
        remaining = ((time.time() - start) / j) * (count - j)
        
        mins, sec = divmod(remaining, 60)
        time_str = f"{int(mins):02}:{sec:05.2f}"
        
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count} Est wait {time_str}", end='\r', file=out, flush=True)
        
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)


# Progress bar: Usage
# for i in progressbar(range(15), "Computing: ", 40):
#     time.sleep(0.1) # any code you need


def broydens_good_method(F, x0, tol=1e-12, max_iter=100):
    """
    Implements Broyden's "Good" Method for solving F(x) = 0.
    
    Parameters:
    - F: The function to find roots for, must return a numpy array.
    - x0: Initial guess for the roots, numpy array.
    - tol: Tolerance for stopping criterion.
    - max_iter: Maximum number of iterations.
    
    Returns:
    - x: Approximation of the root.
    - success: Boolean indicating if a solution was found within tolerance.
    """
    n = len(x0)
    x = x0
    Fx = F(x)
    B = np.eye(n)  # Initial approximation to the inverse Jacobian
    
    for _ in range(max_iter):
        # Calculate step direction and update solution estimate
        delta_x = -np.dot(B, Fx)
        x_new = x + delta_x
        F_new = F(x_new)
        
        # Check if solution is found
        if np.linalg.norm(F_new) < tol:
            return x_new, True
        
        # Update B using Broyden's formula
        delta_F = F_new - Fx
        B += np.outer((delta_x - np.dot(B, delta_F)), np.dot(delta_x, B)) / np.dot(delta_x, np.dot(B, delta_F))
        
        # Prepare for next iteration
        x, Fx = x_new, F_new
    
    return x, False

# # Example usage
# def F_example(x):
#     """
#     A sample function for demonstration: F(x) = [x_1^2 + x_2^2 - 1, x_1^2 - x_2^2 + 0.5]
#     Its root represents the intersection of a circle and a hyperbola.
#     """
#     return np.array([x[0]**2 + x[1]**2 - 1, x[0]**2 - x[1]**2 + 0.5])

# # Initial guess
# x0 = np.array([0.5, 0.5])

# # Solve
# solution, success = broydens_good_method(F_example, x0)

# solution, success

class pendulum_components:
    def __init__(self, α=.1, num_fronts = 256, num_steps = 2880, atol = 1e-12):
        self.num_fronts = num_fronts  # Number of fronts
        self.num_steps = num_steps    # Number of angle steps
        self.atol = atol              # Absolute tolerance
        self.α = α
        self.build_components()
        self.build_optimal_solution_table()

    def build_components(self):
        """construct various components useful
        in making the pendulum controller, for a given value of
        the friction constant α: 
        - H (symbolic expr) : the effective Hamiltonian
        - Hessian (symbolic expr) : The Hessian at 0
        - Vars (list of symbols) : A list of symbols used as variables
        - P (float): The gain matrix (solution to Algebraic Ricatti Equation)
        - VFRev (callable): The Hamiltonian vector field with reversed and scaled
                           time, with cost serving as time
        - eval_H (callable) : Evaluates Hamiltonian on (state, costate)
        - eval_control : Evaluates optimal control (u) on (state, costate)

        """
        # Find symbolic Hamiltonian. Note: γ represents cost
        θ, φ, u, λ1, λ2, γ = symbols("θ φ u λ1 λ2 γ")
        α = self.α
        λ = [1,λ1,λ2]
        f = [ 1/2*φ**2 + (1-cos(θ)) + 1/2*u**2,
              φ,
              sin(θ)+u*cos(θ)-α*φ ]
        H = sum(λ[k]*f[k] for k in range(0,3))
        Vars = [θ,φ,λ1,λ2]

        # Find optimal control
        u_star = solve(diff(H,u),u)[0]

        # Substitute optimal control, find effective Hamiltonian
        H = H.subs([(u,u_star)])

        # Substitution list - every variable set to 0
        zero_sub = [(Vars[k], 0) for k in range(0,len(Vars))]

        # Calculate Hessian of the Hamiltonian at 0
        Hessian = hessian(H, Vars).subs(zero_sub)

        # Make the matrix J
        n = len(Vars) //2
        Z = np.zeros([n,n],dtype=np.float64)
        I = np.identity(n,dtype=np.float64)
        J = np.block([[Z, I], [-I, Z]])

        # Linearize the Hamiltonian system
        LinSys = J @ Hessian
        LinSys = np.array(LinSys, dtype=np.float64)

        # Find the gain matrix P
        Lambda, V = LA.eig(LinSys)    
        idx = np.real(Lambda) < 0

        Vs = V[:,idx]
        Lambda = np.diag(Lambda[idx])

        fuzz = np.linalg.norm(LinSys @ Vs - Vs @ Lambda)
        if fuzz > self.atol:
            raise Exception(f"Failure to diagonalize.")         

        m, n = Vs.shape
        Vs1 = Vs[:n,:]
        Vs2 = Vs[n:,:]

        P = Vs2 @ np.linalg.inv(Vs1)
        P = np.real(P)
        imag = np.linalg.norm(np.imag(P))
        if imag > self.atol:
            raise Exception(f"Gain matrix has imaginary part.") 

        # Compute the Hamiltonian vector field
        XH = J@Matrix([H]).jacobian(Vars).transpose()
        # Reverse time
        XHRev = -XH

        # Make cost to be the time, divide by f[0]
        XH    = XH / f[0].subs([(u,u_star)]) 
        XHRev = XHRev / f[0].subs([(u,u_star)]) 

        # The lamdified Hamiltonian vector field with reversed time, and cost as time (independent variable)
        VarsExt = Vars.copy()
        VarsExt.insert(0,γ)

        # Forward vector field components
        F = lambdify(VarsExt, XH.transpose().tolist()[0])

        # Backward vector field components
        FRev = lambdify(VarsExt, XHRev.transpose().tolist()[0])

        # The non-autonomous version of F
        VF = lambda γ, y: F(γ, y[0],y[1],y[2],y[3])

        # The non-autonomous version of FRev
        VFRev = lambda γ, y: FRev(γ, y[0],y[1],y[2],y[3])

        # Makes a substitution from a state and costate
        make_sub  = lambda state, costate : [(θ,state[0]), (φ,state[1]), (λ1,costate[0]),(λ2,costate[1])]

        # A utility to evaluate Hamiltonian for given state and costate
        eval_H = lambda state, costate : H.subs(make_sub(state, costate))

        # A utility to evaluate control for given state and costate
        eval_control = lambda state, costate : u_star.subs(make_sub(state, costate))

        self.α = α
        self.P = P
        self.H = H
        self.Vars = Vars
        self.Hessian = Hessian
        self.VF = VF
        self.VFRev = VFRev
        self.eval_H = eval_H
        self.eval_control = eval_control
    

    def build_optimal_solution_table(self):
        AngleRange = (0, 2*np.pi)

        a, b = AngleRange

        self.Γ = np.linspace(γ0, γ1, self.num_fronts)
        self.Ψ = np.linspace(a, b, self.num_steps)

        Y = ()
        for ψ in progressbar(self.Ψ, "Computing: ", 40):
            #for k in range(0, self.num_steps):
            x0 = np.array([cos(ψ),sin(ψ)],dtype=np.float64)
            λ0 = self.P @ x0
            γ2 = 0.5 * (λ0.transpose() @ x0)
            # Make cost γ0
            ξ0 = np.append(x0, λ0)
            ξ0 *= np.sqrt(γ0/γ2)
            sol = solve_ivp(self.VFRev, [γ0, γ1], ξ0, t_eval = self.Γ, atol = self.atol, method = method)
            if not sol.success:
                raise Exception(f"ODE solver failed.") 
            _, y = sol.t, sol.y
            Y = Y + (y,)
        self.Y = np.stack(Y)
        self.fronts = self.Y[:,0:2,:].swapaxes(1,2)
        self.costates = self.Y[:,2:4,:].swapaxes(1,2)

    def approximate_solution(self, θ0, φ0):
        """Accept initial state and return approximate nearby solution.
        Parameters:
        θ0 (float): Initial angle
        φ0 (float): Initial angular velocity

        Returns:
        float: best_angle - the approximate angle of entry
        float: best_cost  - the cost of path
        float: best_costate - the costate at the end of the path
        float: best_dist - the distance from the initial condition (θ0, φ0)

        Exceptions:
        None
        """
        # Calculate squared distances to avoid unnecessary sqrt for efficiency
        distances_squared = np.sum((self.fronts - np.array([θ0, φ0]))**2, axis=-1)

        # Find the index of the smallest distance
        min_index = np.unravel_index(np.argmin(distances_squared), distances_squared.shape)

        angle_idx, front_idx = min_index

        ψ = self.Ψ[angle_idx]
        γ = self.Γ[front_idx]
        x = self.fronts[angle_idx, front_idx, :]
        λ = self.costates[angle_idx, front_idx, :]
        best_dist = sqrt(distances_squared[angle_idx, front_idx])

        Hval = self.eval_H(x, λ)
        u = self.eval_control(x, λ)

        if verbose:
            print(f"==== THE INITIAL APPROXIMATION ==== ")
            print(f"The index of the closest state is: {min_index}")
            print(f"The best angle is: {ψ}")
            print(f"The best cost is: {γ}")
            print(f"The closest state is: {x} with distance: {best_dist}")
            print(f"The best costate is: {λ}")
            print(f"The Hamiltonian of the best state + costate: {Hval}")
            print(f"The optimal control: {u}")
            self.plot_fronts(show = False)
            plt.plot(x[0], x[1], 'o')
            plt.show()

        return ψ, γ, λ, best_dist


    def plot_fronts(self, show = True):
        colors = cm.rainbow(np.linspace(0, 1, self.num_fronts))
        for l, c in zip(range(self.num_fronts), colors):
            plt.plot(self.fronts[:,l,0],self.fronts[:,l,1],'-', color=c)
        colors = cm.rainbow(np.linspace(0, 1, self.num_steps))
        for k, c in zip(range(self.num_steps),colors):
            plt.plot(self.fronts[k,:,0],self.fronts[k,:,1],'-', color=c)
        plt.xlabel('θ'),
        plt.ylabel('φ')

        if show:
            plt.show()


    def optimal_path_from_zero(self, ψ, γ):
        """ The call x,λ = optimal_path_from_zero(ψ, γ)
        calculates the end of the optimal path within
        the stable manifold of 0. We are shooting a path from (x,λ), where λ=P*x,
        and x is on the ellipse (1/2)*xᵀ*P*x=e**γ0, thus making γ0 the cost
        of the initial point. The direction of
        the shot is parameterized by ψ, which is the polar coordinate in the
        (θ,φ)-plane. The end point (θ,φ) has cost γ. 
        """
        x0 = np.array([cos(ψ),sin(ψ)],dtype=np.float64)
        λ0 = self.P @ x0
        γ2 = 0.5 * (λ0.transpose() @ x0)
        ξ0 = np.append(x0, λ0)
        ξ0 *= np.sqrt(γ0/γ2)    # Scale to have cost γ0

        t_eval = [γ0, γ]
        sol = solve_ivp(self.VFRev, [γ0, γ], ξ0, t_eval = t_eval, atol = self.atol, method = method)
        if not sol.success:
            raise Exception(f"ODE solver failed.") 
        t, y = sol.t, sol.y
        ξ1 = y[:,-1]
        θ, φ, λ1, λ2 = ξ1
        x = np.array([θ, φ], dtype=np.float64)
        λ = np.array([λ1,λ2], dtype=np.float64)
        return x, λ

    def shoot_from_stable_loop(self, z):
        ψ, γ = z
        x, _ = self.optimal_path_from_zero(ψ, γ)
        return x

    def find_best_path(self, θ0, φ0):
        """
        Brings mathematical pendulum with friction, subject to an
        external horizontal force to rest, minimizing cost.

        Parameters:
        θ0 (float): Initial angle
        φ0 (float): Initial angular velocity
        α (float): Friction constant (>= 0)

        Returns:
        (float, float): The costate which leads to optimal trajectory
        float: The total cost value
        Exceptions:
        None
        """
        # Find the approximate nearby target point
        ψ, γ, λ, dist = self.approximate_solution(θ0, φ0)

        if verbose:
            print(f"Initial approximation distance: {dist}")


        # Refine the approximate solution
        z0 = [ψ,γ]
        x0 = np.array([θ0, φ0])
        max_iter = 100
        z, success = broydens_good_method(lambda z : self.shoot_from_stable_loop(z) - x0,
                                          z0,
                                          tol=self.atol,
                                          max_iter=max_iter)
        if not success:
            raise Exception(f"Broyden failed to converge in {max_iter} iterations.") 

        ψ, γ = z
        x, λ = self.optimal_path_from_zero(ψ, γ)

        return λ, γ

    def check_solution(self, θ0, φ0, λ = None, J = None):
        """ Check the solution by Solving ODE backwards in time
        Parameters:
        θ0 (float): Initial angle
        φ0 (float): Initial angular velocity
        λ (float, float): Costate
        J (float): Cost
        Returns:
        (float): Distance of final state from 0
        Exceptions:
        None
        """
        x0 = np.array([θ0, φ0])

        if λ is None:
            λ = self.P @ x0
            J = 0.5 * (λ.transpose() @ x0)

        γ = J
        t_eval = np.linspace(γ, γ0, 256)

        ξ0 = np.append(x0, λ)
        sol = solve_ivp(self.VFRev, [γ, γ0], ξ0, t_eval = t_eval, atol = self.atol, method = method)
        if not sol.success:
            raise Exception(f"ODE solver failed.") 

        θ = sol.y[0,-1]
        φ = sol.y[1,-1]
        x = np.array([θ, φ], dtype=np.float64)
        cost_from_zero = np.sqrt(0.5*x.transpose()@ self.P@x)

        if verbose:
            print(f"==== SOLUTION CHECK ====")
            print(f"Cost from 0: {cost_from_zero}")
            #print(f"Solution going to 0: {sol}")
            plt.plot(sol.y[0,:], sol.y[1,:],'-o', color='blue')
            plt.xlabel('θ'),
            plt.ylabel('φ')
            plt.plot(θ0, φ0, 'o', color='orange')
            plt.plot(0, 0, 'o', color='red')
            plt.show()

        return cost_from_zero

    def check_gain_matrix(self, θ0, φ0):
        cost_from_zero = self.check_solution(θ0, φ0, None, None)
        return cost_from_zero

    def test_control_pendulum(self, control_pendulum_fun, θ0, φ0):
        λ, J = control_pendulum_fun(θ0, φ0, alpha = self.α)
        x=(θ0,φ0)
        print(f"For state {x} and alpha {self.α} the program returned costate: {λ} and cost: {J}"); 
        x = np.array([θ0,φ0], dtype=np.float64)
        Hval = self.eval_H(x, λ)
        cost_from_zero = self.check_solution(θ0, φ0, λ, J)
        return cost_from_zero, Hval


def control_pendulum(theta0, phi0, alpha = 0):
    """
      Brings mathematical pendulum with friction, subject to an
      external horizontal force to rest, minimizing cost.

      Parameters:
      theta0 (float): Initial angle
      phi0 (float): Initial angular velocity
      alpha (float): Friction constant (>= 0)

      Returns:
      (float, float): The costate which leads to optimal trajectory
      float: The total cost value
      Exceptions:
      None
      """
    ob = pendulum_components(α = alpha)
    θ0 = theta0
    φ0 = phi0

    λ, γ = ob.find_best_path(θ0, φ0)      
    lambda1, lambda2 = λ
    J = γ
    return (lambda1, lambda2), J

def run():
    # Given point
    θ0, φ0, α = 0.01, 0, 0.1

    ob = pendulum_components(α = α)
    cost_from_zero = ob.check_gain_matrix(θ0, φ0)

    θ0, φ0 = 1, 0
    cost_from_zero = ob.test_control_pendulum(control_pendulum, θ0, φ0)

if __name__ == '__main__':
    verbose = True
    run()
