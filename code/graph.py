import scipy.optimize as opt
import numpy as np

def threshold():

    # Coefficients of the objective function to be minimized wi, alpha
    alpha_coe = np.array([0, 0, 0, 0, 1])

    Ine_M = np.array([[-1, -1, 0, 0, 0],
                     [ 0, 0, -1, -1, 0],
                     [1,0, 1, 0, -1],
                     [1, 0, 0, 1, -1],
                     [0, 1, 1, 0, -1],
                     [0, 1, 0, 1, -1]])

    Ine_b = np.array([-1, -1, 0, 0, 0, 0])

    # Bounds for variables (0 <= wi)
    w1 = (0, None)
    w2 = (0, None)
    w3 = (0, None)
    w4 = (0, None)

    # Solve the linear programming problem
    result = opt.linprog(c, A_ub=Ine_M, b_ub=Ine_b, A_eq=None, b_eq=None, bounds=[x1, x2, x3, z], method='highs')

    # Print the results
    print("Status:", result.message)
    print("Optimal Values (x1, x2, x3):", result.x[:-1])
    print("expected scores can be achieved", result.x[-1])