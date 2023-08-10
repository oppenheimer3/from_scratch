import numpy as np

def minimize(A, b):
    x = np.zeros(A.shape[1]).T
    grad_f = A.T @ A @ x - A.T @ b
    hessian = A.T @ A 
    hessian_inv = np.linalg.inv(hessian)
    x = x - hessian_inv @ grad_f
    return x


if __name__ == '__main__':
    A = np.array([[1, 2],
                   [4, 5],
                   [7, 8]])
    b = np.array([3, 4, 5]).T
    x = minimize(A, b)
    print(x)
    print(A @ x)