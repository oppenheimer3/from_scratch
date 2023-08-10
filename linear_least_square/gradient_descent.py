import numpy as np

def minimize(A, b):
    x = np.zeros(A.shape[1]).T
    eps = 1e-2
    sgm = 1e-2
    max_iterations = 1e3
    current_iteration = 0
    while current_iteration < max_iterations:
        current_iteration +=1
        grad_f = A.T @ A @ x - A.T @ b
        norm_grad = np.linalg.norm(grad_f)
        if norm_grad < sgm:
            break
        x = x - eps * grad_f

    return x



if __name__ == '__main__':
    A = np.array([[1, 2],
                   [4, 5],
                   [7, 8]])
    b = np.array([3, 4, 5]).T
    x = minimize(A, b)
    print(x)
    print(A @ x)