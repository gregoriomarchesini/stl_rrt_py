import numpy as np
from scipy.linalg import expm
import control

def my_c2d(A, B, T):
    """
    Discretize (A,B) using matrix exponential trick.
    """
    n = A.shape[0]
    m = B.shape[1]

    # Build block matrix
    M = np.zeros((n+m, n+m))
    M[:n, :n] = A
    M[:n, n:] = B

    # Matrix exponential
    Mexp = expm(M * T)

    # Extract Ad, Bd
    Ad = Mexp[:n, :n]
    Bd = Mexp[:n, n:]
    return Ad, Bd

# Example system
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
T = 0.1  # sampling time

# --- Control package result ---
sysc = control.ss(A, B, np.eye(2), np.zeros((2,1)))  # state-space model
sysd = control.c2d(sysc, T)

Ad_control = np.array(sysd.A)
Bd_control = np.array(sysd.B)

# --- Our method ---
Ad_my, Bd_my = my_c2d(A, B, T)

# --- Compare ---
print("Ad (control):")
print(Ad_control)
print("\nAd (my method):")
print(Ad_my)
print("\nDifference Ad:")
print(Ad_control - Ad_my)

print("\nBd (control):")
print(Bd_control)
print("\nBd (my method):")
print(Bd_my)
print("\nDifference Bd:")
print(Bd_control - Bd_my)
