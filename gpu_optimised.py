from numba import cuda
import math

@cuda.jit(device=True, inline=True)
def custom_sqrt(x):
    # Utilize CUDA's fast math function if possible
    return math.sqrt(x)

@cuda.jit(device=True, inline=True)
def hypot(x, y):
    # Use the optimized custom_sqrt
    return custom_sqrt(x * x + y * y)

@cuda.jit(device=True, inline=True)
def vector_subtract(a, b, result):
    # Unroll the loop to improve performance if length is known
    for i in range(len(a)):
        result[i] = a[i] - b[i]
