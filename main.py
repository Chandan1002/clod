

@cuda.jit(device=True)
def custom_sqrt(x):
  if x==0:
    return 0
  z=x
  for i in range(10):
    z= 0.5 * (z+x/z)
  return z

@cuda.jit(device=True)
def hypot(x,y):
  return custom_sqrt(x*x +y*y)

@cuda.jit(device=True)
def vector_subtract(a,b, result):
  for i in range(len(a)):
    result[i]=a[i]-b[i]

