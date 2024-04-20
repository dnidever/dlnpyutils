import numpy as np
from . import utils as dln

# Create distributions

# The basic idea is the following:
# 1) Create your distribution
# 2) Create a cumulative version and normalize it so the
#      values go from 0 to 1
# 3) Draw random uniformly distributed values from 0 to 1
# 4) Use distribution from #2 to interpolate the values
#     from #3 back to the ordinate value (abcissa)
# The end result should follow the correct distribution

# 1-D distributions

def sqsech(num,h=1.0,maxval=100.0):
    """
    Squared hyperbolic secant: sech^2 (0.5*z/hz)
    """
    # 1) Create squared sech distribution
    # sech() = 1/cosh()
    r = np.linspace(0,maxval,1000)
    y = 1/np.cosh(r/h)**2
    # 2) Create normalized cumulative gersion
    cumy = np.cumsum(y)
    cumy /= np.max(cumy)  # normalize
    # 3) Draw random uniformly disributed values
    rnd = np.random.rand(num)
    # 4) Use distribution from #2 to interpolate the values
    #     from #3 back to the ordinate value (abcissa)
    out = dln.interp(cumy,r,rnd,assume_sorted=False)
    return out

def exp(num,h=1.0,maxval=100.0):
    """
    Exponential distribution: exp(-R/h)
    """
    rnd = np.random.rand(num)
    # 1) Create exponential distribution
    r = np.linspace(0,maxval,1000)
    y = np.exp(-r/h)
    # 2) Create normalized cumulative gersion
    cumy = np.cumsum(y)
    cumy /= np.max(cumy)  # normalize
    # 3) Draw random uniformly disributed values
    rnd = np.random.rand(num)
    # 4) Use distribution from #2 to interpolate the values
    #     from #3 back to the ordinate value (abcissa)
    out = dln.interp(cumy,r,rnd,assume_sorted=False)
    return out
