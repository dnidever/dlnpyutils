import numpy as np
import inspect
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

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class Distribution(object):

    def __init__(self,kind,seed=None,**kwargs):
        if kind not in _distributions.keys():
            raise ValueError(str(kind)+' not supporteed')

        self.kind = kind
        self._input_kwargs = kwargs        
        self._func = _distributions[kind]['func']
        self._precompute = _distributions[kind]['precompute']
        self._def_kwargs = get_default_args(self._func)
        if '_state' in self._def_kwargs.keys():
            self._def_kwargs.pop('_state')
        # Get all arguments for this function (use inputs and defaults)
        allkwargs = self._def_kwargs.copy()
        for k in kwargs.keys():
            if k in allkwargs.keys():
                allkwargs[k] = kwargs[k]
        self._original_allkwargs = allkwargs
        # Now get function parameters
        #   do not include any arguments that start with '_'
        args = {k:v for k,v in self._original_allkwargs.items()
                if k.startswith('_')==False}
        self._original_kwargs = args
        # Make the arguments properties
        for k,v in self._original_kwargs.items():
            setattr(self,k,v)
        # Compute the function
        if self._precompute:
            distr = self._computefunction(**self._getallkwargs())
            self._functionvals = distr
            #  save the arguments used to compute this function
            self._functionvals_allkwargs = self._getallkwargs()
        # Initiate RandomState()
        self.state = np.random.RandomState(seed)

    def _getallkwargs(self):
        """ Get all the arguments."""
        # The user might have changed them
        # Start with the original ones and update them
        kwargs = self._original_allkwargs.copy()
        for k in kwargs.keys():
            if hasattr(self,k) and k != '_state':
                kwargs[k] = getattr(self,k)
        return kwargs

    def _getkwargs(self):
        """ Get all parameters (without the internal '_' keywords)."""
        # Start with all and then only keep the parameters
        #  that do not start with '_'
        allkwargs = self._getallkwargs()
        kwargs = {}
        for k,v in allkwargs.items():
            if k.startswith('_')==False:
                kwargs[k] = v
        return kwargs    
        
    def __call__(self,num):
        # Get current args
        curallkwargs = self._getallkwargs()

        # Precomputed values:
        if self._precompute:
            # if the arguments changed since the function was
            # last computed, then we must recompute the function
            if curallkwargs != self._functionvals_allkwargs:
                # Compute the function
                distr = self._computefunction(**curallkwargs)
                self._functionvals = distr
                #  save the arguments used to compute the function
                self._functionvals_allkwargs = curallkwargs.copy()
            # Return the distribution
            return self._getdistribution(num)
        
        # Direct distributions
        else:
            return self._func(num,_state=self.state,**curallkwargs)
    
    def _computefunction(self,**kwargs):
        """ Compute the function."""
        distr = self._func(**kwargs)
        distr['cumy'] = np.cumsum(distr['y'])
        distr['cumy'] /= np.max(distr['cumy'])
        return distr

    def _getdistribution(self,num):
        """ Get the distribution or a number of points."""
        # 1) Draw random uniformly disributed values
        rnd = self.state.rand(num)
        # 2) Use normalized cumulative distribution to
        #     interpolate the value from #1 back to the
        #     abcissa (x) value.
        x = dln.interp(self._functionvals['cumy'],self._functionvals['x'],
                       rnd,assume_sorted=False)
        return x

    def __repr__(self):
        """ Print out the string representation """
        s = repr(self.__class__)
        s = s.replace("class '","class ")
        s = s[:-2]
        s += '(kind='+str(self.kind)
        for k,v in self._getkwargs().items():
            s += ',{:}={:.4f}'.format(k,v)
        s += ')>\n'
        return s
        
# 1-D distributions

# ---- Precomputed functions ----

def sqsech(h=1.0,_minval=0.0,_maxval=100.0,_n=1000):
    """
    Squared hyperbolic secant: sech^2 (0.5*x/h)
    """
    # Create squared sech distribution
    # sech() = 1/cosh()
    x = np.linspace(0,_maxval,_n)
    y = 1/np.cosh(x/h)**2
    return {'x':x,'y':y}

def exp(h=1.0,_minval=0.0,_maxval=100.0,_n=1000):
    """
    Exponential distribution: exp(-x/h)
    """
    # Create exponential distribution
    x = np.linspace(0,_maxval,_n)
    y = np.exp(-x/h)
    return {'x':x,'y':y}

# --- Direct distributions ---

def normal(size,mu=0.0,sigma=1.0,_state=None):
    """
    Normal/Gaussian distribution.
    """
    if _state is None:
        _state = np.random.RandomState()
    return _state.normal(mu,sigma,size)

def poisson(size,lam=1.0,_state=None):
    """
    Poisson distribution.
    """
    if _state is None:
        _state = np.random.RandomState()
    return _state.poisson(lam,size)

# np.random.RandomState() has it's own distribution functions
# exp, normal, poisson, binomial, etc.

#---------- old -------------

def sqsech2(num,h=1.0,maxval=100.0):
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

def exp2(num,h=1.0,maxval=100.0):
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

# List of distributions
_distributions = {'sqsech':{'func':sqsech,'precompute':True},
                  'exp':{'func':exp,'precompute':True},
                  'normal':{'func':normal,'precompute':False},
                  'poisson':{'func':poisson,'precompute':False}}
