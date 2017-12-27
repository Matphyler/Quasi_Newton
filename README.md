# Quasi Newton Optimization 

## Usage

To use BFGS/LBFGS optimizer, use
```python
from BFGS_Optimizer import *
from LBFGS_Optimizer import *
```

### ObjFun class
To create an objective function: 
```python
def _f(x):
    return x[0]**2 + x[1]**2

f = ObjFun(value_fun=_f, input_dim=2)
```
The gradient will be automatically computed using the package [autograd](https://github.com/HIPS/autograd). Alternatively, a gradient function can be specified manually:

```python
def _f(x):
    return x[0]**2 + .5*x[1]**2

def _g(x):
    return [2*x[0], x[1]]
    
f = ObjFun(value_fun=_f, gradient_fun=_g, input_dim=2)
```

To call an ObjFun:
```python
>>> f(order=0, x=[1, 2])
3.0
>>> f(order=1, x=[1, 2])
array([ 2.,  2.])
```

The ObjFun class will also record the times a function is called and total time used.
```python
>>> f.func_call_stats()
{'gradient': {'time_elapsed': 0.0006029605865478516, 'times': 1},
 'value': {'time_elapsed': 1.0967254638671875e-05, 'times': 1}}
```

### Optimizer
To use the optimizer, first create an optimizer object with options:
```python
opt = LBFGSOptimizer(line_search_c1=0.01, line_search_c2=0.5)
```
To display current options,
```python
>>> opt.option
{'display_level': 1,
 'hessian_inverse_init': 'Identity',
 'hessian_inverse_init_fun': <function base_functions.random_matrix_wishart>,
 'is_debug_mode': False,
 'is_line_search_check_differentiability': True,
 'is_stat_kept': True,
 'is_strong_wolfe': False,
 'line_search_c1': 0.01,
 'line_search_c2': 0.5,
 'line_search_init': 1,
 'max_line_search_iterations': 50,
 'max_optimization_iterations': 1000,
 'memory_length': 10,
 'singular_threshold': 1e-10,
 'tolerance': 1e-10,
 'x_init': <function base_functions.random_vec_normal>}
```

To initialize the optimizer, provide an ObjFun as objective function, and an initial starting point (optional):
```python
opt.initialize(obj_fun=f, x_init=[3.5, 2.3])
```

To run the optimizer, use
```python
>>> opt.run()
0
```
The output **0** is the status code with the following meaning:
-  **-2**: not initialized
-  **-1**: not started
-  **0**: success
-  **1**: max iteration exceeded
-  **2**:  line search failed

To see the summary, use:
```python
>>> opt.summary
{'f_fin': 4.6412459093013654e-27,
 'f_init': 14.895,
 'g_fin': array([ -2.93501347e-14,   9.40838807e-14]),
 'g_init': array([ 7. ,  2.3]),
 'status': 0,
 'total_iterations': 6,
 'x_fin': array([ -1.46750673e-14,   9.40838807e-14]),
 'x_init': array([ 3.5,  2.3])}
```

To see a specific iterate generated, use:
```python
>>> opt.stat[0]
{'a': 1,
 'f': 14.895,
 'g': array([ 7. ,  2.3]),
 'iter': 0,
 'ls_flag': True,
 'ls_num': 1,
 'p': array([-7. , -2.3]),
 'x': array([ 3.5,  2.3])}
```
To see the sequence generated, use, for example
```python
>>> opt.stat('f','g')
[[14.895, array([ 7. ,  2.3])],
 [12.25, array([-7.,  0.])],
 [0.0016456151332890179, array([ 0.01836085, -0.05588085])],
 [3.6277332982177716e-06, array([  4.28906352e-05,   2.69342659e-03])],
 [1.1563412150434447e-12, array([ -2.09993533e-06,  -3.28356865e-07])],
 [6.1048932884115176e-18, array([  4.94127187e-09,  -4.12640803e-11])]]
```

## Non-smooth optimization

Non-smooth functions are supported. For example, consider
```python
def _f(x):
    return 10 * my_abs(x[1]-x[0]**2) + (1-x[0])**2

f = ObjFun(value_fun=_f, input_dim=2)

opt.initialize(obj_fun=f)
```

The optimizer cannot automatically stop with status code **0** because a suitable stopping criterion for non-smooth problems is not provided. However, it would stop when the line search fails (with status code **2**) or when maximum number of iterations are reached (with status code **1**). When it does, it usually gives a stationary point:
```python
>>> opt.run()
2
>>> opt.summary
{'f_fin': 2.3626248999960446e-14,
 'f_init': 105.75,
 'g_fin': array([ 19.99999678, -10.        ]),
 'g_init': array([ 75., -10.]),
 'status': 2,
 'total_iterations': 314,
 'x_fin': array([ 0.99999985,  0.99999971]),
 'x_init': array([ 3.5,  2.3])}
```

