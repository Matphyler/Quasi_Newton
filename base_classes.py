import autograd.numpy as np
from autograd import grad
import time
import copy


class ObjFun:
    """
    Class for objective functions
    """
    def __init__(self, input_dim=1, value_fun=None, gradient_fun=None,
                 auto_def=1, gradient_check_delta=1E-7, gradient_check_tolerance=1E-7
                 ):

        value_fun_loc = copy.copy(value_fun)
        gradient_fun_loc = copy.copy(gradient_fun)
        self.input_dim = input_dim  # dimension of input to the objective function

        # self.is_smooth = is_smooth  # indicating whether function is smooth everywhere

        # evaluation times and time elapsed counter
        # only counts if calls by using __call__ method;
        # directly calling value_fun, gradient_fun, etc. does not count

        # self.is_differentiable_evaluation_times = 0
        # self.is_differentiable_evaluation_time_elapsed = 0
        self.value_evaluation_times = 0
        self.value_evaluation_time_elapsed = 0
        self.gradient_evaluation_times = 0
        self.gradient_evaluation_time_elapsed = 0
        # self.hessian_evaluation_times = 0
        # self.hessian_evaluation_time_elapsed = 0

        # function value_fun
        if value_fun_loc is None:
            def default_value(x):
                return 0

            self.value = default_value
            self.is_value_defined = False

        else:

            def value_func_wrapper(x):
                # x_loc = np.asarray(x, dtype=np.float64)
                x_loc = x
                return value_fun_loc(x_loc)

            self.value = value_func_wrapper
            self.is_value_defined = True

        # gradient_fun
        self.is_gradient_defined = True
        if gradient_fun_loc is None:
            if auto_def == 1:  # use autograd to compute
                self.gradient = grad(self.value)
            # elif auto_def == 2:  # use finite difference
            #     # to be implemented
            #     pass
            else:  # return default gradient_fun 0
                def default_gradient(x):
                    return np.zeros(input_dim)

                self.gradient = default_gradient
                self.is_gradient_defined = False
        else:
            self.gradient = gradient_fun_loc

        # Hessian
        # if hessian_fun is None:
        #     self.is_hessian_defined = True
        #     if auto_def == 1:  # use autograd to compute
        #         auto_hessian = hessian(value_fun)
        #         # auto_hessian = float_ndarray_input(auto_hessian)
        #         self.hessian = auto_hessian
        #
        #     # elif auto_def == 2:  # use finite difference
        #     #     # to be implemented
        #     #     pass
        #
        #     else:
        #         def default_hessian(x):
        #             return np.zeros((input_dim, input_dim))
        #
        #         self.hessian = default_hessian
        #         self.is_hessian_defined = False
        # else:
        #     self.hessian = hessian_fun
        #     self.is_hessian_defined = True

        # is_differentiable
        # if is_differentiable is None:
        #     def default_is_differentiable(x):
        #         return True
        #
        #     self.is_differentiable = default_is_differentiable
        #     self.is_differentiable_defined = False
        # else:
        #     self.is_differentiable = is_differentiable
        #     self.is_differentiable_defined = True

        # other parameters
        self.gradient_check_delta = gradient_check_delta
        self.gradient_check_tolerance = gradient_check_tolerance

    def __call__(self, order, x):

        # cast the input to be a float-typed ndarray
        # this is important for hessian to work properly
        x_var = np.asarray(x, dtype=np.float64)

        # dimension check
        assert np.shape(x_var) == (self.input_dim,), \
            "input shape %r does not match pre-defined input shape %r" \
            % (np.shape(x_var), self.input_dim)  # check the input dimension

        start_time = time.time()
        if order == 0:  # return function value
            value = self.value(x_var)
            self.value_evaluation_times += 1
            self.value_evaluation_time_elapsed += time.time() - start_time
            return value
        elif order == 1:  # return gradient
            gradient = self.gradient(x_var)
            self.gradient_evaluation_times += 1
            self.gradient_evaluation_time_elapsed += time.time() - start_time
            return gradient

        # elif order == 2:  # return
        #     self.hessian_evaluation_times += 1
        #     self.hessian_evaluation_time_elapsed += time.time() - start_time
        #     return self.hessian(x_var)
        # elif order == -1:
        #     self.is_differentiable_evaluation_times += 1
        #     self.is_differentiable_evaluation_time_elapsed += time.time() - start_time
        #     return self.is_differentiable(x_var)
        else:
            raise ValueError('order can only be 0, 1')

    def check(self, total_check_times=1, gradient_check_times_per_iteration=1, random_point_mu=0, random_point_sigma=1):
        # note: check does not call self; therefore does not increase call times and time elapsed.
        for iter_1 in range(total_check_times):
            test_x = np.random.normal(size=self.input_dim, loc=random_point_mu, scale=random_point_sigma)
            test_f = self.value(test_x)
            test_g = self.gradient(test_x)

            # check gradient dimension
            assert np.shape(test_g)[0] == self.input_dim, \
                "gradient dimension %r mismatches with input dimension %r" \
                % (np.shape(test_g)[0], self.input_dim)

            # check gradient value using finite difference
            if gradient_check_times_per_iteration > 0:
                for iter_2 in range(gradient_check_times_per_iteration):
                    test_dx = np.random.normal(size=self.input_dim)
                    test_dx = self.gradient_check_delta * test_dx / np.linalg.norm(
                        test_dx)
                    test_y = test_x + test_dx
                    test_df = self.value(test_y) - test_f
                    assert np.abs(test_df - np.dot(test_dx, test_g)) < \
                           self.gradient_check_tolerance, \
                        "gradient value test fails to pass"

        return 0

    def initialize_counter(self):
        # self.is_differentiable_evaluation_times = 0
        # self.is_differentiable_evaluation_time_elapsed = 0
        self.value_evaluation_times = 0
        self.value_evaluation_time_elapsed = 0
        self.gradient_evaluation_times = 0
        self.gradient_evaluation_time_elapsed = 0
        # self.hessian_evaluation_times = 0
        # self.hessian_evaluation_time_elapsed = 0
        return 0

    def func_call_stats(self):
        return {"value": {"times": self.value_evaluation_times, "time_elapsed": self.value_evaluation_time_elapsed},
                "gradient": {"times": self.gradient_evaluation_times,
                             "time_elapsed": self.gradient_evaluation_time_elapsed}
                }


class DataSet:
    """
    A class which is a list of dictionaries.
    """

    def __init__(self, container=None):
        if container is None:
            self.container = list()
        else:
            self.container = copy(container)

    def log(self, entry):
        assert isinstance(self.container, list)
        assert isinstance(entry, dict)  # entry must be a dict
        self.container.append(entry)

    def __len__(self):
        return len(self.container)

    def __call__(self, keys):
        if isinstance(keys, str):
            return [x[keys] for x in self.container]

        if len(keys) == 1:
            return [x[keys[0]] for x in self.container]
        else:
            return [[x[k] for k in keys] for x in self.container]

    def __getitem__(self, item):
        return self.container.__getitem__(item)


class Optimizer:

    default_option = dict()

    def set_opt(self, option, print_change=False):
        for (key, value) in option.items():
            if print_change:
                print('Option \'%s\' set from %r to %r' % (key, self.option[key], value))
            self.option[key] = value

    def __init__(self, option=None):

        self.option = self.default_option
        if option is not None:
            for (key, value) in option.items():
                self.option[key] = value

        self.stat = DataSet()
        self.summary = None
        self.obj_fun = None
        self.dim = None
        self.x_init = None

        self.flag = -2
        ##################################
        #  value for flag:
        #   -2 => not initialized
        #   -1 => not started
        #    0 => success
        #    1 => max iteration exceeded
        #    2 => line search failed
        ##################################

    def initialize(self, obj_fun=None, x_init=None, keep_last=True):

        """
        Initialization.

            (1) self.stat: will ALWAYS be cleared.

            (2) self.obj_fun: if provided, will ALWAYS update; if not provided, will try to use the last obj_fun;
            if the last obj_fun is not defined, will raise an error. The obj_func's counter will be initialized.

            (3) self.x_init and self.h_init: if provided, will ALWAYS update; if not provided: if keep_last is True,
            then the last one will be used; otherwise will initialize as specified in option.

        """

        self.stat = DataSet()  # self.stat will always be emptied
        self.summary = None

        if obj_fun is not None:  # a new obj_fun is provided, update; keep_last will be ignored!
            self.obj_fun = obj_fun
        elif self.obj_fun is not None:  # use last obj_func; no change
            pass
        else:  # do not have a valid obj_fun, error
            raise ValueError('Objective function is not provided')

        self.obj_fun.initialize_counter()  # initialize the counter for obj_fun

        self.dim = self.obj_fun.input_dim

        if x_init is not None:  # a new init point provided, update; keep_last will be ignored!
            self.x_init = x_init
        elif not keep_last or self.x_init is None:  # use initialization specified in option
            self.x_init = self.option['x_init'](self.dim)
        else:  # use last x_init; no change
            pass

        self.flag = -1  # indicate that initialization is completed

    def __getitem__(self, item):
        return self.stat[item]

    def __call__(self, *args, **kwargs):
        return self.stat(*args, **kwargs)