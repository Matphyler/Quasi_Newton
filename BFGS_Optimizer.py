from base_functions import *
from base_classes import *


class BFGSOptimizer:
    default_option = {'max_optimization_iterations': 1000, 'tolerance': 1E-10,
                      'hessian_inverse_init': random_matrix_wishart, 'x_init': random_vec_normal,
                      'line_search_init': 1, 'line_search_c1': .01, 'line_search_c2': .5, 'singular_threshold': 1E-10,
                      'max_line_search_iterations': 50, 'is_strong_wolfe': False,
                      'is_line_search_check_differentiability': True, 'display_level': 1, 'is_stat_kept': True,
                      'is_debug_mode': False}

    def hessian_inverse_update(self, h, s, y):
        h_var = np.asarray(h, dtype=np.float64)
        dim = np.shape(h_var)[0]
        rho_inv = np.dot(s, y)  # inverse of rho, i.e., dot(s, y)

        assert rho_inv / (np.linalg.norm(s) * np.linalg.norm(y)) > self.option['singular_threshold'], \
            "singularity encountered!"

        v = np.identity(dim) - 1 / rho_inv * np.outer(s, y)
        return np.dot(np.dot(v, h), v.transpose()) + 1 / rho_inv * np.outer(s, s)

    def wolfe_line_search(self, x, f, g, p):

        a = self.option['line_search_init']
        lower = 0
        upper = np.inf
        ls_num = 0
        ls_flag = False
        c_1 = self.option['line_search_c1']
        c_2 = self.option['line_search_c2']

        # main loop
        while ls_num < self.option['max_line_search_iterations']:

            ls_num += 1

            x_new = x + a * p
            f_new = self.obj_fun(0, x_new)
            g_new = self.obj_fun(1, x_new)

            if f_new >= f + c_1 * a * np.dot(g, p):
                upper = a
            elif np.dot(g_new, p) <= c_2 * np.dot(g, p):
                lower = a
            else:
                ls_flag = True
                break
            if upper < np.inf:
                a = (lower + upper) / 2
            else:
                a = 2 * lower

        return ls_flag, a, x_new, f_new, g_new, ls_num

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
        self.h_init = None

        self.flag = -2
        ##################################
        #  value for flag:
        #   -2 => not initialized
        #   -1 => not started
        #    0 => success
        #    1 => max iteration exceeded
        #    2 => line search failed
        ##################################

    def initialize(self, obj_fun=None, x_init=None, h_init=None, keep_last=True):

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

        if h_init is not None:  # a new h_init is provided, update; keep_last will be ignored!
            self.h_init = h_init
        elif not keep_last or self.h_init is None:  # use initialization specified in option
            self.h_init = self.option['hessian_inverse_init'](self.dim)
        else:  # use last h_init; no change
            pass

        self.flag = -1  # indicate that initialization is completed

    def run(self):

        if self.flag is not -1:
            return self.flag

        iteration = 0

        self.flag = 0

        x_last = np.copy(self.x_init)
        f_last = self.obj_fun(order=0, x=x_last)
        g_last = self.obj_fun(order=1, x=x_last)
        h = np.copy(self.h_init)

        x = x_last
        f = f_last
        g = g_last

        ########################
        #     Main Loop        #
        ########################

        while True:

            if iteration > self.option['max_optimization_iterations']:
                self.flag = 1
                break

            if self.option['tolerance'] is not None and np.linalg.norm(g_last) <= self.option['tolerance']:
                self.flag = 0
                break

            p = - np.dot(h, g_last)  # search direction p_k

            ls_flag, a, x, f, g, ls_num = self.wolfe_line_search(x=x_last, f=f_last, g=g_last, p=p)

            if self.option['is_stat_kept']:
                self.stat.log({"iter": iteration, "x": np.copy(x_last),
                               "f": f_last,
                               "g": np.copy(g_last),
                               "h": np.copy(h),
                               "p": np.copy(p),
                               "a": a,
                               "ls_num": ls_num,
                               "ls_flag": ls_flag})

            if not ls_flag:
                self.flag = 2
                break

            s = x - x_last
            y = g - g_last
            h = self.hessian_inverse_update(h=h, s=s, y=y)

            x_last = x
            f_last = f
            g_last = g

            iteration += 1

        self.summary = {'total_iterations': iteration, 'status': self.flag, 'x_fin': x, 'f_fin': f, 'g_fin': np.copy(g)}

        return self.flag

    def __getitem__(self, item):
        return self.stat[item]

    def __call__(self, *args, **kwargs):
        return self.stat(*args, **kwargs)