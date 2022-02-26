import warnings
import numpy as np
from GPy.models import GPRegression
from GPy.kern import Matern32
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
from scipy.special import erfc
from scipy.stats import norm


class OptimisationResult(object):
    def __init__(self, other=None):
        if other is not None:
            for k, v in other.__dict__.items():
                self.__dict__[k] = v.copy()

    def resize(self, new_size):
        for k, v in self.__dict__.items():
            if k == 'par_descr':
                continue
            try:
                v.resize((new_size, v.shape[1]))
            except ValueError:
                self.__dict__[k] = v.copy()
                self.__dict__[k].resize((new_size, v.shape[1]))


def validate_par_descr(par_descr):
    assert len(par_descr) > 0
    for par_name, (ptype, prange) in par_descr.items():
        assert ptype in ['categorical', 'integer', 'continuous']
        if ptype == 'categorical':
            assert len(prange) > 1
        elif ptype == 'integer':
            assert prange[0] < prange[1]
            assert prange[0] == int(prange[0]) and prange[1] == int(prange[1])
        elif ptype == 'continuous':
            assert prange[0] < prange[1]
            assert prange[0] == float(prange[0]) and prange[1] == float(prange[1])
    return True


def random_samples(par_descr, n_samples):
    validate_par_descr(par_descr)
    
    samples = []
    for par_name, (ptype, prange) in sorted(par_descr.items()):
        if ptype == 'categorical' and len(prange) == 2:
            samples.append(np.random.randint(2, size=n_samples))
        elif ptype == 'categorical' and len(prange) > 2:
            # One-hot encoding
            rand = np.random.randint(len(prange), size=n_samples)
            for row in np.eye(len(prange))[rand].T:
                samples.append(row)
        elif ptype == 'integer':
            samples.append(np.random.randint(prange[0], prange[1] + 1, size=n_samples))
        elif ptype == 'continuous':
            samples.append(np.random.uniform(prange[0], prange[1], size=n_samples))
    return np.array(samples).T


def random_search_internal(fitness_function, par_descr, max_f_eval=1000):
    '''Maximise a function.'''
    opt_res = OptimisationResult()
    opt_res.x = random_samples(par_descr, max_f_eval)
    MAX_CHUNK_SIZE = 2000
    if max_f_eval <= MAX_CHUNK_SIZE:
        opt_res.f = fitness_function(opt_res.x)
    else:
        x_split = np.array_split(opt_res.x, np.ceil(1. * max_f_eval / MAX_CHUNK_SIZE))
        opt_res.f = fitness_function(x_split[0])
        opt_res.f.resize((max_f_eval, opt_res.f.shape[1]))#, refcheck=False)
        idx = x_split[0].shape[0]
        for x_chunk in x_split[1:]:
            chunk_size = x_chunk.shape[0]
            opt_res.f[idx:idx+chunk_size, :] = fitness_function(x_chunk)
            idx += chunk_size
        
    return opt_res


def mutate(population, par_descr, s_cat=.2, s_cont=.1, s_int=.1):
    '''Mutate population in place.'''
    for i in range(population.shape[0]):
        j = 0
        for par_name, (ptype, prange) in sorted(par_descr.items()):
            if ptype == 'categorical' and len(prange) == 2:
                if np.random.rand() < s_cat:
                    population[i, j] = np.random.randint(2)
                j += 1
            elif ptype == 'categorical' and len(prange) > 2:
                if np.random.rand() < s_cat:
                    #old_val = np.argmax(population[i, j:j+len(prange)])
                    population[i, j:j+len(prange)] = 0
                    new_val = np.random.randint(len(prange))
                    population[i, j+new_val] = 1
                j += len(prange)
            elif ptype == 'integer':
                old_val = population[i, j]
                xmin, xmax = prange
                skellam_mu = .5 * (s_int * (xmax - xmin)) ** 2
                new_val = old_val
                while True:
                    rand = np.random.poisson(lam=skellam_mu, size=2)
                    new_val = old_val + rand[0] - rand[1]
                    if new_val >= xmin and new_val <= xmax:
                        break
                population[i, j] = new_val
                j += 1
            elif ptype == 'continuous':
                old_val = population[i, j]
                xmin, xmax = prange
                norm_var = s_cont * (xmax - xmin)
                norm_scale = np.sqrt(norm_var)
                new_val = old_val
                while True:
                    new_val = old_val + np.random.normal(scale=norm_scale)
                    if new_val >= xmin and new_val <= xmax:
                        break
                population[i, j] = new_val
                j += 1


def mies_internal(fitness_function, par_descr, max_iter=1000, mu=1,
    lambda_=32, n_starts=1, plus_selection=True, sigma_cat=.4, sigma_cont=.2,
    sigma_int=.2, decay_cat=0.05, decay_cont=0.01, decay_int=0.01,
    initial_population=None):
    '''Maximise a function using a Mixed-Integer Evolution Strategy.'''
    evals_per_iter = n_starts * lambda_

    opt_res = OptimisationResult()
    if initial_population is not None:
        opt_res.x = initial_population
        opt_res.f = fitness_function(opt_res.x)
    else:
        random_samples = random_search_internal(fitness_function, par_descr, 10000)
        idx = np.argsort(random_samples.f.ravel())[-evals_per_iter:]
        opt_res.x = random_samples.x[idx, :]
        opt_res.f = random_samples.f[idx, :]

    children = []
    children_fitness = []
    for chain_no in range(n_starts):
        begin = chain_no * lambda_
        end = (chain_no + 1) * lambda_
        children.append(opt_res.x[begin:end, :])
        children_fitness.append(opt_res.f[begin:end, :])

    opt_res.resize(max_iter * evals_per_iter)

    cur_index = evals_per_iter
    for iter_no in range(1, max_iter):
        parents = []
        parents_fitness = []
        for chain_no in range(n_starts):
            parents_idx = np.argsort(children_fitness[chain_no].ravel())[-mu:]
            parents.append(children[chain_no][parents_idx, :])
            parents_fitness.append(children_fitness[chain_no][parents_idx, :])

        s_cat = sigma_cat * decay_cat ** (1. * iter_no / max_iter)
        s_cont = sigma_cont * decay_cont ** (1. * iter_no / max_iter)
        s_int = sigma_int * decay_int ** (1. * iter_no / max_iter)
        children = []
        for chain_no in range(n_starts):
            children_idx = np.random.choice(mu, lambda_)
            children_ = parents[chain_no][children_idx, :]
            mutate(children_, par_descr, s_cat=s_cat, s_cont=s_cont, s_int=s_int)
            children.append(children_)

        children_fitness = fitness_function(np.vstack(children))
        children_fitness = np.vsplit(children_fitness, n_starts)
        #children_fitness = []
        for chain_no in range(n_starts):
            #children_fitness_ = fitness_function(children[chain_no])
            #children_fitness.append(children_fitness_)
            opt_res.x[cur_index:cur_index+lambda_, :] = children[chain_no]
            #opt_res.f[cur_index:cur_index+lambda_, :] = children_fitness_
            opt_res.f[cur_index:cur_index+lambda_, :] = children_fitness[chain_no]
            cur_index += lambda_

        if plus_selection:
            for chain_no in range(n_starts):
                children[chain_no] = np.vstack([parents[chain_no], children[chain_no]])
                children_fitness[chain_no] = np.vstack([parents_fitness[chain_no], children_fitness[chain_no]])

    return opt_res


def lbfgsb(fitness_func_nograd, fitness_function_with_grad, par_descr):
    random_samples = random_search_internal(fitness_func_nograd, par_descr, 10000)
    idx = np.argsort(random_samples.f.ravel())[-100:]
    random_samples.x = random_samples.x[idx, :]
    random_samples.f = random_samples.f[idx, :]

    categorical_mask = []
    integer_mask = []
    bounds = []
    j = 0
    for par_name, (ptype, prange) in sorted(par_descr.items()):
        if ptype == 'categorical' and len(prange) == 2:
            categorical_mask.append(j)
            bounds.append((0, 1))
            j += 1
        elif ptype == 'categorical' and len(prange) > 2:
            categorical_mask.extend(range(j, j+len(prange)))
            bounds.extend([(0, 1)] * len(prange))
            j += len(prange)
        elif ptype == 'integer':
            integer_mask.append(j)
            bounds.append(prange)
            j += 1
        elif ptype == 'continuous':
            bounds.append(prange)
            j += 1
        else:
            raise RuntimeError('Unknown ptype {0}'.format(ptype))
    bounds_min = [b[0] for b in bounds]
    bounds_max = [b[1] for b in bounds]

    results_x = []
    results_f = []
    for x0 in random_samples.x:
        x0_cat_vals = x0[categorical_mask]
        def fitness_wrapper(x):
            x = x.copy()
            x = np.clip(x, bounds_min, bounds_max)
            x[categorical_mask] = x0_cat_vals
            res, grad = fitness_function_with_grad(x.reshape(1, -1))
            grad[:, categorical_mask] = 0
            return -res.ravel()[0], -grad.ravel()
        res = minimize(fitness_wrapper, x0, method='L-BFGS-B', jac=True, bounds=bounds)
        res.x = np.clip(res.x, bounds_min, bounds_max)
        res.x[categorical_mask] = x0_cat_vals
        res.x[integer_mask] = np.round(res.x[integer_mask])

        results_x.append(res.x)
        #results_f.append(res.fun)
        results_f.append(fitness_func_nograd(res.x.reshape(1, -1)).ravel()[0])
    opt_res = OptimisationResult()
    opt_res.x = np.asarray(results_x)
    opt_res.f = np.asarray(results_f).reshape(-1, 1)
    return opt_res


def bound_sample(x, par_descr):
    x = x.copy()
    j = 0
    for par_name, (ptype, prange) in sorted(par_descr.items()):
        if ptype == 'categorical' and len(prange) == 2:
            x[:, j] = np.clip(x[:, j], 0, 1)
            j += 1
        elif ptype == 'categorical' and len(prange) > 2:
            # Set the minimum value to be 1e-12 to avoid potential divide-by-zero errors
            x[:, j:j+len(prange)] = np.clip(x[:, j:j+len(prange)], 1e-12, None)
            s = np.sum(x[:, j:j+len(prange)], axis=1).reshape(-1, 1)
            x[:, j:j+len(prange)] = x[:, j:j+len(prange)] / s
            j += len(prange)
        elif ptype == 'integer':
            x[:, j] = np.clip(x[:, j], prange[0], prange[1])
            j += 1
        elif ptype == 'continuous':
            x[:, j] = np.clip(x[:, j], prange[0], prange[1])
            j += 1
        else:
            raise RuntimeError('Unknown ptype {0}'.format(ptype))
    return x


def draw_from_sample(x, par_descr, n_draws=10):
    assert len(x.shape) == 1 or (len(x.shape) == 2 and x.shape[0] == 1)
    x = x.reshape(1, -1)
    x_new = np.zeros((n_draws, x.shape[1]))
    j = 0
    for par_name, (ptype, prange) in sorted(par_descr.items()):
        if ptype == 'categorical' and len(prange) == 2:
            p1 = x[0, j]
            x_new[:, j] = np.random.choice(2, size=n_draws, p=[1-p1, p1])
            j += 1
        elif ptype == 'categorical' and len(prange) > 2:
            p = x[0, j:j+len(prange)]
            draws = np.random.choice(len(prange), size=n_draws, p=p)
            x_new[range(n_draws), j+draws] = 1
            j += len(prange)
        elif ptype == 'integer':
            p1 = x[0, j] % 1
            x_new[:, j] = np.floor(x[0, j]) + np.random.choice(2, size=n_draws, p=[1-p1, p1])
            j += 1
        elif ptype == 'continuous':
            x_new[:, j] = x[0, j]
            j += 1
        else:
            raise RuntimeError('Unknown ptype {0}'.format(ptype))
    return x_new


def sgd(fitness_func_nograd, fitness_function_with_grad, par_descr):
    random_samples = random_search_internal(fitness_func_nograd, par_descr, 10000)
    idx = np.argsort(random_samples.f.ravel())[-1:]
    random_samples.x = random_samples.x[idx, :]
    random_samples.f = random_samples.f[idx, :]

    results_x = []
    results_f = []
    max_iter = 1000
    learning_rate_initial = 0.3
    learning_rate_final = 0.001
    lr_list = np.geomspace(learning_rate_initial, learning_rate_final, max_iter)
    if all(ptype == 'continuous' for ptype, prange in par_descr.values()):
        num_samples_per_draw = 1
    else:
        num_samples_per_draw = 32

    for x0 in random_samples.x:
        x_virt = x0.reshape(1, -1)
        for iter_no in range(max_iter):
            learning_rate = lr_list[iter_no]
            samples = draw_from_sample(x_virt, par_descr, num_samples_per_draw)
            ei, grad = fitness_function_with_grad(samples)
            results_x.extend(samples)
            results_f.extend(ei)
            grad = np.mean(grad, axis=0).reshape(1, -1)
            # Adding the gradient for maximisation of EI
            x_virt += learning_rate * grad
            x_virt = bound_sample(x_virt, par_descr)
    opt_res = OptimisationResult()
    opt_res.x = np.asarray(results_x)
    opt_res.f = np.asarray(results_f).reshape(-1, 1)
    return opt_res


class SearchSpaceModel(object):
    def __init__(self, noise=True):
        self.noise = noise

    def fit(self, X, Y):
        self.Y_scaler = StandardScaler()
        Y = self.Y_scaler.fit_transform(Y)
        self.X_scaler = StandardScaler()
        X = self.X_scaler.fit_transform(X)

        self.fit_gp(X, Y)
        self.Ymin = self.Y_scaler.inverse_transform(self.Ymin.reshape(1, -1))

    def fit_gp(self, X, Y):
        kern = Matern32(X.shape[1], ARD=True)
        gp_model = GPRegression(X, Y, kernel=kern)
        if not self.noise:
            gp_model.Gaussian_noise.variance.fix(0)
        gp_model.kern.lengthscale.constrain_bounded(1e-2, 1e2, warning=False)

        for i in range(30):
            try:
                gp_model.randomize()
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    gp_model.optimize()
                if self.noise:
                    Y_pred = gp_model.predict(X)[0]
                    noiseless_gp_model = GPRegression(X, Y_pred, kernel=gp_model.kern)
                    noiseless_gp_model.kern.variance.fix(0)
            except np.linalg.LinAlgError as error:
                continue
            else:
                break
        else:
            raise error

        if self.noise:
            self.noisy_gp = gp_model
            self.gp = noiseless_gp_model
            self.Ymin = np.min(Y_pred)
        else:
            self.gp = gp_model
            self.Ymin = np.min(Y)

    def predict(self, X):
        X = self.X_scaler.transform(X)
        mu, sd = self.predict_gp(X)
        mu = self.Y_scaler.inverse_transform(mu)
        sd *= self.Y_scaler.scale_
        return mu, sd

    def predict_gp(self, X):
        mu, var = self.gp.predict(X)
        sd = np.sqrt(np.maximum(1e-18, var))
        return mu, sd

    def expected_improvement(self, X):
        mu, sd = self.predict(X)
        diff = self.Ymin - mu
        frac = diff / sd
        res = diff * norm.cdf(frac) + sd * norm.pdf(frac)
        return res

    def expected_improvement_with_grad(self, X):
        X = self.X_scaler.transform(X)
        Ymin = self.Y_scaler.transform(self.Ymin)

        mu, var = self.gp.predict(X)
        var = np.maximum(1e-18, var)
        sd = np.sqrt(var)

        dmu_dx, dvar_dx = self.gp.predictive_gradients(X)
        dmu_dx = dmu_dx[:, :, 0]
        dsd_dx = dvar_dx / (2 * np.sqrt(var))

        u = (Ymin - mu) / sd
        phi = np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        Phi = 0.5 * erfc(-u / np.sqrt(2))

        res = sd * (u * Phi + phi)
        res = res * self.Y_scaler.scale_[0]
        dres_dx = dsd_dx * phi - Phi * dmu_dx
        dres_dx = dres_dx * self.Y_scaler.scale_[0] / self.X_scaler.scale_
        return res, dres_dx


def param_vect_to_dict(param_vectors, par_descr):
    validate_par_descr(par_descr)
    assert len(param_vectors.shape) == 2

    param_dicts = []
    for pvector in param_vectors:
        pdict = {}
        index = 0
        for pname, (ptype, prange) in sorted(par_descr.items()):
            if ptype == 'categorical' and len(prange) == 2:
                prange = sorted(prange)
                pdict[pname] = prange[int(pvector[index])]
                index += 1
            elif ptype == 'categorical' and len(prange) > 2:
                prange = sorted(prange)
                value = pvector[index:index+len(prange)]
                i = np.where(value == 1)[0][0]
                pdict[pname] = prange[i]
                index += len(prange)
            elif ptype == 'integer':
                pdict[pname] = int(pvector[index])
                index += 1
            elif ptype == 'continuous':
                pdict[pname] = float(pvector[index])
                index += 1
        param_dicts.append(pdict)
    return param_dicts


def random_search(loss_function, par_descr, max_iter=30):
    opt_res = OptimisationResult()
    opt_res.x = random_samples(par_descr, max_iter)
    opt_res.f = np.zeros((max_iter, 1))
    opt_res.opt_method = np.array([['RANDOM'] * max_iter])
    opt_res.par_descr = par_descr
    for idx, pdict in enumerate(param_vect_to_dict(opt_res.x, par_descr)):
        opt_res.f[idx] = loss_function(**pdict)
    return opt_res


def cakeopt_search(loss_function, par_descr, max_iter=30, n_initial=2,
                random_state=42, internal_search='mies', noise=True):
    assert n_initial >= 2

    np.random.seed(random_state)
    opt_res = random_search(loss_function, par_descr, n_initial)
    opt_res.resize(max_iter)

    model = SearchSpaceModel(noise=noise)
    for cur_iter in range(n_initial, max_iter):
        try:
            model.fit(opt_res.x[:cur_iter, :], opt_res.f[:cur_iter, :])
        except np.linalg.LinAlgError as error:
            print('Iteration {0} LinAlgError:'.format(cur_iter))
            print(opt_res.x[:cur_iter, :])
            print(opt_res.f[:cur_iter, :])
            pdict = param_vect_to_dict(opt_res.x[cur_iter-1].reshape(1, -1), par_descr)[0]
            for ci in range(cur_iter, max_iter):
                opt_res.x[ci, :] = opt_res.x[cur_iter-1, :]
                opt_res.f[ci, :] = loss_function(**pdict)
            return opt_res

        if internal_search == 'mies':
            internal_opt_res = mies_internal(model.expected_improvement,
                par_descr)
        elif internal_search == 'lbfgsb':
            internal_opt_res = lbfgsb(model.expected_improvement,
                model.expected_improvement_with_grad, par_descr)
        elif internal_search == 'sgd':
            internal_opt_res = sgd(model.expected_improvement,
                model.expected_improvement_with_grad, par_descr)

        best = np.argmax(internal_opt_res.f.ravel())
        x = internal_opt_res.x[best, :]

        pdict = param_vect_to_dict(x.reshape(1, -1), par_descr)[0]
        f = loss_function(**pdict)

        opt_res.x[cur_iter, :] = x
        opt_res.f[cur_iter, :] = f
    return opt_res

