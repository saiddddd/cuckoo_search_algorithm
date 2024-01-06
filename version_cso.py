import numpy as np
import matplotlib.pyplot as plt
from math import gamma

class CSO:
    def __init__(self, fitness, P=150, dimension=None, pa=0.25, beta=1.5, bound=None, 
                plot=False, min=True, verbose=False, Tmax=300):
        self.fitness = fitness
        self.P = P 
        self.dimension = dimension
        self.Tmax = Tmax
        self.pa = pa
        self.beta = beta
        self.bound = bound
        self.plot = plot
        self.min = min
        self.verbose = verbose

        self.X = []

        if bound is not None:
            for (U, L) in bound:
                x = (U-L)*np.random.rand(P,) + L 
                self.X.append(x)
            self.X = np.array(self.X).T
        else:
            self.X = np.random.randn(P,dimension)

        self.position_history = []  # To store position history
        self.fitness_history = []  # To store fitness history
        self.best_solution_history = []  # To store history of best solution

    def update_position_1(self):
        num = gamma(1+self.beta)*np.sin(np.pi*self.beta/2)
        den = gamma((1+self.beta)/2)*self.beta*(2**((self.beta-1)/2))
        σu = (num/den)**(1/self.beta)
        σv = 1
        u = np.random.normal(0, σu, self.dimension)
        v = np.random.normal(0, σv, self.dimension)
        S = u/(np.abs(v)**(1/self.beta))

        for i in range(self.P):
            if i == 0:
                self.best = self.X[i, :].copy()
            else:
                self.best = self.optimum(self.best, self.X[i, :])

        Xnew = self.X.copy()
        for i in range(self.P):
            Xnew[i, :] += np.random.randn(self.dimension)*0.01*S*(Xnew[i, :]-self.best) 
            self.X[i, :] = self.optimum(Xnew[i, :], self.X[i, :])

    def update_position_2(self):
        Xnew = self.X.copy()
        Xold = self.X.copy()
        for i in range(self.P):
            d1, d2 = np.random.randint(0, 5, 2)
            for j in range(self.dimension):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i, j] += np.random.rand()*(Xold[d1, j]-Xold[d2, j]) 
            self.X[i, :] = self.optimum(Xnew[i, :], self.X[i, :])

    def optimum(self, best, particle_x):
        if self.min:
            if self.fitness(best) > self.fitness(particle_x):
                best = particle_x.copy()
        else:
            if self.fitness(best) < self.fitness(particle_x):
                best = particle_x.copy()
        return best

    def clip_X(self):
        if self.bound is not None:
            for i in range(self.dimension):
                xmin, xmax = self.bound[i]
                self.X[:, i] = np.clip(self.X[:, i], xmin, xmax)

    def execute(self):
        self.fitness_time, self.time = [], []

        for t in range(self.Tmax):
            self.update_position_1()
            self.clip_X()
            self.update_position_2()
            self.clip_X()
            self.fitness_time.append(self.fitness(self.best))
            self.time.append(t)
            if self.verbose:
                print('Iteration:  ', t, '| current fitness (cost) value:', round(self.fitness(self.best), 7))

            # Save position and fitness history
            self.position_history.append(self.X.copy())
            self.fitness_history.append(self.fitness(self.best))
            self.best_solution_history.append(self.best.copy())

        print('\nOPTIMUM SOLUTION >', np.round(self.best.reshape(-1), 7).tolist())
        print('\nOPTIMUM FITNESS  >', np.round(self.fitness(self.best), 7))
        print()
        if self.plot:
            self.Fplot()
            print("Search History of CSO")
            self.plot_search_history()
            self.plot_trajectory()
            self.plot_fitness_history()
            
        return np.round(self.best.reshape(-1), 7).tolist(), np.round(self.fitness(self.best), 7)

    def Fplot(self):
        plt.plot(self.time, self.fitness_time, marker='o', label='Fitness Curve')
        plt.title('(Convergence curve) Fitness value vs Iteration')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness value')
        plt.legend(loc='best')
        plt.show()

    def plot_search_history(self):
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(152)
        for k1 in range(self.P):
            trajectory = np.array(self.position_history)  
            plt.plot(trajectory[:, k1, 0], '.', markersize=1, markeredgecolor='k', markerfacecolor='k')
        plt.plot(self.best[0], self.best[0], '.', markersize=10, markeredgecolor='r', markerfacecolor='r')
        plt.title('Search history (x1 only)')
        plt.xlabel('Iteration')
        plt.ylabel('x1')

    def plot_trajectory(self):
        plt.subplot(153)
        trajectory = np.array(self.position_history[0])  
        plt.plot(trajectory[:, 0])
        plt.title('Trajectory')
        plt.xlabel('Iteration')
        plt.ylabel('x1')

    def plot_fitness_history(self):
        plt.subplot(154)
        plt.plot(self.fitness_history)
        plt.title('Average fitness')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
