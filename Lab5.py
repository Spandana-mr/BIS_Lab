import numpy as np
import matplotlib.pyplot as plt

class GreyWolfOptimizer:
    def __init__(self, obj_func, lb, ub, dim, num_wolves=20, max_iter=100):
        self.obj_func = obj_func      
        self.lb = np.array(lb)        
        self.ub = np.array(ub)      
        self.dim = dim               
        self.num_wolves = num_wolves
        self.max_iter = max_iter

    
        self.positions = np.random.uniform(self.lb, self.ub, (self.num_wolves, self.dim))
        
   
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')
        
        self.convergence_curve = []
    
    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.num_wolves):
              
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                
                fitness = self.obj_func(self.positions[i])
                
         
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.positions[i].copy()
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.positions[i].copy()
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.positions[i].copy()
            

            a = 2 - t * (2 / self.max_iter)
            
            for i in range(self.num_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * self.alpha_pos[j] - self.positions[i, j])
                    X1 = self.alpha_pos[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * self.beta_pos[j] - self.positions[i, j])
                    X2 = self.beta_pos[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * self.delta_pos[j] - self.positions[i, j])
                    X3 = self.delta_pos[j] - A3 * D_delta

                  
                    self.positions[i, j] = (X1 + X2 + X3) / 3

            self.convergence_curve.append(self.alpha_score)
            print(f"Iteration {t+1}/{self.max_iter}, Best Fitness: {self.alpha_score}")
        
        return self.alpha_pos, self.alpha_score


if __name__ == "__main__":

    def sphere(x):
        return np.sum(x**2)

    dim = 5
    lb = [-10] * dim
    ub = [10] * dim

    gwo = GreyWolfOptimizer(obj_func=sphere, lb=lb, ub=ub, dim=dim, num_wolves=30, max_iter=100)
    best_pos, best_score = gwo.optimize()

    print("Best position found:", best_pos)
    print("Best objective value:", best_score)

    plt.plot(gwo.convergence_curve)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.title("GWO Convergence Curve")
    plt.show()
