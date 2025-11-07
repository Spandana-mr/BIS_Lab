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

    #############OUTPUT###############
 Iteration 1/100, Best Fitness: 52.01372975556269
Iteration 2/100, Best Fitness: 18.021304183436506
Iteration 3/100, Best Fitness: 8.483381510197345
Iteration 4/100, Best Fitness: 4.753884622217296
Iteration 5/100, Best Fitness: 2.436232003053962
Iteration 6/100, Best Fitness: 1.178531950984668
Iteration 7/100, Best Fitness: 0.4431397824642701
Iteration 8/100, Best Fitness: 0.21950537856872548
Iteration 9/100, Best Fitness: 0.1730071907749492
Iteration 10/100, Best Fitness: 0.0640375964534449
Iteration 11/100, Best Fitness: 0.02428977867234973
Iteration 12/100, Best Fitness: 0.019642983371974673
Iteration 13/100, Best Fitness: 0.011156420026897734
Iteration 14/100, Best Fitness: 0.003721641149494008
Iteration 15/100, Best Fitness: 0.0005775221536147114
Iteration 16/100, Best Fitness: 0.0005380697151941617
Iteration 17/100, Best Fitness: 0.00040082183205460083
Iteration 18/100, Best Fitness: 6.630034485146676e-05
Iteration 19/100, Best Fitness: 3.359079495291431e-05
Iteration 20/100, Best Fitness: 1.537117022908936e-05
Iteration 21/100, Best Fitness: 3.2843081289228393e-06
Iteration 22/100, Best Fitness: 3.9065103050929153e-07
Iteration 23/100, Best Fitness: 3.9065103050929153e-07
Iteration 24/100, Best Fitness: 1.516170015514541e-07
Iteration 25/100, Best Fitness: 7.908011900875356e-08
Iteration 26/100, Best Fitness: 1.2105623149719798e-08
Iteration 27/100, Best Fitness: 1.0717310797258905e-08
Iteration 28/100, Best Fitness: 1.6256222169939783e-09
Iteration 29/100, Best Fitness: 6.964877729928713e-10
Iteration 30/100, Best Fitness: 5.257074118462958e-10
Iteration 31/100, Best Fitness: 1.8980638713165655e-10
Iteration 32/100, Best Fitness: 9.61928525256331e-11
Iteration 33/100, Best Fitness: 3.157782557270042e-11
Iteration 34/100, Best Fitness: 1.6359069108766923e-11
Iteration 35/100, Best Fitness: 3.4133375831207272e-12
Iteration 36/100, Best Fitness: 2.2386528959433605e-12
Iteration 37/100, Best Fitness: 4.733662588802067e-13
Iteration 38/100, Best Fitness: 3.303480079870119e-13
Iteration 39/100, Best Fitness: 1.9151881743897634e-13
Iteration 40/100, Best Fitness: 1.2393942134260312e-13
Iteration 41/100, Best Fitness: 4.2273045176928103e-14
Iteration 42/100, Best Fitness: 9.354672108730383e-15
Iteration 43/100, Best Fitness: 6.083065450542482e-15
Iteration 44/100, Best Fitness: 5.440517501644213e-15
Iteration 45/100, Best Fitness: 1.6640840760309916e-15
Iteration 46/100, Best Fitness: 8.92577327984837e-16
Iteration 47/100, Best Fitness: 6.989797307819842e-16
Iteration 48/100, Best Fitness: 5.852012297890142e-16
Iteration 49/100, Best Fitness: 3.27368574944448e-16
Iteration 50/100, Best Fitness: 2.4116824674778233e-16
Iteration 51/100, Best Fitness: 1.3200808593116892e-16
Iteration 52/100, Best Fitness: 7.982513782589512e-17
Iteration 53/100, Best Fitness: 7.982513782589512e-17
Iteration 54/100, Best Fitness: 5.667134405325836e-17
Iteration 55/100, Best Fitness: 3.594021875413922e-17
Iteration 56/100, Best Fitness: 2.498050134747687e-17
Iteration 57/100, Best Fitness: 1.7578158628066894e-17
Iteration 58/100, Best Fitness: 1.4026525723584628e-17
Iteration 59/100, Best Fitness: 9.135134177340249e-18
Iteration 60/100, Best Fitness: 9.135134177340249e-18
Iteration 61/100, Best Fitness: 4.73080482301446e-18
Iteration 62/100, Best Fitness: 4.284060810607568e-18
Iteration 63/100, Best Fitness: 3.1759674882045523e-18
Iteration 64/100, Best Fitness: 2.404313928448472e-18
Iteration 65/100, Best Fitness: 1.816011000272694e-18
Iteration 66/100, Best Fitness: 1.129968806077517e-18
Iteration 67/100, Best Fitness: 8.833762765049228e-19
Iteration 68/100, Best Fitness: 7.051474871148458e-19
Iteration 69/100, Best Fitness: 6.914139615609986e-19
Iteration 70/100, Best Fitness: 4.736194941726867e-19
Iteration 71/100, Best Fitness: 3.795584022038693e-19
Iteration 72/100, Best Fitness: 3.2612779290942476e-19
Iteration 73/100, Best Fitness: 2.724431851717797e-19
Iteration 74/100, Best Fitness: 2.218439897692877e-19
Iteration 75/100, Best Fitness: 1.5406915227199976e-19
Iteration 76/100, Best Fitness: 1.326521189607842e-19
Iteration 77/100, Best Fitness: 1.145628424608182e-19
Iteration 78/100, Best Fitness: 1.031739758349611e-19
Iteration 79/100, Best Fitness: 9.239148782420258e-20
Iteration 80/100, Best Fitness: 7.428631981093873e-20
Iteration 81/100, Best Fitness: 6.652953990532668e-20
Iteration 82/100, Best Fitness: 6.012217917181071e-20
Iteration 83/100, Best Fitness: 5.321930580299526e-20
Iteration 84/100, Best Fitness: 4.63011466481964e-20
Iteration 85/100, Best Fitness: 4.3874487658949165e-20
Iteration 86/100, Best Fitness: 3.967935768007842e-20
Iteration 87/100, Best Fitness: 3.6202263133064444e-20
Iteration 88/100, Best Fitness: 3.4762834062324403e-20
Iteration 89/100, Best Fitness: 3.0952647170655987e-20
Iteration 90/100, Best Fitness: 3.0593186466308635e-20
Iteration 91/100, Best Fitness: 2.8956310514946406e-20
Iteration 92/100, Best Fitness: 2.769279578318933e-20
Iteration 93/100, Best Fitness: 2.6243054165378612e-20
Iteration 94/100, Best Fitness: 2.4538568179807155e-20
Iteration 95/100, Best Fitness: 2.394911478366634e-20
Iteration 96/100, Best Fitness: 2.3280505283276158e-20
Iteration 97/100, Best Fitness: 2.3280505283276158e-20
Iteration 98/100, Best Fitness: 2.270732548515495e-20
Iteration 99/100, Best Fitness: 2.2313776606185878e-20
Iteration 100/100, Best Fitness: 2.2297701376398014e-20
Best position found: [-6.73771760e-11  6.77476395e-11 -6.86247246e-11  7.12183201e-11
  5.81968465e-11]
Best objective value: 2.2297701376398014e-20
