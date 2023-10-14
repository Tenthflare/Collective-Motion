#Improved model

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.stats as stat
import scipy.optimize as fit
import time
from numba import jit


# parameters

Np = 3000
R = 1E-2
dt = 0.1
cutoff_rad = 5

# interaction parameters
u0 = 8
u1 = 0
a0 = 250
a1 = 50
stdev = 1.2

# division
local_density = 50

# dynamics parameter
alpha1 = 1
alpha2 = 1
beta = 1
gamma = 0.002 #not used
eta = 0.02
kappa = 1
initial_border = 0.2

iteration1 = 0
final = 500+1
step1 = 50
step2 = 2


pol_limit = 0.02
vel_limit = 0.02

'''Since the random number are generated in a particular sequence, 
setting it to zero yields the same set of random numbers every run'''
np.random.seed(0) 


'''The Simulation class contains cell generator(including number of cells and timesteps used in the simulation),
conservation of momentum function,  interaction force function, function to update the position and velocity of the cells'''
@jit
class Simulation(): 
    
    def __init__(self, Np, R, dt, cutoff_rad, local_density, alpha1, alpha2, beta, gamma, eta, kappa, initial_border, u0,u1,a0,a1): 
        #create instance of the class with the arguments sim=Simulation(Np,R,dt,cutoff_rad, local_density, alpha, beta, gamma, eta, kappa, initial_border)
        
        # cell characteristic
        self.Np = Np
        self.R = R
        self.dt = dt
        
        #2D simulation box size
        self.Xmax = initial_border*4
        self.Xmin = 0
        self.Ymax = 1
        self.Ymin = 0
        self.initial_border = initial_border
        
        # dynamics parameter
        self.cutoff_rad = cutoff_rad
        self.local_density = local_density
        
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.kappa = kappa
        
        self.u0 = u0
        self.u1 = u1
        self.a0 = a0
        self.a1 = a1

        
        #initial condition
        self.position = np.random.uniform([self.Xmin, self.Ymin],[initial_border,self.Ymax], size =(self.Np,2))
        self.velocity = np.zeros((self.Np, 2))
        self.polarity = np.zeros((self.Np, 2))
        self.local_velocity = np.zeros((self.Np,2))
        self.noise = np.zeros((self.Np,2))
        
        self.initial_pos = np.random.uniform([self.Xmin, self.Ymin],[initial_border,self.Ymax], size =(self.Np,2))
        


    '''conservation of momentum function'''
    def collision(self): 
        for i in range(self.Np):
            x = self.position[i,0]
            y = self.position[i,1]


            '''particle-boundary collision'''
            if x > self.Xmax - self.R: 
                self.velocity[i,0] = 0
                self.velocity[i,1] = 0
                self.position[i,0] = self.Xmax-self.R
                self.polarity[i,0] = 0
                self.polarity[i,1] = 0
            elif x < self.Xmin + self.R:
                self.velocity[i,0] = 0
                self.velocity[i,1] = 0
                self.position[i,0] = self.Xmin+self.R
                self.polarity[i,0] = 0
                self.polarity[i,1] = 0
            if y > self.Ymax - self.R:
                self.velocity[i,0] = 0
                self.velocity[i,1] = 0 
                self.position[i,1] = self.Ymax-self.R
                self.polarity[i,0] = 0
                self.polarity[i,1] = 0
            elif y < self.Ymin + self.R:
                self.velocity[i,0] = 0
                self.velocity[i,1] = 0
                self.position[i,1] = self.Ymin+self.R
                self.polarity[i,0] = 0
                self.polarity[i,1] = 0
                
                for j in range(self.Np):
                    if i == j:
                        continue
                    else:
                        r1 = self.position[i]
                        r2 = self.position[j]
                        v1 = self.velocity[i]
                        v2 = self.velocity[j]
                        zeta = 1
                        dist = np.dot(r1-r2,r1-r2)
                        '''particle-particle elastic collision'''
                        if dist <= (self.R)**2:
                            #if the distance (dot product of displacement vector) between two particles is less than or equal to the sum of the two radius, i.e. particle is touching
                            v1_new = (zeta*(v2-v1) + v1 + v2)/2
                            v2_new = (zeta*(v1-v2) + v1 + v2)/2
                            self.velocity[i] = v1_new
                            self.velocity[j] = v2_new
                            
            

            


    
    '''Functions to calculate mean border position, disorder parameter, and curve fitting'''
    def mean_border_position(self): #calculate the average y-position of the boundary cells
        y_bin1 = [] #bin 0.0-0.2
        y_bin2 = [] #bin 0.2-0.4
        y_bin3 = [] #bin 0.4-0.6
        y_bin4 = [] #bin 0.6-0.8
        y_bin5 = [] #bin 0.8-1.0
        for i in range(self.Np):
            xpos = self.position[i,0]
            ypos = self.position[i,1]
            if 0.0*self.Ymax <= ypos < 0.2*self.Ymax:
                y_bin1.append(xpos)
            elif 0.2*self.Ymax <= ypos < 0.4*self.Ymax:
                y_bin2.append(xpos)
            elif 0.4*self.Ymax <= ypos < 0.6*self.Ymax:
                y_bin3.append(xpos)
            elif 0.6*self.Ymax <= ypos < 0.8*self.Ymax:
                y_bin4.append(xpos)
            else:
                y_bin5.append(xpos)
        
        topcells = 15
        y_bin1.sort(reverse=True)
        if len(y_bin1) > topcells:
            y_bin1 = y_bin1[:topcells]
        else:
            pass
        y_bin2.sort(reverse=True)
        if len(y_bin2) > topcells:
            y_bin2 = y_bin2[:topcells]
        else:
            pass
        y_bin3.sort(reverse=True)
        if len(y_bin3) > topcells:
            y_bin3 = y_bin3[:topcells]
        else:
            pass
        y_bin4.sort(reverse=True)
        if len(y_bin4) > topcells:
            y_bin4 = y_bin4[:topcells]
        else:
            pass
        y_bin5.sort(reverse=True)
        if len(y_bin5) > topcells:
            y_bin5 = y_bin5[:topcells]
        else:
            pass
        boundary = y_bin1 + y_bin2 + y_bin3 + y_bin4 + y_bin5
        mean_border_progression = np.mean(boundary)
        
        return mean_border_progression
    
                 
    def local_cell_division(self):
        for i in range(self.Np):
            particle_count = 0
            for j in range(self.Np):
                r1 = self.position[i]
                r2 = self.position[j]
                
                r = np.sqrt(np.dot(r1-r2,r1-r2))
                if r <=self.cutoff_rad*self.R:
                    particle_count += 1
            
            if self.position[i,0] < self.mean_border_position()-(self.R*self.cutoff_rad) or self.position[i,0] > self.Xmin+(self.R*self.cutoff_rad):
                u = 1E-3*np.random.normal(0,1, size=2)
                if particle_count < self.local_density:

                    y = np.random.uniform(-2*self.R,2*self.R)
                    n = np.random.randint(1,3)
                    x = (-np.sqrt(4*self.R**2 - y**2))**n
                    new_xlocation = self.position[i,0] + x
                    new_ylocation = self.position[i,1] + y 
                    if x < 0:
                        if new_xlocation > self.mean_border_position()-(self.R*self.cutoff_rad):
                            new_xlocation = self.position[i,0] + x
                        if new_xlocation < self.Xmin:
                            new_xlocation = self.position[i,0] - x
                    else:
                        if new_xlocation > self.mean_border_position()-(self.R*self.cutoff_rad):
                            new_xlocation = self.position[i,0] - x
                        if new_xlocation < self.Xmin:
                            new_xlocation = self.position[i,0] + x
                            
                    if y < 0:
                        if new_ylocation > self.Ymax:
                            new_ylocation = self.position[i,1] + y
                        if new_ylocation < self.Ymin:
                            new_ylocation = self.position[i,1] - y
                    else:
                        if new_ylocation > self.Ymax:
                            new_ylocation = self.position[i,1] - y
                        if new_ylocation < self.Ymin:
                            new_ylocation = self.position[i,1] + y
                        
                    new_pos = np.array([new_xlocation, new_ylocation])
                    self.position = np.vstack((self.position, new_pos))
                    self.initial_pos = np.vstack((self.initial_pos, new_pos))
            
                    # new_vel = self.velocity[i]/2 + u
                    # self.velocity = np.vstack((self.velocity, new_vel))
                    # self.velocity[i] = self.velocity[i]/2 + u
                    
                    self.velocity = np.vstack((self.velocity, np.zeros(2)))
                    
                    self.polarity = np.vstack((self.polarity, np.zeros(2)))
                    self.local_velocity = np.vstack((self.local_velocity, np.zeros(2)))
                    self.noise = np.vstack((self.noise, np.zeros(2)))
                    
    
                    
                    self.Np = self.Np + 1
            else:
                pass


    

    
    
    '''function to update the position and velocity of the cells'''            
    def increment(self): 
        
        self.local_cell_division()
        

        for i in range(self.Np):
            v_cluster = np.zeros(2)
            particle_count = 0
            r1 = self.position[i]
            


            for j in range(self.Np):
                
                
                r2 = self.position[j]
                v2 = self.velocity[j]
                
                r = np.sqrt(np.dot(r1-r2,r1-r2))

                if r <=self.cutoff_rad*self.R:
                    v_cluster += v2
                    particle_count += 1

                r = np.sqrt(np.dot(r1-r2,r1-r2))
                
            self.local_velocity[i] = v_cluster/(particle_count)
            

        self.noise = np.random.normal(0,stdev,size=(self.Np,2))
            
        self.polarity += self.dt * (-self.alpha1*self.polarity + self.beta*(self.local_velocity)+ self.eta*self.noise)    

        self.velocity += self.dt * (-self.alpha2*(self.velocity)  +  self.kappa*self.polarity)

        self.position += self.dt * self.velocity
        self.collision()
          
            

    def order_parameter(self):
        v =  self.velocity
        
        norm = np.linalg.norm(v, axis=1)

        np.place(norm,norm==0, 1)
        new_norm = np.column_stack((norm,norm))

        normalised_v = v/new_norm
        sum_velocity = np.sum(normalised_v, axis=0)
        abs_sum = np.linalg.norm(sum_velocity)
        order = abs_sum/self.Np
        return order
    
    def correlationvp(self):

        v=self.velocity
        p=self.polarity
        
        norm_v = np.linalg.norm(v, axis=1)
        np.place(norm_v,norm_v==0, 1)
        new_norm_v = np.column_stack((norm_v,norm_v))
        normalised_v = v/new_norm_v


        norm_p = np.linalg.norm(p, axis=1)
        np.place(norm_p,norm_p==0, 1)
        new_norm_p = np.column_stack((norm_p,norm_p))
        normalised_p = p/new_norm_p


        dot = normalised_v[:,0]*normalised_p[:,0] + normalised_v[:,1]*normalised_p[:,1]
        # element = np.count_nonzero(dot)
        # if element == 0:
        #     correlation = 0
        # else:
        #     correlation =np.sum(dot)/element
        correlation = np.mean(dot)
        
        return correlation
    
    
    def avg_local_density(self):
        particle_count = np.zeros(self.Np)
        for i in range(self.Np):
            r1 = self.position[i]
            count = 0


            for j in range(self.Np):
                r2 = self.position[j]
                r = np.sqrt(np.dot(r1-r2,r1-r2))
                if r <=self.cutoff_rad*self.R:
                    count += 1
            particle_count[i] = count
        return np.mean(particle_count)

    def mean_square_displacement(self):

        disp_norm = np.linalg.norm(self.position-self.initial_pos, axis=1)
        mean_sq_disp = np.sum(disp_norm**2)/self.Np
        
        return mean_sq_disp

    
    def correlation(self):
        corr_p = np.zeros(10)
        corr_v = np.zeros(10)
        count_i = 0
        for i in range(self.Np):
            
            if 0.4 < self.position[i,1] < 0.6 and self.mean_border_position()-0.4 < self.position[i,0] < self.mean_border_position():
                count_i += 1
                r1 = self.position[i]
                p1 = self.polarity[i]
                norm_p1 = np.linalg.norm(p1)
                if norm_p1 == 0:
                    np1 = np.zeros(2)
                else:
                    np1 = p1/norm_p1
                v1 = self.velocity[i]
                norm_v1 = np.linalg.norm(v1)
                if norm_v1 == 0:
                    nv1 = np.zeros(2)
                else:
                    nv1 = v1/norm_v1

                
                total_p1 = []
                total_v1 = []
                
                total_p2 = []
                total_v2 = []
                
                total_p3 = []
                total_v3 = []
                
                total_p4 = []
                total_v4 = []

                total_p5 = []
                total_v5 = []
                
                total_p6 = []
                total_v6 = []
                
                total_p6 = []
                total_v6 = []
                
                total_p7 = []
                total_v7 = []
                
                total_p8 = []
                total_v8 = []
                
                total_p9 = []
                total_v9 = []
                
                total_p10 = []
                total_v10 = []
                
                for j in range(self.Np):
                    r2 = self.position[j]
                    p2 = self.polarity[j]
                    norm_p2 = np.linalg.norm(p2)
                    if norm_p2 == 0:
                        np2 = np.zeros(2)
                    else:
                        np2 = p2/norm_p2
                    v2 = self.velocity[j]
                    norm_v2 = np.linalg.norm(v2)
                    if norm_v2 == 0:
                        nv2 = np.zeros(2)
                    else:
                        nv2 = v2/norm_v2
                    
                    r = np.sqrt(np.dot(r1-r2,r1-r2))
                    if 0*self.R < r <= 2*self.R:
                        total_p1.append(np.dot(np1,np2))
                        total_v1.append(np.dot(nv1,nv2))
                    if 2*self.R < r <= 4*self.R:
                        total_p2.append(np.dot(np1,np2))
                        total_v2.append(np.dot(nv1,nv2))
                    if 4*self.R < r <= 6*self.R:
                        total_p3.append(np.dot(np1,np2))
                        total_v3.append(np.dot(nv1,nv2))
                    if 6*self.R < r <= 8*self.R:
                        total_p4.append(np.dot(np1,np2))
                        total_v4.append(np.dot(nv1,nv2))
                    if 8*self.R < r <= 10*self.R:
                        total_p5.append(np.dot(np1,np2))
                        total_v5.append(np.dot(nv1,nv2))
                    if 10*self.R < r <= 12*self.R:
                        total_p6.append(np.dot(np1,np2))
                        total_v6.append(np.dot(nv1,nv2))
                    if 12*self.R < r <= 14*self.R:
                        total_p7.append(np.dot(np1,np2))
                        total_v7.append(np.dot(nv1,nv2))
                    if 14*self.R < r <= 16*self.R:
                        total_p8.append(np.dot(np1,np2))
                        total_v8.append(np.dot(nv1,nv2))
                    if 16*self.R < r <= 18*self.R:
                        total_p9.append(np.dot(np1,np2))
                        total_v9.append(np.dot(nv1,nv2))
                    if 18*self.R < r <= 20*self.R:
                        total_p10.append(np.dot(np1,np2))
                        total_v10.append(np.dot(nv1,nv2))
                

                avg_p1 = np.mean(total_p1)
                avg_p2 = np.mean(total_p2) 
                avg_p3 = np.mean(total_p3)
                avg_p4 = np.mean(total_p4)
                avg_p5 = np.mean(total_p5)
                avg_p6 = np.mean(total_p6)
                avg_p7 = np.mean(total_p7)
                avg_p8 = np.mean(total_p8)
                avg_p9 = np.mean(total_p9)
                avg_p10 = np.mean(total_p10)
                
                

                avg_v1 = np.mean(total_v1)
                avg_v2 = np.mean(total_v2) 
                avg_v3 = np.mean(total_v3)
                avg_v4 = np.mean(total_v4)
                avg_v5 = np.mean(total_v5)
                avg_v6 = np.mean(total_v6)
                avg_v7 = np.mean(total_v7)
                avg_v8 = np.mean(total_v8)
                avg_v9 = np.mean(total_v9)
                avg_v10 = np.mean(total_v10)
                
                
                
                corr_p1 = np.array([avg_p1,avg_p2,avg_p3,avg_p4,avg_p5, avg_p6, avg_p7, avg_p8, avg_p9, avg_p10])
                corr_v1 = np.array([avg_v1,avg_v2,avg_v3,avg_v4,avg_v5, avg_v6, avg_v7, avg_v8, avg_v9, avg_v10])
                corr_p += corr_p1
                corr_v += corr_v1
        corr_p = corr_p/count_i
        corr_v = corr_v/count_i
            
        return corr_p, corr_v
                
        
        
        
        
        
        

sim = Simulation(Np,R,dt,cutoff_rad, local_density, alpha1, alpha2, beta, gamma, eta, kappa, initial_border, u0,u1,a0,a1) #assigning the class to a variable so the functions inside can be accessed
#-------------------------------------------------------------------------------------------
  
def best_fit(x,y): #in case of linear set of data
    denominator = np.dot(x,x) - np.mean(x)*np.sum(x)
    m = (np.dot(x,y) - np.mean(y) * np.sum(x))/denominator
    b = (np.mean(y) * np.dot(x,x) - np.mean(x) * np.dot(x,y))/denominator
    yfit = m*x + b
    return yfit

def curve(t,a,b,c): #in case of parabolic set of data
    return a * (t**b) + c


# ---------------------------------------------------------------------------------------------------

border = []
order = []
polarity = []
correlationvp = []
meansqdisp = []
avglocaldensity = []
avgglobaldensity = []
correlationv = np.zeros(10)
correlationp = np.zeros(10)

'''The position and velocity of cells are passed through increment function. 
The mean border position and disorder parameter is then calculated given updated position and velocity'''

'''The updating function above works only for one timestep, hence iteration(repeatedly calling the function) is needed'''

# fig, ax = plt.subplots()
# fig2, ax2 = plt.subplots()
# fig3, ax3 = plt.subplots()
# fig4, ax4 = plt.subplots()
# fig5, ax5 = plt.subplots()
# fig6, ax6 = plt.subplots()
# plt.style.use("default")





j = 1
k = 1
while iteration1 < final:
    sim.increment()
    
    
    if iteration1 % step1 == 0:
        r=sim.position
        v=sim.velocity
        np.savetxt('Velocity{}.csv'.format(iteration1), v, delimiter=',', fmt='%s')
        p=sim.polarity
        np.savetxt('Polarity{}.csv'.format(iteration1), p, delimiter=',', fmt='%s')

        l = sim.local_velocity
        n = sim.noise
        corr = sim.correlation()

        
        magvel = np.linalg.norm(v, axis=1)
        magpol = np.linalg.norm(p, axis=1)
        
        
        plt.scatter(sim.position[:,0],sim.position[:,1],color="red", s=2)
        
        plt.xlim(0, sim.Xmax)
        plt.ylim(0, sim.Ymax)
        plt.savefig("Config{}.pdf".format(iteration1))
        plt.close()
        

        plt.quiver(sim.position[:,0],sim.position[:,1], sim.velocity[:,0],sim.velocity[:,1], magvel, cmap="Blues",   width= 1E-3)
        plt.xlim(0, sim.Xmax)
        plt.ylim(0, sim.Ymax)
        
        plt.savefig("PositionVelocity{}.pdf".format(iteration1))
        plt.close()
        
        plt.quiver(sim.position[:,0],sim.position[:,1], sim.polarity[:,0],sim.polarity[:,1], magpol, cmap="Reds",  width= 1E-3)
        plt.xlim(0, sim.Xmax)
        plt.ylim(0, sim.Ymax)
        plt.savefig("PositionPolarity{}.pdf".format(iteration1))
        plt.close()
    

    
    if iteration1 % step2 == 0:
        border.append(sim.mean_border_position())
        order.append(sim.order_parameter())
        correlationvp.append(sim.correlationvp())
        avglocaldensity.append(sim.avg_local_density())
        meansqdisp.append(sim.mean_square_displacement())
        correlationv = np.column_stack((correlationv, sim.correlation()[1]))
        correlationp = np.column_stack((correlationp, sim.correlation()[0]))
        
        
    iteration1 += 1

np.array(border)
np.array(order)
np.array(correlationvp)
np.array(avglocaldensity)
np.array(correlationv)
np.array(correlationp)

np.savetxt('MeanBorder.csv', border, delimiter=',', fmt='%s')
np.savetxt('Order.csv', order, delimiter=',', fmt='%s')
np.savetxt('Correlationvp.csv', correlationvp, delimiter=',', fmt='%s')
np.savetxt('AvgLocalDensity.csv', avglocaldensity, delimiter=',', fmt='%s')
np.savetxt('MeanSqDisp.csv', meansqdisp, delimiter=',', fmt='%s')
np.savetxt('CorrelationV.csv', correlationv, delimiter=',', fmt='%s')
np.savetxt('CorrelationP.csv', correlationp, delimiter=',', fmt='%s')
  





