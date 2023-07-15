import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import os.path

##Numericke reseni difuzni rovnice u_t - D(u_xx + u_yy) = 0
##Vyuzita metoda stridavych smeru (adjected direction implicit)
class ADI_heat:
    def __init__(self, my_Bx, my_By, my_C, my_Jx, my_Jy, my_D, bc, my_bc1, my_bc2, my_bc3, my_bc4):
        #Vytvori slozku Img, pokud neexistuje
        if not os.path.exists('Img'):
            os.makedirs('Img')
        #interval na x-ove souradnici (-B, B)
        self.Bx = my_Bx
        #interval na y-ove souradnici (-B, B)
        self.By = my_By
        #CFL cislo
        self.C = my_C
        #Pocet bunek v x
        self.Jx = my_Jx
        #Pocet bunek v y
        self.Jy = my_Jy
        #Koeficient difuze
        self.D = my_D
        
        #Prostorovy krok v x a y
        self.dx = 2.0*self.Bx/(self.Jx-1);
        self.dy = 2.0*self.By/(self.Jy-1);
        #Dle CFL cisla se urci casovy krok
        self.dt = self.C*min(self.dx,self.dy)/abs(self.D);

        
        #Implicitni matice soustavy pro reseni v x
        self.Mx = np.zeros((self.Jx,self.Jx))
        #Implicitni matice soustavy pro reseni v y
        self.My = np.zeros((self.Jy,self.Jy))
        #Matice reseni 
        self.U = np.zeros((self.Jx,self.Jy))
        #Matice reseni v dalsim casovem kroce
        self.UN = np.zeros((self.Jx,self.Jy))
        #Matice prave strany
        self.R1 = np.zeros((self.Jx,self.Jx))
        self.R2 = np.zeros((self.Jy,self.Jy))
        #Vektor x a y pro dane prostorove kroky
        self.x=np.linspace(-self.Bx,self.Bx,self.Jx)
        self.y=np.linspace(-self.By,self.By,self.Jy)
        self.Xx, self.Yy = np.meshgrid(self.x, self.y)
        
        #Inicializace matice soustavy pro implicitni schema
        self.Mx[1,0] = ((2.0/self.dt) + (2.0*self.D/self.dx**2))
        self.Mx[1,1] = (-self.D/self.dx**2)
        for i in range (2,self.Jx-2):
            self.Mx[i,i-1] = -self.D/self.dx**2
            self.Mx[i,i] = ((2.0/self.dt) + (2.0*self.D/self.dx**2))
            self.Mx[i,i+1] = (-self.D/self.dx**2)
        self.Mx[self.Jx-2,self.Jx-2] = -self.D/self.dx**2
        self.Mx[self.Jx-2,self.Jx-1] = ((2.0/self.dt) + (2.0*self.D/self.dx**2))
            
        #Inicializace matice soustavy pro implicitni schema
        self.My[1,0] = ((2.0/self.dt) + (2.0*self.D/self.dy**2))
        self.My[1,1] = (-self.D/self.dy**2)
        for i in range (2,self.Jy-2):
            self.My[i,i-1] = -self.D/self.dy**2
            self.My[i,i] = ((2.0/self.dt) + (2.0*self.D/self.dy**2))
            self.My[i,i+1] = (-self.D/self.dy**2)
        self.My[self.Jy-2,self.Jy-2] = -self.D/self.dy**2
        self.My[self.Jy-2,self.Jy-1] = ((2.0/self.dt) + (2.0*self.D/self.dy**2))
            
        if bc == "dirichlet":
            self.Mx[0,0] = 1
            self.Mx[0,1] = 0
            self.Mx[self.Jx-1,self.Jx-2] = 0
            self.Mx[self.Jx-1,self.Jx-1] = 1
            
            self.My[0,0] = 1
            self.My[0,1] = 0
            self.My[self.Jy-1,self.Jy-2] = 0
            self.My[self.Jy-1,self.Jy-1] = 1
        if bc == "neumann":
            self.Mx[0,0] = 1
            self.Mx[0,1] = -1
            self.Mx[self.Jx-1,self.Jx-2] = -1
            self.Mx[self.Jx-1,self.Jx-1] = 1
            
            self.My[0,0] = 1
            self.My[0,1] = -1
            self.My[self.Jy-1,self.Jy-2] = -1
            self.My[self.Jy-1,self.Jy-1] = 1
        
        #Invertuje matici typu Jx X Jx
        self.Mx = inv(self.Mx)
        #Invertuje matici typu Jy X Jy
        self.My = inv(self.My)

        #Matice prave strany R1 typu Jx X Jx v aktualnim case
        self.R1[1,0] = ((2.0/self.dt) - (2.0*self.D/self.dy**2))
        self.R1[1,1] = self.D/self.dy**2
        for i in range (2,self.Jx-2):
            self.R1[i,i-1] = self.D/self.dy**2
            self.R1[i,i] = ((2.0/self.dt) - (2.0*self.D/self.dy**2))
            self.R1[i,i+1] = self.D/self.dy**2
        self.R1[self.Jx-2,self.Jx-2] = self.D/self.dy**2
        self.R1[self.Jx-2,self.Jx-1] = ((2.0/self.dt) - (2.0*self.D/self.dy**2))
        
        #Matice prave strany R2 typu Jy X Jy v aktualnim case
        self.R2[1,0] = ((2.0/self.dt) - (2.0*self.D/self.dx**2))
        self.R2[1,1] = self.D/self.dx**2
        for i in range (2,self.Jy-2):
            self.R2[i,i-1] = self.D/self.dx**2
            self.R2[i,i] = ((2.0/self.dt) - (2.0*self.D/self.dx**2))
            self.R2[i,i+1] = self.D/self.dx**2
        self.R2[self.Jy-2,self.Jy-2] = self.D/self.dx**2
        self.R2[self.Jy-2,self.Jy-1] = ((2.0/self.dt) - (2.0*self.D/self.dx**2))
            
        #Nastav pocatecni podminky
        self.bc1 = my_bc1
        self.bc2 = my_bc2
        self.bc3 = my_bc3
        self.bc4 = my_bc4
        self.SetIC()
        
    #Aplikace okrajovych podminek
    def SetBoundary(self):
        self.U[0,:] = self.bc1
        self.U[:,0] = self.bc2
        self.U[self.Jx-1,:] = self.bc3
        self.U[:,self.Jy-1] = self.bc4
        self.U[0,0] = (self.U[0,1]+self.U[1,0])/2
        self.U[0,self.Jy-1] = (self.U[0,self.Jy-2]+self.U[1,self.Jy-1])/2
        self.U[self.Jx-1,0] = (self.U[self.Jx-2,0]+self.U[self.Jx-1,1])/2
        self.U[self.Jx-1,self.Jy-1] = (self.U[self.Jx-1,self.Jy-2]+self.U[self.Jx-2,self.Jy-1])/2

    #Cykresleni reseni do souboru
    def PlotResults(self, num, pl):
        n='{:04}'.format(num)
        
        if pl=="2d":
            fig, ax = plt.subplots()
            p = ax.pcolor(self.Xx, self.Yy, self.U.transpose(), cmap=cm.hsv, vmin=0, vmax=1)
            cb = fig.colorbar(p, ax=ax)
        
        if pl == "3d":
            fig = plt.figure(figsize=(8,6))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            surf = ax.plot_surface(self.Xx, self.Yy, np.transpose(self.U), rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_zlim(0,1)

        fig.savefig("Img/img_"+str(n)+".png")
        fig.clf()
        plt.close()
        
    def GetTimeStep(self):
        return self.dt
        

    #Cyklus pro samotne reseni rovnice
    def SolveStep(self):
        #Aplikace okrajove podminky
        self.SetBoundary()
        #implicitne v x
        for j in range (0,self.Jy):
            #Napocita pravou stranu
            rh = self.R1.dot((self.U[:,j]))
            #Vyuzije implicitni schema
            self.UN[:,j] = self.Mx.dot(rh)
        self.U = self.UN
        
        #Aplikace okrajove podminky
        self.SetBoundary()
        #implicitne v y
        for j in range (0,self.Jx):
            rh = self.R2.dot(np.transpose(self.U[j,:]))
            self.UN[j,:] = np.transpose(self.My.dot(rh))
        self.U = self.UN
        
    #Pocatecni podminky
    def SetIC(self):
        for i in range (1,self.Jx):
            for j in range (1,self.Jy):
                #Kvadr
                if np.abs(self.x[i]) < 2 and np.abs(self.y[j]) < 6:
                    self.U[i,j] = 1
                  
                  
                #2D Gauss
                #self.omega = 20
                #self.U[i,j] = np.exp(-(self.x[i]**2 + self.y[j]**2)/self.omega)
                
                #self.U[i,j] = np.abs((self.y[j]/max(self.y))*np.sin(self.x[i]/len(self.x)*8*np.pi))
                #self.U[i,j] = abs(self.x[i]/max(self.x))
                #self.U[int(round(self.Jx/2)),int(round(self.Jy/2))] = 1
    
#######################################################################################
##############################__==END_CLASS==__########################################
#######################################################################################


#ADI_heat(Bx, By, C. Jx, Jy, D, Bc)
eq = ADI_heat(10.0, 15.0, 0.8, 120, 160, 1, "neumann", 0.0, 0.0, 0.0, 0.0)

#Cas simulace
T = 15.0
t = 0
num = 0
dt = eq.GetTimeStep()
while t<T:
    eq.PlotResults(num,"3d")
    num = num + 1
    eq.SolveStep()
    t=t+(dt)
