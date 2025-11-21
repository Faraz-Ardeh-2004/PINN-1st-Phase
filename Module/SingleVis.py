import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
class Vis():
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    plt.rcParams['xtick.labelsize'] = 10    # Axis tick label size
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['axes.titlesize'] = 17     # Title font size
    plt.rcParams['axes.labelsize'] = 16     # Axis label font size
    plt.rcParams['axes.linewidth'] = 1      # Axis line width

    def __init__(self, ques_name, ini_num, file_desti, module_name, input =[], u = [], mode: str = 'teacher'):
        input_num = input.T.shape[0]
        if input_num == 2:
            self.x, self.y = input[:,0], input[:,1]
        elif input_num == 3:
            self.x, self.y, self.z = input[:,0], input[:,1], input[:,2]
        self.u = u
        self.ques_name = ques_name
        self.ini_num = ini_num
        self.file_densti =  file_desti 
        self.module_name = module_name
        if mode == 'student':
            self.module_name += '_student'

    def loss_vis(self):
        self.loss_desti = self.file_densti + '/Loss/'
        df = pd.read_csv(f'{self.loss_desti}{self.ques_name}_{self.ini_num}_loss_{self.module_name}.csv').values
        header = pd.read_csv(f'{self.loss_desti}{self.ques_name}_{self.ini_num}_loss_{self.module_name}.csv', nrows=0).columns

        # Sometimes iterations are continued from previous runs; create a fresh iteration array here
        iter = np.arange(0, len(df[:,0]), 1)

        for j in range (len(header)-1):
            # Skip columns that start with zero values
            if df[0,j+1] == 0:
                continue
            plt.figure(figsize=(3.85, 3.5)) # adjust figure size
            plt.plot(iter , df[:,j+1])
            plt.yscale('log')
            ax = plt.gca()
            ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='x')    # use scientific notation on x-axis
            plt.grid()
            plt.xlabel(header[0])
            plt.ylabel(header[j+1])
            plt.title(self.ques_name + ' ' + header[j+1] + ' ' +self.module_name, pad=10)
            plt.savefig(self.loss_desti + self.ques_name +'_'+str(self.ini_num)+'_' + header[j+1] + '_' + self.module_name + '.png', bbox_inches='tight')
            plt.close()
    
    def figure_2d(self):
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)
        
        for i in range(len(self.u.T)):
            print(f"Drawing {self.ques_name} {self.module_name} figure {i+1}...")

            if 'Flow' in self.ques_name:
                fig, ax = plt.subplots(figsize=(4.4, 2)) # special aspect for flow cases
                x = self.x.reshape([-1,])
                y = (self.y + 0.2).reshape([-1,])  # shift y by 0.2 to align coordinate axis

                # Cylinder parameters
                center_x, center_y, radius = 0.2, 0.2, 0.05

                # Triangulation
                triang = tri.Triangulation(x, y)

                # Mask triangles inside the cylinder region
                mask = []
                for tri_idx in triang.triangles:
                    # compute triangle centroid
                    xc = x[tri_idx].mean()
                    yc = y[tri_idx].mean()
                    # check if centroid is inside the cylinder
                    if ((xc - center_x)**2 + (yc - center_y)**2) < radius**2:
                        mask.append(True)
                    else:
                        mask.append(False)
                triang.set_mask(mask)

                cf = plt.tripcolor(triang, self.u[:,i], cmap='rainbow', vmin=0 if i < 2 else -0.6, vmax=4 if i == 0 else 1.3 if i == 1 else 0.6)

            else: 
                fig, ax = plt.subplots(figsize=(3.85, 3.5))        # figure size
                cf = plt.scatter(self.x, self.y, c=self.u[:,i], alpha=1 - 0.1, edgecolors='none', cmap='rainbow', marker='s', s=int(8))  # s is point size
            plt.xlabel('x', style='italic')
            plt.ylabel('y', style='italic')
            plt.margins(0) # tighten axis margins
            plt.title (self.ques_name + ' ' + self.module_name, pad=10)
            fig.colorbar(cf, fraction=0.046, pad=0.04)
            plt.savefig(f"{self.figure_desti}{self.ques_name}_figure_{i+1}_{self.module_name}.png", bbox_inches='tight')
            plt.close()

    def figure_3d(self):
        self.figure_desti = self.file_densti + '/Figure/'
        if not os.path.exists(self.figure_desti):
            os.mkdir(self.figure_desti)
        
        for i in range(len(self.u.T)):
            print(f"Drawing {self.ques_name} {self.module_name} figure {i+1}...")
            fig = plt.figure(figsize=(4.4, 4))  
            cf = fig.add_subplot(111, projection='3d')
            scatter = cf.scatter(self.x, self.y, self.z, c= self.u[:,i], cmap='rainbow', edgecolors='none', vmin=self.u[:,i].min(), vmax=self.u[:,i].max())
            # cf.plot(self.x, self.y, c=self.u, alpha=1 - 0.1, edgecolors='none', cmap='rainbow',marker='s', s=int(8))  #s就是size也就是点的大小
            cf.set_xlabel('x', style='italic')
            cf.set_ylabel('y', style='italic')
            cf.set_zlabel('z', style='italic')
            cf.view_init(elev=20, azim=160)
            # cf.set_title(self.ques_name + ' ' + self.module_name)
            # fig.colorbar(cf, fraction=0.046, pad=0.04)
            colorbar = plt.colorbar(scatter, fraction=0.04, pad=0.2)  # add colorbar in 3D axes
            colorbar.set_label('T')  # set colorbar label
            plt.savefig(f"{self.figure_desti}{self.ques_name}_figure_{i+1}_{self.module_name}.png", bbox_inches='tight')
            plt.close()

    def para_vis(self):
        self.para_desti = self.file_densti + '/Parameters/'
        df = pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv').values
        header = pd.read_csv(self.para_desti + self.ques_name + '_' + str(self.ini_num) + '_paras_' + self.module_name + '.csv',  nrows=0).columns

        iter = np.arange(0, len(df[:,0]), 1)

        for j in range (len(header)-1):
            plt.plot(iter , df[:,j+1])
            # plt.plot(df[:,0],df[:,j+1])
            # plt.yscale('log')
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 16}
            # plt.grid()
            plt.xlabel(header[0], fontdict=font)
            plt.ylabel(header[j+1], fontdict=font)
            plt.title(self.ques_name + ' ' + header[j+1] + ' ' +self.module_name)
            plt.savefig(self.para_desti + self.ques_name +'_'+str(self.ini_num)+'_' + header[j+1] + '_' +self.module_name + '.png', bbox_inches='tight')
            plt.close()
