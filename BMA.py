import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

class posterior:

    def __init__(self,post_param):
        self.mean = post_param[0]
        self.std = post_param[1]

    def dist(self):
        return np.divide(1,2.*np.sqrt(self.std*np.pi))*np.exp(-0.5*np.divide(x - self.mean, self.std)**2)

class plot2D:

    def __init__(self,data,color,ax_size,xtick_size,ytick_size,legend_size,font_size,width):
        self.c = color
        self.ax = ax_size
        self.xt = xtick_size
        self.yt = ytick_size
        self.leg = legend_size
        self.font = font_size
        self.datax = data[0]
        self.datay = data[1]
        self.width = width

    def set_params(self,line_style):
        mpl.rcParams.update({
        'text.usetex' : True,
        'lines.linewidth' : self.width,
        'axes.labelsize' : self.ax,
        'xtick.labelsize' : self.xt,
        'ytick.labelsize' : self.yt,
        'legend.fontsize' : self.leg,
        'font.family' : 'palatino',
        'font.size' : self.font,
        'savefig.format' : 'png',
        'lines.linestyle' : line_style
        })
        return

    def density(self,x_label,y_label):
        self.set_params()
        sns.kdeplot(self.datay,shade=True,color=self.c)
        plt.xlabel(x_label,fontsize = self.font)
        plt.ylabel(y_label,fontsize = self.font)
        return

    def chain_plot(self,x_label,y_label,line_style,leg):
        self.set_params(line_style)
        plt.plot(self.datax,self.datay,color=self.c,label=leg)
        plt.xlabel(x_label,fontsize = self.font)
        plt.ylabel(y_label,fontsize = self.font)
        return

    def fill(self,color,alph,leg):
        plt.fill_between(x,0.,self.datay,facecolor=color,alpha=alph,label=leg)
        return

    def show(self,ylim,namefig):
        plt.ylim(ylim[0],ylim[1])
        plt.legend()
        plt.savefig('/Users/anabel/Documents/PhD/Code/THESIS/Chapter_4/'+namefig)
        # plt.show()
        return

# model1 = 1.
# model2 = 1.

# posterior1 = posterior([0.,1.])
# posterior2 = posterior([2.,2.])

# x = np.linspace(-10,10,num=100)

# data = [x,np.divide(model1*posterior1.dist() + model2*posterior2.dist(),(model1+model2))]
# data1 = [x,posterior1.dist()]
# data2 = [x,posterior2.dist()]

# fig = plt.figure(figsize=(12,11))

# BMA = plot2D(data,'black',20,40,40,30,40,3)
# model_1 = plot2D(data1,'red',20,40,40,30,40,1)
# model_2 = plot2D(data2,'green',20,40,40,30,40,1)

# BMA.chain_plot('$x$','$p(x)$','--','BMA')
# model_1.chain_plot('$x_{0}$','$\mathcal{P}(x_{0})$','-','$\mathcal{P}(x_{0} | \mathbf{y}_{\mathrm{obs}}, \mathcal{M}_{0})$')
# model_2.chain_plot('$x_{0}$','$\mathcal{P}(x_{0} | \mathbf{y}_{\mathrm{obs}})$','-','$\mathcal{P}(x_{0} | \mathbf{y}_{\mathrm{obs}}, \mathcal{M}_{1})$')

# model_1.fill('red',0.2)
# model_2.fill('green',0.2)


# BMA.show([0.,0.4],'BMA_equal')

model1_eq = 1.
model2_eq = 1.

model1_h = 4.
model2_l = 1.

model1_l = 1.
model2_h = 4.

posterior1 = posterior([0.,1.])
posterior2 = posterior([2.,2.])

x = np.linspace(-10,10,num=100)

data_eq = [x,np.divide(model1_eq*posterior1.dist() + model2_eq*posterior2.dist(),(model1_eq+model2_eq))]
data_2 = [x,np.divide(model1_l*posterior1.dist() + model2_h*posterior2.dist(),(model1_l+model2_h))]
data_1 = [x,np.divide(model1_h*posterior1.dist() + model2_l*posterior2.dist(),(model1_h+model2_l))]

data1 = [x,posterior1.dist()]
data2 = [x,posterior2.dist()]

fig = plt.figure(figsize=(15,11))

BMA_eq = plot2D(data_eq,'black',20,40,40,30,40,3)
BMA_2 = plot2D(data_2,'green',20,40,40,30,40,3)
BMA_1 = plot2D(data_1,'red',20,40,40,30,40,3)

model_1 = plot2D(data1,'red',20,40,40,30,40,1)
model_2 = plot2D(data2,'green',20,40,40,30,40,1)

BMA_eq.chain_plot('$x$','$p(x)$','--','BMA $\mathcal{M}_{0}$ 1:1 $\mathcal{M}_{1}$')
BMA_2.chain_plot('$x$','$p(x)$','--','BMA $\mathcal{M}_{0}$ 1:4 $\mathcal{M}_{1}$')
BMA_1.chain_plot('$x$','$p(x)$','--','BMA $\mathcal{M}_{0}$ 4:1 $\mathcal{M}_{1}$')

model_1.chain_plot('$x_{0}$','$\mathcal{P}(x_{0})$','-',None)
model_2.chain_plot('$x_{0}$','$\mathcal{P}(x_{0} | \mathbf{d}_{\mathrm{obs}})$','-',None)

model_1.fill('red',0.2,'$\mathcal{P}(x_{0} | \mathbf{d}_{\mathrm{obs}}, \mathcal{M}_{0})$')
model_2.fill('green',0.2,'$\mathcal{P}(x_{0} | \mathbf{d}_{\mathrm{obs}}, \mathcal{M}_{1})$')


BMA_1.show([0.,0.4],'BMA_all_d')


