import matplotlib.pyplot as Graph
import math
from scipy.optimize import curve_fit
import numpy as np

L = 155.55*10**(-3)
R = 0.15
labda = 632.8*10**(-9)
n = 1.5
d = 0.0194
#c = 0.913
#c = 0.9985
a_L = 24*10**(-6)
x = 0
#theta = math.asin((c+x*(a*L/R))**2)

def fringes_angle(x, n_):
    return 2*d*(n_-1)*(np.cos(x*math.pi/180)-1)/(labda*(1-np.cos(x*math.pi/180)-n_))

#curve_fit(func, xdata=temp_fringes[0][a:b], ydata=temp_fringes[1][a:b], p0 = start, maxfev=6000, absolute_sigma=True ,sigma = fringe_error)
delta_theta = [2.5, 2.7, 3.5]
fringes = [13, 26, 37]
fringes_error = [2, 2, 2]
popt, pcov = curve_fit(fringes_angle, xdata=delta_theta, ydata=fringes, p0=[1], maxfev=6000, absolute_sigma=True, sigma=fringes_error)
print("boi", popt, np.sqrt(pcov[0]))
n = popt[0]
Graph.plot(delta_theta, fringes, 'o')
Graph.plot([x for x in np.arange(0, 4.01, 0.01)], [fringes_angle(x, popt[0]) for x in np.arange(0, 4.01, 0.01)])
Graph.errorbar(delta_theta, fringes, fringes_error, fmt = "bo-")
Graph.xlim([0, 4])
Graph.ylim([0, 50])
Graph.ylabel("aantal fringes")
Graph.xlabel(r"$\Delta \theta$ ($^\circ$)")
Graph.savefig("brekingsindex.png")
Graph.show()
time_fringes = open("heatgun_3.txt", 'r')

limits = [0.32, 0.8]
maximum = True
fringes = 0

time_limits = [50, 100]
delta_fringes = 0

data = [[[], []], [[], []]]

toppen = [[[0],[0]], [[0], [0]]]
for line in time_fringes:
    time = float(line.split('\t')[0])
    value = float(line.split('\t')[1])
    data[0][0].append(time)
    data[0][1].append(value)
    if maximum:
        if value >= limits[1]:
            if(time_limits[0] < time < time_limits[1]):
                delta_fringes += 1
            fringes += 1
            data[1][0].append(time)
            data[1][1].append(fringes)
            maximum = False
    else:
        if value <= limits[0]:
            maximum = True

print(fringes)

Graph.xlabel("tijd (s)")
Graph.ylabel("spanning (V)")
Graph.xlim([550, 600])
Graph.ylim([0, 1])
Graph.plot(data[0][0], data[0][1], label="metingen")
Graph.plot([0, 600], [limits[0], limits[0]], '--r', label="grenswaardes")
Graph.plot([0, 600], [limits[1], limits[1]], '--r')
#Graph.legend()
Graph.savefig("licht_over_tijd.png")
Graph.show()


# Graph.xlim([0, 600])
# Graph.ylim([0, 200])
# Graph.xlabel("tijd (s)")
# Graph.ylabel("aantal fringes")
# Graph.plot(data[1][0], data[1][1])
# Graph.savefig("fringes_over_tijd.png")
#Graph.show()

def fringes_to_dL(fringes, R, labda, theta, n, d):
    return R*fringes*labda*(1-math.cos(theta)-n)**2/(2*d*n*(n-1)*math.sin(theta))

f_time_temp = open("heatgun_3_temp.txt", 'r')
time_temp = [[], []]
i = 0
delta_temps = [0,0]
for line in f_time_temp:
    if i >= 8:
        time = float(line.split('\t')[0])
        temp = float(line.split('\t')[1])
        time_temp[0].append(time)
        time_temp[1].append(temp)
        if(time == time_limits[0]):
            delta_temps[0] = temp
        elif(time == time_limits[1]):
            delta_temps[1] = temp

    i += 1

#print("bruh", (1/L)*fringes_to_dL(delta_fringes, R, labda, theta, n, d)/(delta_temps[0]-delta_temps[1]))

#time_temp = open("heatgun_3_temp.txt", 'r')

i = 0
j = 0

# for line in time_temp:
#     if i >= 10:
#         line_ = [float(line.split('\t')[0]), float(line.split('\t')[1])]
#         time = line_[0]
#         temp = line_[1]
#         k = 0
#         while k < len(data[1][0]):
#             if data[1][0][k] < time - 0.5:
#                 k+=1
#             else:
#                 break
#         average_fringes = 0
#         amount_of_fringes = 0
#         while k < len(data[1][0]):
#             if(data[1][0][k] <= time + 0.5):
#                 average_fringes += data[1][1][k]
#                 amount_of_fringes += 1
#             k += 1
#         if amount_of_fringes != 0:
#             average_fringes /= amount_of_fringes
#         fringes_temp[0].append(temp)
#         fringes_temp[1].append(average_fringes)
#         j += 1
#     i+=1
#


#Graph.plot(data[1][0], data[1][1])
# Graph.plot(time_temp[0], time_temp[1])
# Graph.xlabel("tijd (s)")
# Graph.ylabel("temperatuur ($^{\circ}$C)")
# Graph.xlim([time_temp[0][0], time_temp[0][-1]])
# Graph.ylim([40, 110])
# Graph.savefig("temperatuur_over_tijd.png")
#Graph.show()

temp_fringes = [[], []]
fringe_error = []
for i in range(len(data[1][0])):
    rounded_time = int(round(data[1][0][i]*2))/2
    for j in range(len(time_temp[0])):
        if time_temp[0][j] == rounded_time:
            temp_fringes[0].append(time_temp[1][j])
            temp_fringes[1].append(fringes-data[1][1][i])
            fringe_error.append(math.sqrt(fringes-data[1][1][i]))
            break
#FIX!!
print("ramdamdam", len(time_temp[1]), len(data[1][1]))
#temp_fringes[0].reverse()

Graph.plot(temp_fringes[0], temp_fringes[1])
Graph.show()

def func(x, alpha, e, c=20, theta_0=0):
    theta = np.arcsin((x-c)*alpha*L/R) - theta_0
    f = 2*d*(n-1)*(np.cos(theta)-1)/(labda*(1-np.cos(theta)-n)) + e
    return f

def func_approx(x, alpha, e, theta_0=0, c=20):
    f = (d*(n-1)/(n*labda))*(alpha*L*(x-c)/R + theta_0)**2 + e
    return f

def func_3(x, alpha, theta_0, c):
    theta = np.arcsin((x-c)*alpha*L/R) - theta_0
    f = theta*2*d*(n-1)*n*np.sin(theta_0)/(labda*(1-np.cos(theta_0)-n)**2)
    return f

fractions = [0, 5/5]
#a = math.floor(len(temp_fringes[0])*(1-fractions[1]))
#b = math.floor(len(temp_fringes[0])*(1-fractions[0]))
a = 0
b = len(temp_fringes[0])-1
start_ = [a_L, 0.001, 100]
p, c = curve_fit(func_3, xdata=temp_fringes[0][a:b], ydata=temp_fringes[1][a:b], p0 = start_[:], maxfev=100000) #absolute_sigma=True ,sigma = fringe_error[a:b])
print("oioioioi", temp_fringes[0][a], temp_fringes[0][b])
#p = [a_L, 0.00001]
print("bruhhhh", p)

#print(temp_fringes[0][0])
fraction_size = 5
a_L_ = [[], []]
lowest_difference = [10, 10]
parameters_opt_ = []
parameters_opt = []
cov_matrix_ = []
cov_matrix = []

def g(x, a, b):
    return a*x+b
#for i in range(fraction_size-1):
for i in range(1, len(temp_fringes[0])-math.floor(len(temp_fringes[0])/fraction_size)):

    fractions = [i, i+math.floor(len(temp_fringes[0])/fraction_size)]
    print(i)
    #a = math.floor(len(temp_fringes[0])*(fractions[0]))-1
    #b = math.floor(len(temp_fringes[0])*(fractions[1]))-1
    a = fractions[0]
    b = fractions[1]
    #print(a, b, len(temp_fringes[0]))
    start = (a_L, 50)
    #q, qcov = curve_fit(g, xdata=temp_fringes[0][a:b], ydata=temp_fringes[1][a:b], p0 = [1, 1], maxfev=100000, absolute_sigma=True ,sigma = fringe_error[a:b])
    parameters_, cov_ = curve_fit(func, xdata=temp_fringes[0][a:b], ydata=temp_fringes[1][a:b], p0 = start[:], maxfev=100000, absolute_sigma=True, sigma = fringe_error[a:b])
    #parameters_, cov_ = curve_fit(func, xdata = [x for x in range(121)], ydata=[g(x, q[0], q[1]) for x in range(121)], p0 = start[:], maxfev=100000)#, absolute_sigma=True ,sigma = fringe_error[a:b])
    #Graph.plot([x for x in range(121)], [g(x, q[0], q[1]) for x in range(121)], alpha=0.1)
    #parameters, cov = curve_fit(func_approx, xdata=temp_fringes[0][a:b], ydata=temp_fringes[1][a:b], p0 = start[:], maxfev=100000, absolute_sigma=True ,sigma = fringe_error[a:b])
    a_L_[0].append(parameters_[0])
    #a_L_[1].append(parameters[0])
    #if(0 < parameters_[0] < 0.001):
    if(i==1):
        print(parameters_[0])
        parameters_opt_ = parameters_[:]
    # if cov_[0][0] < lowest_difference[0]:
    #     lowest_difference[0] = cov_[0][0]
    #     parameters_opt_ = parameters_[:]
    #     cov_matrix_ = cov_[:]
    # if(abs(parameters_[0] - a_L) < lowest_difference[0]):
    #     lowest_difference[0] = abs(parameters_[0] - a_L)
    #     parameters_opt_ = parameters_[:]
    #     cov_matrix_ = cov_[:]
    #if(abs(parameters[0] - a_L) < lowest_difference[1]):
        #lowest_difference[1] = abs(parameters[0] - a_L)
        #parameters_opt = parameters[:]
        #cov_matrix = cov[:]
print("bruh", parameters_opt_, p)
print(cov_matrix_)
Graph.hist(a_L_[0], bins=np.arange(-0.004, 0.004, 0.001))# bins=np.arange(-0.004, 0.004, 8))
Graph.xlim([-0.004, 0.004])
#Graph.hist(a_L_[1], alpha=0.5)
Graph.show()
Graph.plot(temp_fringes[0], temp_fringes[1], label="metingen")
#Graph.plot([x for x in range(0, 120)], [func_approx(x, parameters_opt[0], parameters_opt[1]) for x in range(0, 120)], ':', label="fit")
#Graph.plot([x for x in range(0, 120)], [func(x, a_L, 20) for x in range(0, 120)], ':', label="bruh")
#Graph.plot([x for x in range(0, 120)], [func_3(x, 3.01517140*10**(-5), 1.05803742, -2.78344060*10**4) for x in range(0, 120)], ':', label="bruh")
Graph.plot([x for x in range(0, 120)], [func(x, parameters_opt_[0], parameters_opt_[1]) for x in range(0, 120)], '--')
Graph.plot([x for x in range(0, 120)], [func_3(x, p[0], p[1], p[2]) for x in range(0, 120)], '-.', color="orange", label="fit")
errors = [math.sqrt(temp_fringes[1][i]) for i in range(len(temp_fringes[1]))]
#Graph.errorbar(temp_fringes[0], temp_fringes[1], errors, fmt = "-")
#Graph.plot([temp_fringes[0][a], temp_fringes[0][a]], [0, 200], '--r', label="gekozen domein voor de fit")
#Graph.plot([temp_fringes[0][b], temp_fringes[0][b]], [0, 200], '--r')
#print(temp_fringes[0][a], temp_fringes[0][b])
#Graph.xlim(temp_fringes[0][-1], temp_fringes[0][0])
#Graph.ylim([0, 250])
Graph.xlabel("temperatuur ($^{\circ}$C)")
Graph.ylabel("aantal fringes")
#Graph.legend()
Graph.xlim([40, 105])
Graph.ylim([0, 200])
Graph.savefig("fringes_over_temperatuur.png")
Graph.show()


Graph.plot(temp_fringes[0], temp_fringes[1], label="metingen")
Graph.plot([x for x in range(0, 120)], [func(x, parameters_opt_[0], parameters_opt_[1]) for x in range(0, 120)], '--')
Graph.plot([x for x in range(0, 120)], [func_3(x, p[0], p[1], p[2]) for x in range(0, 120)], '-.', color="orange", label="fit")
errors = [math.sqrt(temp_fringes[1][i]) for i in range(len(temp_fringes[1]))]
Graph.errorbar(temp_fringes[0], temp_fringes[1], errors, fmt = "-")
Graph.xlabel("temperatuur ($^{\circ}$C)")
Graph.ylabel("aantal fringes")
a = 1
b = 1+math.floor(len(temp_fringes[0])/fraction_size)
Graph.xlim([temp_fringes[0][b], temp_fringes[0][a]])
Graph.ylim([130, 200])
Graph.xticks(np.arange(temp_fringes[0][b], temp_fringes[0][a], 2))
Graph.savefig("gebied.png")
Graph.show()
