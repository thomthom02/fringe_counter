import matplotlib.pyplot as Graph
import math
from scipy.optimize import curve_fit
import numpy as np

L = 155.55*10**(-3)
R = 0.15
labda = 632.8*10**(-9)
n = 1.5
d = 0.0194
c = 0.913
#c = 0.9985
a_L = 24*10**(-6)
x = 0
#theta = math.asin((c+x*(a*L/R))**2)

time_fringes = open("heatgun_3.txt", 'r')

limits = [0.32, 0.8]
maximum = True
fringes = 0

time_limits = [50, 100]
delta_fringes = 0

data = [[[], []], [[], []]]

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
Graph.xlim([0, 600])
Graph.ylim([0, 1])
Graph.plot(data[0][0], data[0][1], label="metingen")
Graph.plot([0, 600], [limits[0], limits[0]], '--r', label="grenswaardes")
Graph.plot([0, 600], [limits[1], limits[1]], '--r')
Graph.legend()
Graph.savefig("licht_over_tijd.png")
Graph.show()

Graph.xlim([0, 600])
Graph.ylim([0, 200])
Graph.xlabel("tijd (s)")
Graph.ylabel("aantal fringes")
Graph.plot(data[1][0], data[1][1])
Graph.savefig("fringes_over_tijd.png")
Graph.show()

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
Graph.plot(time_temp[0], time_temp[1])
Graph.xlabel("tijd (s)")
Graph.ylabel("temperatuur ($^{\circ}$C)")
Graph.xlim([time_temp[0][0], time_temp[0][-1]])
Graph.ylim([40, 110])
Graph.savefig("temperatuur_over_tijd.png")
Graph.show()

temp_fringes = [[], []]
for i in range(len(data[1][0])):
    rounded_time = int(round(data[1][0][i]*2))/2
    for j in range(len(time_temp[0])):
        if time_temp[0][j] == rounded_time:
            temp_fringes[0].append(time_temp[1][j])
            temp_fringes[1].append(fringes-data[1][1][i])
            break

#Graph.plot(temp_fringes[0], temp_fringes[1])
#Graph.show()

def func(x, alpha, c, d, theta_0):
    theta = np.arcsin((x-c)*alpha*L/R)
    f = 2*d*(n-1)*(np.cos(theta+theta_0)-1)/(labda*(1-np.cos(theta +theta_0)-n)) + d
    return f

#print(temp_fringes[0][0])
fractions = [3/4, 5/6]
a = int(len(temp_fringes[0])*(1-fractions[1]))-1
b = int(len(temp_fringes[0])*(1-fractions[0]))-1
start = (a_L, 20, 40, 0)
parameters_opt, cov_matrix = curve_fit(func, xdata=temp_fringes[0][a:b], ydata=temp_fringes[1][a:b], p0 = start, maxfev=6000)#absolute_sigma=True ,sigma = events_err)
print(parameters_opt)
print(cov_matrix)

Graph.plot(temp_fringes[0], temp_fringes[1], label="metingen")
Graph.plot([x for x in range(0, 120)], [func(x, parameters_opt[0], parameters_opt[1], parameters_opt[2], parameters_opt[3]) for x in range(0, 120)], label="fit")
Graph.plot([temp_fringes[0][a], temp_fringes[0][a]], [0, 200], '--r', label="gekozen domein voor de fit")
Graph.plot([temp_fringes[0][b], temp_fringes[0][b]], [0, 200], '--r')
print(temp_fringes[0][a], temp_fringes[0][b])
Graph.xlim(temp_fringes[0][-1], temp_fringes[0][0])
Graph.ylim([0, 200])
Graph.xlabel("temperatuur ($^{\circ}$C)")
Graph.ylabel("aantal fringes")
Graph.legend()
Graph.savefig("fringes_over_temperatuur.png")
Graph.show()
