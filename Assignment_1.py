import library
from copy import deepcopy
import numpy as np


#Plot variable in different functions is for plotting residual vs iteration for that method

# Q1
a,b,c,d,e,f = np.loadtxt("Mat_A1.txt" , unpack = True)
matrix1=[]
for i in range(len(a)):
    matrix1.append([a[i],b[i],c[i],d[i],e[i],f[i]])

sol1 = [[19],[2],[13],[-7],[-9],[2]]


matrix1t = deepcopy(matrix1)
sol1t = deepcopy(sol1)


library.luDecompose(matrix1t,len(matrix1t))
library.forwardBackwardSubstitution(matrix1t,sol1t)
ans11 = deepcopy(sol1t)

matrix1t = deepcopy(matrix1)
sol1t = deepcopy(sol1)


for i in range(len(matrix1)):
    matrix1t[i].append(sol1t[i][0])


ans12 = library.gauss_jordan(matrix1t)

print(f'LU answer {ans11}') 
print(f'gauss jordan answer {ans12}')

#results
#[[-1.798534910432803], [0.940816002346099], [4.129218700359246], [-1.6656259928150743], [2.059813289669836], [0.18380825533370762]]
#[-1.7985349104328026, 0.9408160023460994, 4.129218700359247, -1.665625992815075, 2.0598132896698367, 0.18380825533370726]


# Q2

a,b,c,d,e,f = np.loadtxt("Mat_A2.txt" , unpack = True)
matrix2=[]
for i in range(len(a)):
    matrix2.append([a[i],b[i],c[i],d[i],e[i],f[i]])

sol2 = [[-5/3],[2/3],[3],[-4/3],[-1/3],[5/3]]

matrix2t = deepcopy(matrix2)
sol2t = deepcopy(sol2)

library.luDecompose(matrix2t , len(matrix2t))
library.forwardBackwardSubstitution(matrix2t,sol2t)
ans21 = deepcopy(sol2t)


matrix2t = deepcopy(matrix2)
sol2t = sol2t = [-5/3,2/3,3,-4/3,-1/3,5/3]
ans22 = library.jacobi(matrix2t,sol2t, 10**(-4))

print(f'LU answer {ans21}') 
print(f'jacobi answer {ans22}')

#results
#LU answer [[-0.3333333333333335], [0.33333333333333326], [0.9999999999999999], [-0.6666666666666665], [5.401084984662924e-17], [0.6666666666666667]]
#jacobi answer [-0.33351053724756563, 0.3332560592279631, 0.9999574730377104, -0.6668083208946178, -9.666687705900401e-05, 0.6666326712307783]


matrix2t = deepcopy(matrix2)
ans23 = library.Inverse(matrix2t,tol = 1e-4,xsolvername = "JacobiInv" ,plot = True)


matrix2t = deepcopy(matrix2)
ans24 = library.Inverse(matrix2t,tol = 1e-4,xsolvername = "Gauss-Seidel",plot=True)


matrix2t = deepcopy(matrix2)
ans25 = library.Inverse(matrix2t,tol = 1e-4,xsolvername = "ConjugateGrad",plot=True)

print(f'jacobi answer {ans23}') 
print(f'guass-seidel answer {ans24}')
print(f'Conjugate gradient answer {ans25}')

#results

#jacobi answer 
# [[0.93490644 0.86991763 0.25956983 0.20759851 0.41541074 0.16862289]
# [0.28993763 0.57999184 0.17304656 0.13844168 0.27694049 0.11246114]
# [0.08654205 0.17310924 0.32030542 0.05623057 0.11251243 0.10817512]
# [0.20759851 0.41541074 0.16862289 0.93490644 0.86991763 0.25956983]
# [0.13844168 0.27694049 0.11246114 0.28993763 0.57999184 0.17304656]
# [0.05623057 0.11251243 0.10817512 0.08654205 0.17310924 0.32030542]]
#guass-seidel answer 
# [[0.93495881 0.87003472 0.25962615 0.20766252 0.41546813 0.16869171]
# [0.28999592 0.58004411 0.17310924 0.13847025 0.27700437 0.11249186]
# [0.08656303 0.17314488 0.32032798 0.05625622 0.11253543 0.1082027 ]
# [0.20770537 0.41550656 0.1687378  0.93495881 0.87003472 0.25962615]
# [0.13848938 0.27702152 0.11251243 0.28999592 0.58004411 0.17310924]
# [0.0562631  0.1125416  0.1082101  0.08656303 0.17314488 0.32032798]]
#Conjugate gradient answer 
# [[0.97622704 0.87470864 0.25749913 0.26334709 0.42110279 0.16590037]
# [0.26075934 0.58151502 0.17032966 0.1541745  0.27934571 0.11099605]
# [0.10383786 0.17349607 0.31795613 0.1354747  0.11302106 0.10731257]
# [0.26334709 0.42110279 0.16590037 0.97622704 0.87470864 0.25749913]
# [0.1541745  0.27934571 0.11099605 0.26075934 0.58151502 0.17032966]
# [0.1354747  0.11302106 0.10731257 0.10383786 0.17349607 0.31795613]]



#Q3

def Fly_func(x,y, N, m = 0.2):
    # Creating lattice coordinates from lattice numbers
    i1 = x%N
    j1 = x//N
    i2 = y%N
    j2 = y//N

    # Condition for diagonal term
    if (x==y): return (m**2-1)

    # Condition for interaction terms
    if(((i1+1)%N,j1)==(i2,j2)): return 0.5
    if(((i1-1)%N,j1)==(i2,j2)): return 0.5
    if((i1,(j1+1)%N)==(i2,j2)): return 0.5
    if((i1,(j1-1)%N)==(i2,j2)): return 0.5

    return 0



N = 20
n = N*N
I = np.eye(n)
Inverse = np.zeros((n,n))

resl = []
for i in range(n):
    inv = library.ConjGrad_onfly(Fly_func, I[i],N, tol = 1e-4, max_iter = 20 ,plot=False)
    Inverse[:,i] = inv
    
