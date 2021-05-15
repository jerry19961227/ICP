import numpy as np 
import matplotlib.pyplot as plt


def fitting_function():

    dis = []

    for i in range(len(M)):
        x = M[i][0]
        y = M[i][1]
        for j in range(len(M)):
            x1 = P[j][0]
            y1 = P[j][1]

            z = (x - x1)**2 + (y - y1) **2

            if z < Threshold:
                dis.append(z)
                print(P[j][0],P[j][1])
            else:
                pass
    # movement = [abs(dis[i] - dis1[i])for i in range(len(dis))]

    return  dis


def plot_data(M, P):

    x = list() 
    y = list()
    x1 = list()
    y1 = list()
    for i in range(len(M)):
        x.append(M[i][0])
        y.append(M[i][1])
        x1.append(P[i][0])
        y1.append(P[i][1])

 
    fig = plt.figure()
    ax = plt.subplot()
    area = [200]
    
    ax.scatter(x, y, s=area, alpha=0.5, edgecolors='BLUE')
    ax.scatter(x1, y1, s=area, alpha=0.5, edgecolors='YELLOW')
    plt.title('initial point sets')
    plt.show()


def rotation_vector(): 
    
    mass_center_of_M = np.array([sum(M[:,0])/len(M), sum(M[:,1])/len(M), sum(M[:,2])/len(M)])
    mass_center_of_P = np.array([sum(P[:,0])/len(P), sum(P[:,1])/len(P), sum(P[:,2])/len(P)])
    # print("mass_center_of_M:",mass_center_of_M)
    # print("mass_center_of_P:",mass_center_of_P)
   
    return mass_center_of_M,mass_center_of_P
    
   

def cross_covariance_matrix(mass_center_of_M , mass_center_of_P , num_of_data):

    M_result = []
    for i in range(len(M)):
        m = M[i] - mass_center_of_M
        M_result.append(m)

    M_result = np.array(M_result)
    M_result = np.transpose(M_result)
    print(M_result)

    P_result = []
    for i in range(len(P)):
        p = P[i] - mass_center_of_P
        P_result.append(p)

    P_result = np.array(P_result)
    print(P_result)

    pm = M_result.dot(P_result)/len(P)
    print("pm:",pm)



    return pm


def rotation_matrix(pm):        #pm為m資料集和p資料集之間的斜方差矩陣

    Q11 = pm[0][0] + pm[1][1] + pm[2][2]  
    Q12 = pm[1][2] - pm[2][1]
    Q13 = pm[2][0] - pm[0][2]
    Q14 = pm[0][1] - pm[1][0]
    Q21 = pm[1][2] - pm[2][1]
    Q22 = pm[0][0] - pm[1][1] - pm[2][2]  
    Q23 = pm[0][1] + pm[1][0]
    Q24 = pm[0][2] + pm[2][0]
    Q31 = pm[2][0] - pm[0][2]
    Q32 = pm[0][1] + pm[1][0]
    Q33 = pm[1][1] - pm[0][0] - pm[2][2]
    Q34 = pm[1][2] + pm[2][1]
    Q41 = pm[0][1] - pm[1][0]
    Q42 = pm[0][2] + pm[2][0]
    Q43 = pm[1][2] + pm[2][1]
    Q44 = pm[2][2] - pm[0][0] - pm[1][1]

    # Q = np.zeros([4,4])
    Q = np.array([[Q11 ,Q12 ,Q13 ,Q14],
                    [Q21 ,Q22 ,Q23 ,Q24],
                    [Q31 ,Q32 ,Q33 ,Q34],
                    [Q41 ,Q42 ,Q43 ,Q44]])

    print("Q:",Q)
    Q = np.mat(Q)
    e_vals,e_vecs = np.linalg.eig(Q)
    e_vecs = e_vecs.transpose()
    print("e_vals:",e_vals)
    print("e_vecs:",e_vecs)
    max_e_vals = 0
    for i in range (len(e_vals)):           #找出最大特徵質

        if max_e_vals < e_vals[i] :
            max_e_vals = e_vals[i]

        elif max_e_vals > e_vals[i]:
            pass

    print("biggest eigenvalue:",max_e_vals)
    f_e_vecs = 0
    for i in range (len(e_vals)):           #找出最大特徵質所對應特徵向量

        if max_e_vals == e_vals[i]:
            f_e_vecs = e_vecs[i]
    
    print("eigenvector corresponding to the biggest eigenvalue:",f_e_vecs)
    q = f_e_vecs[0][0]
    print("q:", q)
    q =  [0.9660429  ,0.     ,0.     ,-0.25838171]
    # q = [1, 0 , 0 ,1.740705681382165e-16]
    R11 = np.square(q[0]) + np.square(q[1]) - np.square(q[2]) - np.square(q[3])
    R12 = 2 * (q[1]*q[2] - q[0]*q[3])
    R13 = 2 * (q[1]*q[3] + q[0]*q[2])
    R21 = 2 * (q[1]*q[2] + q[0]*q[3])
    R22 = np.square(q[0]) - np.square(q[1]) + np.square(q[2]) - np.square(q[3])
    R23 = 2 * (q[2]*q[3] - q[0]*q[1])
    R31 = 2 * (q[1]*q[3] - q[0]*q[2])
    R32 = 2 * (q[2]*q[3] + q[0]*q[1])
    R33 = np.square(q[0]) - np.square(q[1]) - np.square(q[2]) - np.square(q[3])

    R = np.array([[R11 ,R12 ,R13],
                [R21 ,R22 ,R23],
                [R31 ,R32 ,R33]])

    print("Rotation Matrix:\n",R)

    
    return R


def iteration():



    error = 0

def translation(mass_center_of_M, mass_center_of_P, R):

    p_prime = []

    for i in range(len(P)):
        z = P[i].dot(R)
        p_prime.append(z)

    print("p_prime:" , p_prime)

    T = mass_center_of_M - mass_center_of_P.dot(R)
    p_prime = p_prime + T
    
    return p_prime

def iteration():



    error = 0


def show_converge(p_prime):
    x = list() 
    y = list()
    x1 = list()
    y1 = list()
    for i in range(len(M)):
        x.append(M[i][0])
        y.append(M[i][1])
        x1.append(p_prime[i][0])
        y1.append(p_prime[i][1])


    
    fig = plt.figure()
    ax = plt.subplot()
    area = [200]
    
    ax.scatter(x, y, s=area, alpha=0.5, edgecolors='BLUE')
    ax.scatter(x1, y1, s=area, alpha=0.5, edgecolors='YELLOW')

    plt.title('point sets after convergence')
    plt.show()


    
if __name__ == "__main__":
    M = np.array([[1,0,0],[1,1,0],[2,1,0],[2,0,0]])
    P = np.array([[0,-1,0],[0.5,-0.114,0],[1.336,-0.634,0],[0.866,-1.5,0]])

    mass_center_of_M = rotation_vector()[0]
    mass_center_of_P = rotation_vector()[1]
    num_of_data = len(M)

    # print(rotation_vector())
    plot_data(M, P)

    pm = cross_covariance_matrix(mass_center_of_M ,mass_center_of_P ,num_of_data)
    R = rotation_matrix(pm)
    print("cross_covariance_matrix_pm:", cross_covariance_matrix(mass_center_of_M, mass_center_of_P, num_of_data))
    R = rotation_matrix(cross_covariance_matrix(mass_center_of_M,mass_center_of_P,num_of_data))
    p_prime = translation(mass_center_of_M, mass_center_of_P, R)
    show_converge(p_prime)
   


    
 



    
    
   
    
   
 
