import numpy as np 



def rotate(data,theta): #逆時針旋轉

    x = data[0]
    y = data[1]

    theta = theta * np.pi / 180

    x_prime = x*np.cos(theta) - y*np.sin(theta)
    y_prime = x*np.sin(theta) + y*np.cos(theta) 

    new_data = [x_prime,y_prime,0]
    # print(new_data)

    return new_data


def change_all_data(data,theta):

    new_array = []
    for i in range (len(data)):       
        new_array.append(rotate(data[i],theta))

    print(new_array)
    return new_array




    







if __name__ == "__main__":
    a = np.array([[1,1,0],[1,2,0],[1.5,3,0],[2,2,0],[2,1,0]])
    theta = 50
    change_all_data(a,theta)
    # rotate(a, theta)
