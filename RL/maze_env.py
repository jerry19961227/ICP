import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time


class maze():

    def __init__(self):
        self.MAZE_H = 6
        self.MAZE_W = 6
        

        self.last_point = None
        self.next_point = None
        self.start_point = np.array([0,0])
        self.goal_coords = np.array([4,4])
        self.blocks_coords = np.array([[2,2],[2,3],[1,3]])
        self.flag = False
        self.reward = None
        self.fig = plt.figure()
        self.sub = plt.subplot(111)

        self.action_space = ['U', 'D', 'L', 'R']
        self.n_actions = len(self.action_space)
        plt.grid()
    
    def check_in_array(self,a,b):

        for i in range (len(b)):
            if a[0] == b[i][0] and a[1] == b[i][1]:
                return True
            else:
                pass
    
    def step(self, action):

        s = self.start_point
        base_action = np.array([0, 0])
        if action == 0:  # up 
            base_action[1] += 1

        elif action == 1:  # down
            base_action[1] -= 1

        elif action == 2:  # left
            base_action[0] -= 1
                
        elif action == 3:  # right
            base_action[0] += 1
        
        self.last_point = self.start_point
        self.next_point = self.start_point + base_action
        self.flag = False

        if (self.next_point == self.goal_coords).all():
            print("congrats")
            self.reward = 1
            self.flag = True
        elif self.check_in_array(self.next_point, self.blocks_coords) == True:
            print("bump into blocks")
            self.reward = -1
            self.flag = True
        elif self.next_point[0] < 0 or self.next_point[0] > 4 or self.next_point[1] < 0 or self.next_point[1] > 4 :
            print("out of bound")
            self.reward = -1
            self.flag = True
        else:
            self.flag = False
            self.reward = -0.1

        print("last_point:",self.last_point,"next point:", self.next_point, "flag:", self.flag, "r:",self.reward, "move dis:", base_action)

        return self.last_point, self.next_point, self.flag, self.reward

    def get_command(self):
        
        name = input('get command ')
        
        if name == 'w':
            self.last_point = self.start_point
            self.next_point = self.start_point + np.array([0,1])
            
        elif name == 's':
            self.last_point = self.start_point
            self.next_point = self.start_point + np.array([0,-1])
            
        elif name == 'a':
            self.last_point = self.start_point
            self.next_point = self.start_point + np.array([-1,0])
            
        elif name == 'd':
            self.last_point = self.start_point
            self.next_point = self.start_point + np.array([1,0])
        
        elif name == 'end':
            self.flag = True


        else:
            print("type again")
            self.next_point = self.start_point + np.array([0,0])

        self.start_point = self.next_point    
        # print("last point:", self.last_point)
        # print("next point:", self.next_point)
        
        return self.last_point, self.next_point, self.start_point, self.flag

   
    def draw_rect(self, x, y, color):
        x_ = x + 0.1
        y_ = y + 0.1
        color_ = color

        return plt.Rectangle((x_,y_), 0.8, 0.8, color = color_, alpha = 1)  


    def update_maze(self):

        if self.last_point is None:

            self.player_show = self.draw_rect(0, 0, 'red')
            self.sub.add_patch(self.player_show)
        
            
        elif self.last_point is not None :

            # self.player_show.remove()
            self.player = self.next_point
            self.player_show = self.draw_rect(self.player[0], self.player[1], 'red')
            self.sub.add_patch(self.player_show)

        print(self.last_point)
        plt.pause(0.1)
        # plt.grid()
        plt.show()
        self.player_show.remove()
   

    def build_maze(self):

        self.sub.set_xticks(np.arange(0, self.MAZE_H, 1))
        self.sub.set_yticks(np.arange(0, self.MAZE_W, 1))

        b = self.blocks_coords
        rect = self.draw_rect(b[0][0], b[0][1], 'black')
        rect1 = self.draw_rect(b[1][0], b[1][1], 'black')
        rect2 = self.draw_rect(b[2][0], b[2][1], 'black')
        # start = self.draw_rect(0, 0, 'red')
        # self.sub.add_patch(start)

        
        # self.goal_coords = np.array([4,4])
        goal = self.draw_rect(self.goal_coords[0], self.goal_coords[1], 'yellow')

        
        self.sub.add_patch(rect)
        self.sub.add_patch(rect1)
        self.sub.add_patch(rect2)
        self.sub.add_patch(goal)
        
        # goal.remove() 
        # plt.grid()
        plt.pause(1)
        self.title = 'Maze'
        self.fig.suptitle(self.title,fontsize = 18 )
        

class QLearningTable():

    def __init__(self, actions, learning_rate=0.02, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = np.zeros((4,5,5))
        


    def choose_action(self,next_point):

        state = next_point
        if np.random.uniform() < self.epsilon or state is None : # randomly choose action ,act greedy

            action = np.random.choice(self.actions)

        else:   
            action = np.argmax(self.q_table[:,state[0],state[1]])  #返回最大值的索引

        # print(action)
        return action

    def learn(self,r,a,last_point,next_point):

        old_state = last_point
        state = next_point
        if state[0] < 0 or state[0] > 4 or state[1] < 0 or state[1] > 4 :
            pass

        else:
            q_predict = self.q_table[a,old_state[0],old_state[1]]    #估計
            q_target = r + self.gamma * self.q_table[:,state[0],state[1]].max()  # 現實

            self.q_table[a,state[0],state[1]] += self.lr * (q_target - q_predict)
            # print(self.q_table[i][a])

def restart():
    flag2 = False
    if (m.next_point == m.goal_coords).all():
        print('THE END')
        flag2 = True
    else:
        m.last_point = None
        m.next_point = None
        m.start_point = np.array([0,0])
        print(RL.q_table)
        plt.show()
    
    print("last point:",m.last_point)
    print("last point:",m.next_point)
    return flag2


   
def update():

    while True:
        plt.ion()
        m.build_maze()
        num_of_state = 0
        while True:

            m.update_maze()

            action = RL.choose_action(m.next_point)

            last_point, next_point, flag, reward = m.step(action)

            RL.learn(reward,action,last_point,next_point)

            num_of_state += 1

            m.start_point = next_point

            if flag:
                flag2 = restart()
                break      
        if flag2:
            break

        






            






        
    

    
           

    
    

if __name__ == "__main__":

    m = maze()
    RL = QLearningTable(actions = list(range(m.n_actions)))
    update()
    
        
   
  

