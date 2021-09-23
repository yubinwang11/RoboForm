#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Du Yong, Wang Yubin

import math
import numpy
import numpy as np
import sys
import termios
import matplotlib.pyplot as plt
import random 
import rclpy

#from tkinter import Button
from matplotlib.widgets import Button
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from turtlebot3_msgs.msg import SensorState


from turtlebot3_example.turtlebot3_position_control.turtlebot3_path import Turtlebot3Path

terminal_msg = """
Turtlebot3 Position Control
------------------------------------------------------
From the current pose,
x: goal position x (unit: m)
y: goal position y (unit: m)
theta: goal orientation (range: -180 ~ 180, unit: deg)
------------------------------------------------------
"""


class Turtlebot3PositionControl(Node):

    def __init__(self):
        super().__init__('turtlebot3_position_control')

        """************************************************************
        ** Initialise variables
        ************************************************************"""
        self.form_num=1
        self.node_num=6
        self.trajectory_len=79
        self.time=0.1
        self.count=0.0
        self.position_init=1

        self.axis=5
        self.vw=self.axis/5

        self.Kv=0.22/2*self.vw
        self.Kw=2.84/2*self.vw/math.pi
        self.B_oz=0.0
        self.B_of=0.0
        #self.B_safed=0.1
        self.B_safed=0.08
        self.buttonstart=0


        self.r_o=np.array([0.2 for x in range(self.node_num)])
        self.d_d=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.d_1=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.d_2=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.d_t=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        
        self.x_cur_trajectory=np.array([[0.0 for x in range(self.trajectory_len+1)] for y in range(self.node_num)]) 
        self.y_cur_trajectory=np.array([[0.0 for x in range(self.trajectory_len+1)] for y in range(self.node_num)]) 

        self.x_head=np.array([0.0 for x in range(self.node_num)]) 
        self.y_head=np.array([0.0 for x in range(self.node_num)])

        self.x_left=np.array([0.0 for x in range(self.node_num)])
        self.y_left=np.array([0.0 for x in range(self.node_num)])
    
        self.x_right=np.array([0.0 for x in range(self.node_num)])
        self.y_right=np.array([0.0 for x in range(self.node_num)])

        self.x_curl=np.array([[0.0 for x in range(2)] for y in range(self.node_num)])
        self.y_curl=np.array([[0.0 for x in range(2)] for y in range(self.node_num)])
        
        self.x_curr=np.array([[0.0 for x in range(2)] for y in range(self.node_num)])
        self.y_curr=np.array([[0.0 for x in range(2)] for y in range(self.node_num)])
        
        self.x_body=[0.0 for x in range(self.node_num)]
        self.y_body=[0.0 for x in range(self.node_num)]

        self.x_o_body=[0.0 for x in range(self.node_num)]
        self.y_o_body=[0.0 for x in range(self.node_num)]

        self.x_tar_body=[0.0 for x in range(self.node_num)]
        self.y_tar_body=[0.0 for x in range(self.node_num)]


        self.v_cur=np.array([0.0 for x in range(self.node_num)])
        self.w_cur=np.array([0.0 for x in range(self.node_num)])

        
        
        self.pose_state_cur=np.array([0.0 for x in range(self.node_num)])

        self.odom = Odometry()
        self.twisttb30 = Twist()
        self.twisttb31 = Twist()
        self.twisttb32 = Twist()
        self.twisttb33 = Twist()
        self.twisttb34 = Twist()
        self.twisttb35 = Twist()

        self.gama=np.array([1.000 for x in range(self.node_num)])
       
        self.x_target=np.array([0.0 for x in range(self.node_num)])   
        self.y_target=np.array([0.0 for x in range(self.node_num)])
        self.d_target=np.array([0.0 for x in range(self.node_num)])

        self.x_cur=np.array([0.0 for x in range(self.node_num)])      #[2.0,2.0,2.0,2.0,2.0,2.0]
        self.y_cur=np.array([0.0 for x in range(self.node_num)])  

        self.x_per=np.array([0.0 for x in range(self.node_num)])       #[2.0,2.0,2.0,2.0,2.0,2.0]
        self.y_per=np.array([0.0 for x in range(self.node_num)])

        self.d_cur=np.array([1.0 for x in range(self.node_num)])
        self.q_cur=np.array([0.0 for x in range(self.node_num)])
        self.qd_cur=np.array([0.0 for x in range(self.node_num)])

        
        self.adjacency_matrix=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        
####################vector field configure  ###########################################################
        
        self.Fx=np.array([0.0 for x in range(self.node_num)])
        self.Fy=np.array([0.0 for x in range(self.node_num)])
        self.FFx=np.array([0.0 for x in range(self.node_num)])   
        self.FFy=np.array([0.0 for x in range(self.node_num)])        
        self.Fgx=np.array([0.0 for x in range(self.node_num)])
        self.Fgy=np.array([0.0 for x in range(self.node_num)])
        self.Fox=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.Foy=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.sigma=np.array([[1.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.sigma_multi=np.array([0.0 for x in range(self.node_num)])
        self.sigma_plus=np.array([0.0 for x in range(self.node_num)])

        self.x_o=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.y_o=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        
        self.d_o=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
        self.B_o=np.array([[0.0 for x in range(self.node_num)] for y in range(self.node_num)])
##########################################################################################################
        self.colorArr = ['r','g','b','c','m','k','gray','tan','pink','navy','r','g','b','c','m','k','gray','tan','pink','navy']
        
###########################################################################################################
        """************************************************************
        ** Initialise ROS publishers and subscribers
        ************************************************************"""
        qos = QoSProfile(depth=10)

        # Initialise publishers
        '''
        self.tb30_cmd_vel_pub = self.create_publisher(Twist,'tb3_0/cmd_vel', qos)
        self.tb31_cmd_vel_pub = self.create_publisher(Twist,'tb3_1/cmd_vel', qos)
        self.tb32_cmd_vel_pub = self.create_publisher(Twist,'tb3_2/cmd_vel', qos)
        self.tb33_cmd_vel_pub = self.create_publisher(Twist,'tb3_3/cmd_vel', qos)
        self.tb34_cmd_vel_pub = self.create_publisher(Twist,'tb3_4/cmd_vel', qos)
        self.tb35_cmd_vel_pub = self.create_publisher(Twist,'tb3_5/cmd_vel', qos)
        '''

        # Initialise subscribers
        '''
        self.odom_sub = self.create_subscription(Odometry, 'odom',self.odom_callback,qos)

        #####################position################################
        
        self.optitracktb30_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_0/pose',self.optitracktb30_callback,qos)
        self.optitracktb31_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_1/pose',self.optitracktb31_callback,qos)
        self.optitracktb32_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_2/pose',self.optitracktb32_callback,qos)
        self.optitracktb33_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_3/pose',self.optitracktb33_callback,qos)
        self.optitracktb34_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_4/pose',self.optitracktb34_callback,qos)
        self.optitracktb35_sub = self.create_subscription(PoseStamped,'vrpn_client_node/tb3_5/pose',self.optitracktb35_callback,qos)
        '''

        '''#####################sounds###################################      
        self.tb30_sensor_state_sub = self.create_subscription(SensorState,'/tb3_0/sensor_state',self.tb30_sensor_state_callback,qos)
        self.tb31_sensor_state_sub = self.create_subscription(SensorState,'/tb3_1/sensor_state',self.tb31_sensor_state_callback,qos)
        self.tb32_sensor_state_sub = self.create_subscription(SensorState,'/tb3_2/sensor_state',self.tb32_sensor_state_callback,qos)
        self.tb33_sensor_state_sub = self.create_subscription(SensorState,'/tb3_3/sensor_state',self.tb33_sensor_state_callback,qos)
        '''
        
        """************************************************************
        ** Initialise timers
        ************************************************************"""
        self.update_timer = self.create_timer(self.time, self.update_callback)  # unit: s
        self.get_logger().info("Turtlebot3 position control node has been initialised.")

    """*******************************************************************************
    ** Callback functions and relevant functions
    *******************************************************************************"""
    def odom_callback(self, msg):
        self.last_pose_x = msg.pose.pose.position.x
        self.last_pose_y = msg.pose.pose.position.y
        _, _, self.last_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        self.init_odom_state = True

    #####################position callback#################################### 

    def optitracktb30_callback(self, msg):
        self.x_cur[0] = msg.pose.position.x
        self.y_cur[0]= msg.pose.position.y
        _, _, self.d_cur[0] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[0] = True

    def optitracktb31_callback(self, msg):
        self.x_cur[1] = msg.pose.position.x
        self.y_cur[1]= msg.pose.position.y
        _, _, self.d_cur[1] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[1] = True

    def optitracktb32_callback(self, msg):
        self.x_cur[2] = msg.pose.position.x
        self.y_cur[2]= msg.pose.position.y
        _, _, self.d_cur[2] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[2] = True

    def optitracktb33_callback(self, msg):
        self.x_cur[3] = msg.pose.position.x
        self.y_cur[3]= msg.pose.position.y
        _, _, self.d_cur[3] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[3] = True

    def optitracktb34_callback(self, msg):
        self.x_cur[4] = msg.pose.position.x
        self.y_cur[4]= msg.pose.position.y
        _, _, self.d_cur[4] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[4] = True

    def optitracktb35_callback(self, msg):
        self.x_cur[5] = msg.pose.position.x
        self.y_cur[5]= msg.pose.position.y
        _, _, self.d_cur[5] = self.euler_from_quaternion(msg.pose.orientation)
        self.pose_state_cur[5] = True

    
    #####################update callback   10ms    ####################################  
    
    def update_callback(self):      
    
        self.transition()

        self.twisttb30.linear.x=self.v_cur[0]
        self.twisttb30.angular.z=self.w_cur[0]

        '''
        self.tb30_cmd_vel_pub.publish(self.twisttb30)
        '''

        '''
        print("self.v_cur[0]",self.v_cur[0])
        print("self.w_cur[0]",self.w_cur[0])
        
        print("self.x_cur[0]",self.x_cur[0])
        print("self.y_cur[0]",self.y_cur[0])
        '''    
        
        self.twisttb31.linear.x=self.v_cur[1]
        self.twisttb31.angular.z=self.w_cur[1]
        '''
        self.tb31_cmd_vel_pub.publish(self.twisttb31)
        '''

        '''
        print("self.v_cur[1]",self.v_cur[1])
        print("self.w_cur[1]",self.w_cur[1])
        print("self.x_cur[1]",self.x_cur[1])
        print("self.y_cur[1]",self.y_cur[1])
        '''

        self.twisttb32.linear.x=self.v_cur[2]
        self.twisttb32.angular.z=self.w_cur[2]
        '''
        self.tb32_cmd_vel_pub.publish(self.twisttb32)
        '''
        
        '''
        print("self.v_cur[2]",self.v_cur[2])
        print("self.w_cur[2]",self.w_cur[2])
        print("self.x_cur[2]",self.x_cur[2])
        print("self.y_cur[2]",self.y_cur[2])
        '''    

        self.twisttb33.linear.x=self.v_cur[3]
        self.twisttb33.angular.z=self.w_cur[3]
        '''
        self.tb33_cmd_vel_pub.publish(self.twisttb33)
        '''
        
        '''
        print("self.v_cur[3]",self.v_cur[3])
        print("self.w_cur[3]",self.w_cur[3])
        print("self.x_cur[3]",self.x_cur[3])
        print("self.y_cur[3]",self.y_cur[3])
        '''

        self.twisttb34.linear.x=self.v_cur[4]
        self.twisttb34.angular.z=self.w_cur[4]
        '''
        self.tb34_cmd_vel_pub.publish(self.twisttb34)
        '''

        self.twisttb35.linear.x=self.v_cur[5]
        self.twisttb35.angular.z=self.w_cur[5]
        '''
        self.tb35_cmd_vel_pub.publish(self.twisttb35)
        '''
        
       
    def transition(self):
        
        plt.figure(1)

        if self.buttonstart==0:
            axcut = plt.axes([0.0, 0.0, 1.0, 1.0])
            bcut = Button(axcut, 'Collision-Free Point-To-Point Transiction', color='pink', hovercolor='pink')
            bcut.on_clicked(self.start)
            plt.pause(5)
            
        
        if self.buttonstart==1:
        
            self.init_system()
            self.formation_ancl()
            self.unicycle_vector_field()
            self.unicycle_control_law()
            self.unicycle_simulation()
            

            for i in range (0,self.node_num,1):
                self.sigma_multi[i]=1.0
                self.sigma_plus[i]=0.0
               


    def start(self,event):
        if self.buttonstart==0:
            self.buttonstart=1
        else :
            self.buttonstart=0 
        print(self.buttonstart)    

    def init_system(self): 
      
        if self.position_init==1:


            for i in range (0,self.node_num,1):

                
                self.x_cur[i]=1*random.randint(-self.axis*2,self.axis*2) 
                self.y_cur[i]=1*random.randint(-self.axis*2,self.axis*2)

                self.x_per[i]=self.x_cur[i]
                self.y_per[i]=self.y_cur[i]


                for j in range(0,self.trajectory_len+1,1):
                    self.x_cur_trajectory[i][j]=self.x_cur[i]
                    self.y_cur_trajectory[i][j]=self.y_cur[i]
                



                for k in range(0,self.node_num,1):
                     
                    if k-i==1 or i-k==1:
                        self.adjacency_matrix[i][k]=1
           
            self.position_init=0

    def formation_ancl(self):

        A_x=np.array([-1.8, -0.9, 0.0, 0.0, 0.9,  1.8])
        A_y=np.array([-1.8,  0.0, 1.8, 0.0, 0.0, -1.8])

        N_x=np.array([-1.8, -1.8, 1.8, -0.5,  0.5,  1.8])
        N_y=np.array([-1.8,  1.8, 1.8,  0.6, -0.6, -1.8])

        C_x=np.array([-1.0, -1.0, 1.0, 0.0,  0.0,  1.0])
        C_y=np.array([-1.0,  1.0, 1.0, 2.0, -2.0, -1.0])

        L_x=np.array([-1.8, -1.8, -1.8, -1.8,  0.0,  1.8])
        L_y=np.array([-1.8, -0.6,  1.8,  0.6, -1.8, -1.8])
        

        dis_e=0
        for i in range(0,self.node_num,1):
            dis_e+= math.sqrt((self.x_cur[i]-self.x_target[i])**2+(self.y_cur[i]-self.y_target[i])**2)
        if dis_e<=0.15*self.node_num:
            self.form_num+=1    
       
        if self.form_num==1:
            self.x_target=A_x    
            self.y_target=A_y  
        if self.form_num==2:
            self.x_target=N_x    
            self.y_target=N_y  
        if self.form_num==3:
            self.x_target=C_x    
            self.y_target=C_y  
        if self.form_num==4:
            self.x_target=L_x    

            self.y_target=L_y  
           
        #if self.form_num==5:
        #    self.form_num=0  

    def unicycle_vector_field(self):
        
        


        for i in range(0,self.node_num,1):
             for j in range(0,self.node_num,1):
                 if i!=j:
                     self.x_o[i][j]=self.x_cur[j]
                     self.y_o[i][j]=self.y_cur[j]

        self.B_oz=-2*self.r_o[0]*(self.r_o[0]+self.B_safed)-(self.r_o[0]+self.B_safed)**2
        self.B_of=2*self.B_oz
        a=2/(self.B_oz-self.B_of)**3
        b=-3*(self.B_oz+self.B_of)/(self.B_oz-self.B_of)**3
        c=6*self.B_oz*self.B_of/(self.B_oz-self.B_of)**3
        d=self.B_oz**2*(self.B_oz-3*self.B_of)/(self.B_oz-self.B_of)**3
       
        
        ############ sigma configuer   #############################################################
        for i in range(0,self.node_num,1):
             for j in range(0,self.node_num,1):
                 if i!=j:
                     self.B_o[i][j]=self.r_o[i]**2-(self.x_cur[i]-self.x_o[i][j])**2-(self.y_cur[i]-self.y_o[i][j])**2


                     if self.B_o[i][j]>=self.B_oz: 
                         self.sigma[i][j]=0.0
                     elif self.B_o[i][j]<=self.B_of: 
                         self.sigma[i][j]=1.0
                     else:
                         self.sigma[i][j]=a*(self.B_o[i][j]**3)+b*(self.B_o[i][j]**2)+c*(self.B_o[i][j])+d


                     self.sigma_multi[i]=self.sigma_multi[i]*self.sigma[i][j] 
                     self.sigma_plus[i]=self.sigma_plus[i]+(1-self.sigma[i][j])     
        ''''''
        
        ############ attractive vector field   #############################################################

        for i in range(0,self.node_num,1):
        
            
             if math.sqrt((self.x_cur[i]-self.x_target[i])**2+(self.y_cur[i]-self.y_target[i])**2)==0 :
                 self.Fgx[i]=0
                 self.Fgy[i]=0
             else:
                 self.Fgx[i]=-(self.x_cur[i]-self.x_target[i])/math.sqrt((self.x_cur[i]-self.x_target[i])**2+(self.y_cur[i]-self.y_target[i])**2)
                 self.Fgy[i]=-(self.y_cur[i]-self.y_target[i])/math.sqrt((self.x_cur[i]-self.x_target[i])**2+(self.y_cur[i]-self.y_target[i])**2)
             
        ############ repulsive vector field   ########################################
        
        for i in range(0,self.node_num,1):
             for j in range(0,self.node_num,1):
                 if i!=j:
                     self.Fox[i][j]=(self.x_cur[i]-self.x_cur[j])/math.sqrt((self.x_cur[i]-self.x_cur[j])**2+(self.y_cur[i]-self.y_cur[j])**2)
                     self.Foy[i][j]=(self.y_cur[i]-self.y_cur[j])/math.sqrt((self.x_cur[i]-self.x_cur[j])**2+(self.y_cur[i]-self.y_cur[j])**2)
                     self.Fx[i]+=(1-self.sigma[i][j])*self.Fox[i][j]
                     self.Fy[i]+=(1-self.sigma[i][j])*self.Foy[i][j]





        ############  vector field   ########################################
        for i in range(0,self.node_num,1):
            
             self.FFx[i]=self.Fx[i]+self.sigma_multi[i]*self.Fgx[i]
             self.FFy[i]=self.Fy[i]+self.sigma_multi[i]*self.Fgy[i]

             self.q_cur[i]=math.atan2(self.FFy[i],self.FFx[i])

             self.Fx[i]=0
             self.Fy[i]=0
             


    def unicycle_control_law(self):
        for i in range(0,self.node_num,1):
            self.x_per[i]=self.x_cur[i]
            self.y_per[i]=self.y_cur[i] 

        for i in range(0,self.node_num,1):
            
            

            if self.sigma_multi[i]<1 and self.sigma_multi[i]>0:
                 self.v_cur[i]=0.8*self.Kv*math.tanh((self.x_cur[i]-self.x_target[i])**2+(self.y_cur[i]-self.y_target[i])**2)
            elif self.sigma_multi[i]==0:
                 self.v_cur[i]=0.5*self.Kv*math.tanh((self.x_cur[i]-self.x_target[i])**2+(self.y_cur[i]-self.y_target[i])**2)
            else:
                 self.v_cur[i]=self.Kv*math.tanh((self.x_cur[i]-self.x_target[i])**2+(self.y_cur[i]-self.y_target[i])**2)
            
            if self.v_cur[i]<=0.01:
                 self.v_cur[i]=0.01



            if self.d_cur[i]-self.q_cur[i]>=3.14:
                 self.w_cur[i]=-self.Kw*(self.d_cur[i]-self.q_cur[i]-6.28) #+self.qd_cur[i]
            elif self.d_cur[i]-self.q_cur[i]<=-3.14:
                 self.w_cur[i]=-self.Kw*(self.d_cur[i]-self.q_cur[i]+6.28) #+self.qd_cur[i]
            else :
                 self.w_cur[i]=-self.Kw*(self.d_cur[i]-self.q_cur[i]) #+self.qd_cur[i]

            if ((self.x_cur[i]-self.x_target[i])**2+(self.y_cur[i]-self.y_target[i])**2) <=0.0:
                 self.v_cur[i]=0
                 self.w_cur[i]=0

            ###########The following comments should be corrected when conducting physical experiment##############
            
            self.x_cur[i]=self.x_cur[i]+self.time*self.v_cur[i]*math.cos(self.d_cur[i])
            self.y_cur[i]=self.y_cur[i]+self.time*self.v_cur[i]*math.sin(self.d_cur[i])
            self.d_cur[i]=self.d_cur[i]+self.time*self.w_cur[i]
            
            if ((self.x_cur_trajectory[i][self.trajectory_len]-self.x_cur[i])**2+(self.y_cur_trajectory[i][self.trajectory_len]-self.y_cur[i])**2)>=0.001:


                 self.x_cur_trajectory[i][self.trajectory_len]=self.x_cur[i]
                 self.y_cur_trajectory[i][self.trajectory_len]=self.y_cur[i]
                 for j in range(0,self.trajectory_len,1):
                        
                    self.x_cur_trajectory[i][j]=self.x_cur_trajectory[i][j+1]
                    self.y_cur_trajectory[i][j]=self.y_cur_trajectory[i][j+1]
           
            
            if self.d_cur[i]>math.pi:
                self.d_cur[i]=self.d_cur[i]-2*math.pi
            if self.d_cur[i]<-math.pi:
                self.d_cur[i]=self.d_cur[i]+2*math.pi



        '''  
        self.twisttb30.angular.z=self.w_cur[0]
        self.twisttb30.linear.x=self.v_cur[0]
        self.tb30_cmd_vel_pub.publish(self.twisttb30)
         
        self.twisttb31.angular.z=self.w_cur[1]
        self.twisttb31.linear.x=self.v_cur[1]
        self.tb31_cmd_vel_pub.publish(self.twisttb31) 
        
        '''

##########################################################################################################################          
    def unicycle_simulation(self):
        roo=np.array([0.0 for x in range(self.node_num)])
        r = 0.2
        ro=self.B_safed+r
        sumx=0
        sumy=0
        
        theta = np.arange(0, 2*np.pi, 0.01)
        for i in range (0,self.node_num,1):

            self.x_body[i] = self.x_cur[i] + r * np.cos(theta)
            self.y_body[i] = self.y_cur[i] + r * np.sin(theta)

            self.x_tar_body[i] = self.x_target[i] + 0.2 * np.cos(theta)
            self.y_tar_body[i] = self.y_target[i] + 0.2 * np.sin(theta)

            self.x_o_body[i] = self.x_cur[i] + roo[i] * np.cos(theta)
            self.y_o_body[i] = self.y_cur[i] + roo[i] * np.sin(theta)

        for i in range (0,self.node_num,1): 
            
            self.x_head[i]=self.x_cur[i]+r*math.cos(self.d_cur[i])
            self.y_head[i]=self.y_cur[i]+r*math.sin(self.d_cur[i])

            self.x_left[i]=self.x_cur[i]-r*math.cos(self.d_cur[i]+1.57)
            self.y_left[i]=self.y_cur[i]-r*math.sin(self.d_cur[i]+1.57)

            self.x_right[i]=self.x_cur[i]-r*math.cos(self.d_cur[i]-1.57)
            self.y_right[i]=self.y_cur[i]-r*math.sin(self.d_cur[i]-1.57)

            self.x_curl[i][0]=self.x_head[i]
            self.x_curl[i][1]=self.x_cur[i]

            self.y_curl[i][0]=self.y_head[i]
            self.y_curl[i][1]=self.y_cur[i]

                          
        for i in range(0,self.node_num,1):
            
            ''''''
            plt.plot(self.x_curl[i], self.y_curl[i],self.colorArr[i])
            
            plt.plot(self.x_head[i],self.y_head[i],self.colorArr[i],self.x_left[i],self.y_left[i],self.colorArr[i],self.x_right[i],self.y_right[i],self.colorArr[i],self.x_cur[i],self.y_cur[i],self.colorArr[i],marker='.')
            
            plt.plot(self.x_body[i],self.y_body[i],self.colorArr[i],self.x_tar_body[i],self.y_tar_body[i],self.colorArr[i])

            

            plt.plot(self.x_target[i],self.y_target[i],self.colorArr[i])

            plt.plot(self.axis*3,self.axis*2,self.colorArr[i],marker='*')
            plt.plot(self.axis*3,-self.axis*2,self.colorArr[i],marker='*')
            plt.plot(-self.axis*3,self.axis*2,self.colorArr[i],marker='*')
            plt.plot(-self.axis*3,-self.axis*2,self.colorArr[i],marker='*')
                
            #plt.plot(self.x_cur_trajectory[i],self.y_cur_trajectory[i],self.colorArr[i]) 
        
            

            plt.text(self.x_cur[i], self.y_cur[i]-self.axis/10, '%.2f' %self.x_cur[i], ha='center', va= 'bottom',fontsize=6,color = self.colorArr[i])
            plt.text(self.x_cur[i], self.y_cur[i]-self.axis/5, '%.2f' %self.y_cur[i], ha='center', va= 'bottom',fontsize=6,color = self.colorArr[i])

            x=-self.axis*3
            y=self.axis*2
            d1=self.axis/2
            d2=self.axis/10 #0.5
           
            fsize=9
       
        x=-self.axis*3
        y=0
        d3=self.axis/10
        fsize=10

        
        plt.grid(True)
        plt.axis('equal')
        plt.axis([-self.axis*3,self.axis*3,-self.axis*2,self.axis*2])

        if self.count<=10000:
            plt.pause(0.01)
            plt.clf()
            self.count=self.count+1.0
        else :
            plt.show()   

             
########################################################################################################################
   
######################################################################################################################## 
 
########################################################################################################################       

    def get_key(self):
        # Print terminal message and get inputs
        print(terminal_msg)
        input_x = float(input("Input x: "))
        input_y = float(input("Input y: "))
        input_theta = float(input("Input theta: "))
        while input_theta > 180 or input_theta < -180:
            self.get_logger().info("Enter a value for theta between -180 and 180")
            input_theta = input("Input theta: ")
        input_theta = numpy.deg2rad(input_theta)  # Convert [deg] to [rad]

        settings = termios.tcgetattr(sys.stdin)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

        return input_x, input_y, input_theta

    """*******************************************************************************
    ** Below should be replaced when porting for ROS 2 Python tf_conversions is done.
    *******************************************************************************"""
    def euler_from_quaternion(self, quat):
        """
        Converts quaternion (w in last place) to euler roll, pitch, yaw
        quat = [x, y, z, w]
        """
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w*x + y*z)
        cosr_cosp = 1 - 2*(x*x + y*y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w*y - z*x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w*z + x*y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw