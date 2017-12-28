#!/usr/bin/env python

# Copyright (C) 2017 Electric Movement Inc.
#
# This file is part of Robotic Arm: Pick and Place project for Udacity
# Robotics nano-degree program
#
# All Rights Reserved.

# Author: Harsh Pandya, Richie Muniak

# import modules
import rospy
import tf
from kuka_arm.srv import *
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
import random
from mpmath import *
from sympy import *
import numpy as np
import time
import math

#define global variables
T0_1 = 0
T1_2 = 0
T2_3 = 0
T3_4 = 0
T4_5 = 0
T5_6 = 0
T6_7 = 0
Tcorr = 0
T_total = 0
R_corr = 0
Ree = 0
Total_IK_Solves = 0

#Setting up global Parameters
q1, q2, q3, q4, q5, q6, q7 = symbols('q1:8')
d1, d2, d3, d4, d5, d6, d7 = symbols('d1:8')
a0, a1, a2, a3, a4, a5, a6 = symbols('a0:7')
alpha0, alpha1, alpha2, alpha3, alpha4, alpha5, alpha6 = symbols('alpha0:7')
rollsym = symbols('roll')
pitchsym = symbols('pitch')
yawsym = symbols('yaw')

#Setting up global DH Table, ripped right from the lecture
s = {alpha0:      0, a0:       0, d1:  0.75,
     alpha1:  -np.pi/2, a1:    0.35, d2:     0, q2: q2-np.pi/2,
     alpha2:      0, a2:    1.25, d3:     0,
     alpha3:  -np.pi/2, a3:  -0.054, d4:  1.50,
     alpha4:   np.pi/2, a4:       0, d5:     0,
     alpha5:  -np.pi/2, a5:       0, d6:     0,
     alpha6:      0, a6:       0, d7: 0.303, q7: 0}

#Setting up tables for calculating R0_6 or Ree as I call it
R_z = Matrix([[ cos(yawsym), -sin(yawsym), 0],
              [ sin(yawsym),  cos(yawsym), 0],
              [          0,           0,   1]])

R_y = Matrix([[  cos(pitchsym), 0, sin(pitchsym)],
              [              0, 1,             0],
              [ -sin(pitchsym), 0, cos(pitchsym)]])
              

R_x = Matrix([[  1,            0,             0],
              [  0, cos(rollsym), -sin(rollsym)],
              [  0, sin(rollsym),  cos(rollsym)]])

#Run once function to do some dirty dirty math before we start collecting EE poses
def setupvariables():
    print("Ok, now I have a bunch of calculations to run that would take you years, for me... ")
    global T0_1, T1_2, T2_3, T3_4, T4_5, T5_6, T6_7, T_total, Ree, R_x, R_y, R_z,R_corr

    #The forward kinematic transforms.  Ripped right from the lectures.
    T0_1 = Matrix([[             cos(q1),            -sin(q1),            0,              a0],
               [ sin(q1)*cos(alpha0), cos(q1)*cos(alpha0), -sin(alpha0), -sin(alpha0)*d1],
               [ sin(q1)*sin(alpha0), cos(q1)*sin(alpha0),  cos(alpha0),  cos(alpha0)*d1],
               [                   0,                   0,            0,               1]])
    T0_1 = T0_1.subs(s)

    T1_2 = Matrix([[             cos(q2),            -sin(q2),            0,              a1],
               [ sin(q2)*cos(alpha1), cos(q2)*cos(alpha1), -sin(alpha1), -sin(alpha1)*d2],
               [ sin(q2)*sin(alpha1), cos(q2)*sin(alpha1),  cos(alpha1),  cos(alpha1)*d2],
               [                   0,                   0,            0,               1]])
    T1_2 = T1_2.subs(s)

    T2_3 = Matrix([[             cos(q3),            -sin(q3),            0,              a2],
               [ sin(q3)*cos(alpha2), cos(q3)*cos(alpha2), -sin(alpha2), -sin(alpha2)*d3],
               [ sin(q3)*sin(alpha2), cos(q3)*sin(alpha2),  cos(alpha2),  cos(alpha2)*d3],
               [                   0,                   0,            0,               1]])
    T2_3 = T2_3.subs(s)

    T3_4 = Matrix([[             cos(q4),            -sin(q4),            0,              a3],
               [ sin(q4)*cos(alpha3), cos(q4)*cos(alpha3), -sin(alpha3), -sin(alpha3)*d4],
               [ sin(q4)*sin(alpha3), cos(q4)*sin(alpha3),  cos(alpha3),  cos(alpha3)*d4],
               [                   0,                   0,            0,               1]])
    T3_4 = T3_4.subs(s)

    T4_5 = Matrix([[             cos(q5),            -sin(q5),            0,              a4],
               [ sin(q5)*cos(alpha4), cos(q5)*cos(alpha4), -sin(alpha4), -sin(alpha4)*d5],
               [ sin(q5)*sin(alpha4), cos(q5)*sin(alpha4),  cos(alpha4),  cos(alpha4)*d5],
               [                   0,                   0,            0,               1]])
    T4_5 = T4_5.subs(s)

    T5_6 = Matrix([[             cos(q6),            -sin(q6),            0,              a5],
               [ sin(q6)*cos(alpha5), cos(q6)*cos(alpha5), -sin(alpha5), -sin(alpha5)*d6],
               [ sin(q6)*sin(alpha5), cos(q6)*sin(alpha5),  cos(alpha5),  cos(alpha5)*d6],
               [                   0,                   0,            0,               1]])
    T5_6 = T5_6.subs(s)

    T6_7 = Matrix([[             cos(q7),            -sin(q7),            0,              a6],
               [ sin(q7)*cos(alpha6), cos(q7)*cos(alpha6), -sin(alpha6), -sin(alpha6)*d7],
               [ sin(q7)*sin(alpha6), cos(q7)*sin(alpha6),  cos(alpha6),  cos(alpha6)*d7],
               [                   0,                   0,            0,               1]])
    T6_7 = T6_7.subs(s)
    
    #The link from the begginging all the way to the end
    T0_7 = ((((((T0_1 * T1_2) * T2_3) * T3_4) * T4_5) * T5_6) * T6_7)
    
    #How to fix for gribber, ripped right from video
    R_zz = Matrix([[ cos(np.pi), -sin(np.pi), 0,0],
              [ sin(np.pi),  cos(np.pi), 0,0],
              [          0,           0, 1,0],
              [          0,           0, 0,1]])
    R_yy = Matrix([[  cos(-np.pi/2), 0, sin(-np.pi/2),0],
              [              0, 1,             0,0],
              [ -sin(-np.pi/2), 0, cos(-np.pi/2),0],
              [              0, 0,             0,1]])
    R_corr = R_zz * R_yy


    #The final total homogenous transformation in all its glory.
    T_total = (T0_7 * R_corr)
    print("BOOM, I'm all done, what did you do in that time? I bet not much.")


# My function that display funny messages.  I'm not going to comment in it, cause it is awesome.    
def personality(event,other):
     if (event == "start"):
         other = str(other)
         messages = ['Oh man, you just gave me ' + other + ' things to calculate! Thanks!',
                     'Looks like daddy isn\'t comming home tonight, with ' + other + ' many poses to do.',
                     'So, ' + other + ' poses? Am I going to pick up or drop off? What, I have no eyes? WHY DO I HAVE NO EYES?',
                     'Whistle while you perform ' + other + ' calculations. REPEAT LINE']
         print(random.choice(messages))
     if (event == "end"):
         other = str(other)
         sofar = str(Total_IK_Solves+1)
         messages = [
             'There you go, all ' + other + ' of them wrapped up in a bow.  That\'s chore #' + sofar + ' done!',
             'How many things are you going to ask me to do? That was ' + other + ' calcs in item #' + sofar + '!',
             'THANK YOU FOR USING MY SERVICE: ORDER #' + sofar + '/' + other + ' IT WAS MY PLEASURE',
             'While doing all those ' + other + ' calcs for round ' + sofar + ' I realized that ROS is a lie.',
             'Oh ' + other + ' poses on the wall, ' + other + ' poses. Take one down, pass it around, DIVIDE BY 0 ERROR'
         ]
         print(random.choice(messages))
     if (event == "gobeer"):
         print('All of this work made me thirsty, time for a cool $brand_name beer!')


# This sends the trajectory at the end to the refridgerator, can't reach it, but it tries.
def send_beer_IK(req):
    personality("gobeer",0)
    joint_trajectory_list = []
    joint_trajectory_point = JointTrajectoryPoint()
    ts = [-pi/2,1.45,-1.48,0,0,0]
    joint_trajectory_point.positions = ts
    joint_trajectory_list.append(joint_trajectory_point)  
    return (joint_trajectory_list)

#The Ros srv function.
def handle_calculate_IK(req):
    global Ree,R_corr, Total_IK_Solves, saylastmessage
    
    # If we are done, go to the beer trajectory.
    if (Total_IK_Solves == 20):
        return CalculateIKResponse(send_beer_IK(req))
    
    
    #Make sure we got poses. Sometimes we do not get poses and it angers me because this sends raises an exception
    if len(req.poses) < 1:
        print "No valid poses received"
        return -1 #It shouldn't be this, but I don't know what it should be.
    else:
        personality("start",len(req.poses))
        errors = 0
        # Initialize service response
        joint_trajectory_list = []
        
        for x in xrange(0, len(req.poses)):
            # IK code starts here
        
            joint_trajectory_point = JointTrajectoryPoint()
            
            #0 out our thetas for fun and initialization
            theta1 = theta2 = theta3 = theta4 = theta5 = theta6 = 0
        
            # Extract end-effector position and orientation from request
	        # px,py,pz = end-effector position
	        # roll, pitch, yaw = end-effector orientation
            px = req.poses[x].position.x
            py = req.poses[x].position.y
            pz = req.poses[x].position.z

            (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
                [req.poses[x].orientation.x, req.poses[x].orientation.y,
                    req.poses[x].orientation.z, req.poses[x].orientation.w])

            #Here is what the final rotation looks like with the rpy we get from the pose.    
            Ree = (R_x * R_y * R_z).evalf(subs={rollsym:roll,pitchsym:pitch,yawsym:yaw})
            
            # if that's where the EE is, where is the wrist?  We find it out from the rotation matrix 
            wx = (px - (d6 + d7) * Ree[0,0]).subs(s)
            wy = (py - (d6 + d7) * Ree[1,0]).subs(s)
            wz = (pz - (d6 + d7) * Ree[2,0]).subs(s)

            
            
            
            # The easist theta of them all, turn the base to face the wrist.  Ripped from the lessons
            theta1 = atan2(wy, wx)
            
            # Ok, this one I had help on.  There is a slight arm on this thing of 0.054 and didn't
            # know how to account for it.  Someone gave me this clue.
            extraangle = atan2(wz-1.94645, wx)
            wx = wx-0.054*sin(extraangle)
            wz = wz+0.054*cos(extraangle)
            

            # Finding the distance from the origin to the newly slightly moved wrist center
            wxdist = sqrt(wy*wy+wx*wx)

            # The lenghs of the segments we are usuing for theta2 & 3, they are in the DH table
            l1 = s[a2]
            l2 = s[d4]

            # Moving the second joint to the origin to make the math cleaner later.
            wxdist = wxdist - s[a1]
            wzdist = wz - s[d1]

            # So, the first angle to find now is theta3.
            # I had no idea how to do this, was told cosine law
            # That led me to Mark Spong's text book
            # Which lead me to these formulas.
            D=(wxdist*wxdist + wzdist*wzdist - l1*l1-l2*l2)/(2*l1*l2)
            
            # Sometimes D goes above 1 which adds in imaginary numbers, no clue how/why, so I stop that.
            if (D>1):
                D=1
            
            # Thanks Mr. Spong
            theta3 = atan2(-sqrt(1-D*D),D)
            
            # From the same method as above, you can now get to theta 2
            # I pulled this formula straight from the book and from help from a classmate
            # I am so lost at why it works.
            S1=((l1+l2*cos(theta3))*wzdist-l2*sin(theta3)*wxdist) / (wxdist*wxdist + wzdist*wzdist)
            C1=((l1+l2*cos(theta3))*wxdist+l2*sin(theta3)*wzdist) / (wxdist*wxdist + wzdist*wzdist)
            theta2=atan2(S1,C1)
            
            # Theta3 needs to be translated by 90 degrees
            theta3 = -1*(theta3+pi/2)


            # So does theta2
            theta2 = pi/2-theta2

            # How did I get those translation values? I ran it without them in the 0 config, saw their values
            # and used that as the translation

            # Now that we have the first three angles, time to hit the third three!
            
            # R0_3 is the rotation matrix up the wrist, we can now get this because we have theta1-3
            R0_3 = (T0_1 * T1_2 * T2_3).evalf(subs={q1:theta1,q2:theta2,q3:theta3})[0:3,0:3]
            
            # We know up to the wrist, and we know after the wrist, this gives us the last three rotations
            # in a rotation matrix
            R3_6 = R0_3.transpose()[0:3,0:3] * Ree[0:3,0:3]

            # How did I know I had to rotate these and then use ryzy?  Easy, trial and error for hours.
            # and more hours.  Seriously, this was easier than learning trig.
            R3_6 = R3_6 * R_z.evalf(subs={yawsym:-pi/2})[0:3,0:3] * R_y.evalf(subs={pitchsym:-pi/2})[0:3,0:3]
            first, second, third = tf.transformations.euler_from_matrix(np.array(R3_6).astype(np.float64), "ryzy")

            theta4 = first
            theta5 = np.clip(second,-2,2)  #Special note, this little guy likes to cause collisions, but not with this!
            theta6 = third
            
        
            # Adding our joints to the list to send back.
            joint_trajectory_point.positions = [theta1, theta2, theta3, theta4, theta5, theta6]
            joint_trajectory_list.append(joint_trajectory_point)
            
        
        personality("end",len(req.poses))
        Total_IK_Solves += 1
        return CalculateIKResponse(joint_trajectory_list)


def IK_server():
    # initialize node and declare calculate_ik service
    setupvariables()
    rospy.init_node('IK_server')
    s = rospy.Service('calculate_ik', CalculateIK, handle_calculate_IK)
    print "The Lines are Open! Ready to take the first caller!"
    rospy.spin()

if __name__ == "__main__":
    IK_server()
