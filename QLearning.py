import numpy as np
import pandas as pd
import time

#初始化参数
N_STATE = 6
ACTIONS = ['left','right']
EPSION = 0.9
ALPHA = 0.5
LAMBDA = 0.9
EP = 13
FRESH_TIME = 0.03

#初始值为0的q表
def build_q_table(n_state,actions):
    Data_Frame = pd.DataFrame(
        np.zeros((n_state,len(actions))),
        columns=actions
    )
    return  Data_Frame

def choose_action(state,q_table):
    state_actios = q_table.iloc[state,:]
    if np.random.uniform()>EPSION or state_actios.loc['right']==state_actios.loc['left']:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = ACTIONS[state_actios.argmax()]
    return action_name

def get_env_feedback(S,A):
    if A == 'right':
        if S == N_STATE - 2:
            S_ = 'terminal'
            R = 1
        else:
            R = 0
            S_ = S+1
    else:
        R = 0
        if S ==0:
            S_ = S
        else:
            S_ = S-1
    return S_,R

def update_env(S,episode,step_counter):
    env_list = ['-']*(N_STATE-1)+['T']
    if S == 'terminal':
        interaction = 'episode %s: toral_steps = %s' % (episode+1,step_counter)
        print('\r{}'.format(interaction),end='')
        time.sleep(2)
        print('\r                          ',end='')
    else:
        env_list[S] = 'o'
        interantion = ''.join(env_list)
        print('\r{}'.format(interantion),end = '')
        time.sleep(FRESH_TIME)

def rl():
    q_table = build_q_table(N_STATE,ACTIONS)
    for ep in range(EP):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S,ep,step_counter)
        while not is_terminated:
            A = choose_action(S,q_table)
            S_,R = get_env_feedback(S,A)
            q_predict = q_table.loc[S, A]
            if S_ !='terminal':
                q_target = R + LAMBDA*q_table.iloc[S_,:].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S,A] +=ALPHA*(q_target-q_predict)
            S = S_
            update_env(S,ep,step_counter+1)
            step_counter +=1
    return q_table

table = rl()