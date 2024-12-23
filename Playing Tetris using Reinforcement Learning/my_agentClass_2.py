import copy
import pickle
import numpy as np
import random
import h5py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# This file provides the skeleton structure for the classes TQAgent and TDQNAgent to be completed by you, the student.
# Locations starting with # TO BE COMPLETED BY STUDENT indicates missing code that should be written by you.

class TQAgent:
    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,episode_count):
        self.reward_tots = [0]*episode_count
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.episode=0
        self.episode_count=episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        self.states = 2 ** (self.gameboard.N_row * self.gameboard.N_col)
        self.actions = [4 + 3, 3 * 2, 3 * 4, 3]
        self.Q_table = [np.zeros((self.states, action)) for action in self.actions]
        self.rewards = np.zeros(self.episode_count)
        
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions and Q-table and storage for the rewards
        # This function should not return a value, store Q table etc as attributes of self

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.episode_count' the total number of episodes in the training
        


    def fn_load_strategy(self,strategy_file):
        with h5py.File(strategy_file, "r") as hf:
            self.Q_table[0] = np.array(hf['Q_table_pillar'])
            self.Q_table[1] = np.array(hf['Q_table_slash'])
            self.Q_table[2] = np.array(hf['Q_table_L'])
            self.Q_table[3] = np.array(hf['Q_table_square'])
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-table (to Q-table of self) from the input parameter strategy_file (used to test how the agent plays)

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as an integer entry in the Q-table
        # This function should not return a value, store the state as an attribute of self
        # 从空的二进制字符串开始
        state_binary = ""

        # 遍历游戏板上的每个单元格
        for index_row in range(self.gameboard.N_row):
            for index_col in range(self.gameboard.N_col):
                # 检查当前单元格的值，并向状态二进制字符串中添加 '1' 或 '0'
                if self.gameboard.board[index_row, index_col] == 1:
                    state_binary += '1'
                else:
                    state_binary += '0'
        
        # 将二进制字符串转换为整数
        state_integer = int(state_binary, 2)
        
        # 在 self.state 中为当前剧集存储状态属性
        self.state = state_integer
        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    def index_to_action(self, action_index):
        
        if self.gameboard.cur_tile_type == 0:
            
            if action_index < 4:
                return (action_index, 0)
            
            else:               
                return (action_index - 4, 1)
            
        elif self.gameboard.cur_tile_type == 1:
            
            if action_index < 3:                
                return (action_index, 0)
            
            else:                
                return (action_index - 3, 1)
            
        elif self.gameboard.cur_tile_type == 2:
            
            if action_index < 3:               
                return (action_index, 0)
            
            elif action_index < 6:               
                return (action_index - 3, 1)
            
            elif action_index < 9:                
                return (action_index - 6, 2)
            
            else:
                return (action_index - 9, 3)
            
        elif self.gameboard.cur_tile_type == 3:            
            return (action_index, 0)

    def fn_select_action(self):
        if np.random.rand()<self.epsilon:
            # Select random action
            self.action_index = np.random.randint(self.actions[self.gameboard.cur_tile_type])
        else:
            # Select action with highest Q-value
            self.action_index=np.argmax(self.Q_table[self.gameboard.cur_tile_type][self.state])

        tile_x, tile_orientation = self.index_to_action(self.action_index)
        if self.gameboard.fn_move(tile_x, tile_orientation):
            raise Exception("Movement failed!")
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the Q-table or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not
    
    def fn_reinforce(self, old_state, reward):
        old_tile_type = old_state[1]
        old_gameboard_state = old_state[0]
        
        # 获取当前状态和动作对应的 Q 值
        current_q_value = self.Q_table[old_tile_type][old_gameboard_state][self.action_index]
        
        # 计算在下一个状态下所有可能动作的最大 Q 值
        next_max_q_value = np.max(self.Q_table[self.gameboard.cur_tile_type][self.state])
        
        # 计算 Q 值更新的增量
        delta_q = reward + next_max_q_value - current_q_value
        
        # 更新 Q 表
        self.Q_table[old_tile_type][old_gameboard_state][self.action_index] += self.alpha * delta_q
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q table using state and action stored as attributes in self and using function arguments for the old state and the reward
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # 'self.alpha' learning rate

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.mean(self.rewards[self.episode-100:self.episode])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    print("Saving data at episode", self.episode)
                    print("Current working directory:", os.getcwd())
                    with h5py.File("Data_" + str(self.episode) + ".h5", "w") as hf:
                        hf.create_dataset('Q_table_pillar', data=np.array(self.Q_table[0]))
                        hf.create_dataset('Q_table_slash', data=np.array(self.Q_table[1]))
                        hf.create_dataset('Q_table_L', data=np.array(self.Q_table[2]))
                        hf.create_dataset('Q_table_square', data=np.array(self.Q_table[3]))
                        hf.create_dataset('rewards', data=np.array(self.rewards))
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-table to data files for plotting of the rewards and the Q-table can be used to test how the agent plays
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later passed to fn_reinforce()
            old_state = (self.state, self.gameboard.cur_tile_type)
            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.rewards[self.episode] += reward
            # Read the new state
            self.fn_read_state()
            # Update the Q-table using the old state and the reward (the new state and the taken action should be stored as attributes in self)
            self.fn_reinforce(old_state, reward)

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TDQNAgent:


    # Agent for learning to play tetris using Q-learning
    def __init__(self,alpha,epsilon,epsilon_scale,replay_buffer_size,batch_size,sync_target_episode_count,episode_count):
        # Initialize training parameters
        self.alpha=alpha
        self.epsilon=epsilon
        self.epsilon_scale=epsilon_scale
        self.replay_buffer_size=replay_buffer_size
        self.batch_size=batch_size
        self.sync_target_episode_count=sync_target_episode_count
        self.episode=0
        self.episode_count=episode_count
        self.reward_tots=[0]*episode_count

    def fn_init(self,gameboard):
        self.gameboard=gameboard
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could set up and initialize the states, actions, the Q-networks (one for calculating actions and one target network), experience replay buffer and storage for the rewards
        # You can use any framework for constructing the networks, for example pytorch or tensorflow
        # This function should not return a value, store Q network etc as attributes of self
        self.states = self.gameboard.N_row * self.gameboard.N_col
        self.actions = [4 + 3, 3 * 2, 3 * 4, 3]
        
        self.model = DQN(self.states + len(self.actions), 4 * 4)
        self.model_target = copy.deepcopy(self.model)
        
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.alpha)
        
        self.exp_buffer = []

        # Useful variables: 
        # 'gameboard.N_row' number of rows in gameboard
        # 'gameboard.N_col' number of columns in gameboard
        # 'len(gameboard.tiles)' number of different tiles
        # 'self.alpha' the learning rate for stochastic gradient descent
        # 'self.episode_count' the total number of episodes in the training
        # 'self.replay_buffer_size' the number of quadruplets stored in the experience replay buffer

    def fn_load_strategy(self,strategy_file):
        # Load strategy from file
        self.model.load_state_dict(torch.load(strategy_file))
        pass
        # TO BE COMPLETED BY STUDENT
        # Here you can load the Q-network (to Q-network of self) from the strategy_file

    def fn_read_state(self):
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # In this function you could calculate the current state of the gane board
        # You can for example represent the state as a copy of the game board and the identifier of the current tile
        # This function should not return a value, store the state as an attribute of self
        tile_state = self.gameboard.board.flatten()
        
        tile_type = -np.ones(len(self.gameboard.tiles))
        tile_type[self.gameboard.cur_tile_type] = 1
        
        self.state = np.append(tile_state, tile_type)
        # Useful variables: 
        # 'self.gameboard.N_row' number of rows in gameboard
        # 'self.gameboard.N_col' number of columns in gameboard
        # 'self.gameboard.board[index_row,index_col]' table indicating if row 'index_row' and column 'index_col' is occupied (+1) or free (-1)
        # 'self.gameboard.cur_tile_type' identifier of the current tile that should be placed on the game board (integer between 0 and len(self.gameboard.tiles))

    #def get_valid_action(self, sorted_out):
    #    for idx in sorted_out:
    #        rotation = int(idx / 4)
    #        position = idx % 4
    #        if not self.gameboard.fn_move(position, rotation): # If the move is possible, break
    #            return idx   

    def fn_select_action(self):
        self.model.eval()
        out = self.model(torch.tensor(self.state)).detach().numpy()
        if np.random.rand() < max(self.epsilon, 1-self.episode/self.epsilon_scale): # epsilon-greedy
            self.action = random.randint(0, (4*4)-1)
        else: 
            self.action = np.argmax(out)
        
        rotation = int(self.action / 4)
        position = self.action % 4
        self.gameboard.fn_move(position, rotation)
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Choose and execute an action, based on the output of the Q-network for the current state, or random if epsilon greedy
        # This function should not return a value, store the action as an attribute of self and exectute the action by moving the tile to the desired position and orientation

        # Useful variables: 
        # 'self.epsilon' parameter epsilon in epsilon-greedy policy
        # 'self.epsilon_scale' parameter for the scale of the episode number where epsilon_N changes from unity to epsilon

        # Useful functions
        # 'self.gameboard.fn_move(tile_x,tile_orientation)' use this function to execute the selected action
        # The input argument 'tile_x' contains the column of the tile (0 <= tile_x < self.gameboard.N_col)
        # The input argument 'tile_orientation' contains the number of 90 degree rotations of the tile (0 < tile_orientation < # of non-degenerate rotations)
        # The function returns 1 if the action is not valid and 0 otherwise
        # You can use this function to map out which actions are valid or not

    def fn_reinforce(self,batch):
        targets = []
        action_value = []
        self.model.train()
        self.model_target.eval()

        for transition in batch:
            state = transition[0]
            action = transition[1]
            reward = transition[2]
            next_state = transition[3]
            terminal = transition[4]

            y = reward
            if not terminal:
                out = self.model_target(torch.tensor(next_state)).detach().numpy()
                y += max(out)
            targets.append(torch.tensor(y, dtype=torch.float64))
            out = self.model(torch.tensor(state))
            action_value.append(out[action])
            
            

        targets = torch.stack(targets)
        action_value = torch.stack(action_value)
        loss = self.criterion(action_value, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # TO BE COMPLETED BY STUDENT
        # This function should be written by you
        # Instructions:
        # Update the Q network using a batch of quadruplets (old state, last action, last reward, new state)
        # Calculate the loss function by first, for each old state, use the Q-network to calculate the values Q(s_old,a), i.e. the estimate of the future reward for all actions a
        # Then repeat for the target network to calculate the value \hat Q(s_new,a) of the new state (use \hat Q=0 if the new state is terminal)
        # This function should not return a value, the Q table is stored as an attribute of self

        # Useful variables: 
        # The input argument 'batch' contains a sample of quadruplets used to update the Q-network

    def fn_turn(self):
        if self.gameboard.gameover:
            self.episode+=1
            if self.episode%100==0:
                print('episode '+str(self.episode)+'/'+str(self.episode_count)+' (reward: ',str(np.mean(self.reward_tots[self.episode-100:self.episode])),')')
            if self.episode%1000==0:
                saveEpisodes=[1000,2000,5000,10000,20000,50000,100000,200000,500000,1000000];
                if self.episode in saveEpisodes:
                    # TO BE COMPLETED BY STUDENT
                    # Here you can save the rewards and the Q-network to data files
                    torch.save(self.model.state_dict(), 'qn_'+str(self.episode)+'.pth')
                    torch.save(self.model_target.state_dict(), 'qnhat_'+str(self.episode)+'.pth')
                    pickle.dump(self.reward_tots, open('reward_tots_'+str(self.episode)+'.p', 'wb'))
            if self.episode>=self.episode_count:
                raise SystemExit(0)
            else:
                if (len(self.exp_buffer) >= self.replay_buffer_size) and ((self.episode % self.sync_target_episode_count)==0):
                    pass
                    # TO BE COMPLETED BY STUDENT
                    # Here you should write line(s) to copy the current network to the target network
                    self.model_target = copy.deepcopy(self.model)
                self.gameboard.fn_restart()
        else:
            # Select and execute action (move the tile to the desired column and orientation)
            self.fn_select_action()
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to copy the old state into the variable 'old_state' which is later stored in the ecperience replay buffer
            old_state = self.state.copy()

            # Drop the tile on the game board
            reward=self.gameboard.fn_drop()

            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to add the current reward to the total reward for the current episode, so you can save it to disk later
            self.reward_tots[self.episode] += reward

            # Read the new state
            self.fn_read_state()
            
            # TO BE COMPLETED BY STUDENT
            # Here you should write line(s) to store the state in the experience replay buffer
            self.exp_buffer.append((old_state, self.action, reward, self.state.copy(), self.gameboard.gameover)) # Transition = {s_t, a_t, r_t, s_t+1}

            if len(self.exp_buffer) >= self.replay_buffer_size:
                # TO BE COMPLETED BY STUDENT
                # Here you should write line(s) to create a variable 'batch' containing 'self.batch_size' quadruplets 
                batch = random.sample(self.exp_buffer, k=self.batch_size)
                self.fn_reinforce(batch)
                if len(self.exp_buffer) >= self.replay_buffer_size + 2:
                    self.exp_buffer.pop(0) # Remove the oldest transition from the buffer


class THumanAgent:
    def fn_init(self,gameboard):
        self.episode=0
        self.reward_tots=[0]
        self.gameboard=gameboard

    def fn_read_state(self):
        pass

    def fn_turn(self,pygame):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit(0)
            if event.type==pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.reward_tots=[0]
                    self.gameboard.fn_restart()
                if not self.gameboard.gameover:
                    if event.key == pygame.K_UP:
                        self.gameboard.fn_move(self.gameboard.tile_x,(self.gameboard.tile_orientation+1)%len(self.gameboard.tiles[self.gameboard.cur_tile_type]))
                    if event.key == pygame.K_LEFT:
                        self.gameboard.fn_move(self.gameboard.tile_x-1,self.gameboard.tile_orientation)
                    if event.key == pygame.K_RIGHT:
                        self.gameboard.fn_move(self.gameboard.tile_x+1,self.gameboard.tile_orientation)
                    if (event.key == pygame.K_DOWN) or (event.key == pygame.K_SPACE):
                        self.reward_tots[self.episode]+=self.gameboard.fn_drop()