import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_NEUTRAL = 3  # beacon/minerals
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

class DQNAgent:
	def __init__(self, state_size,action_size):
		self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
		model = Sequential()

		model.add(Dense(24, input_dim=self.state_size,activation='relu'))

		model.add(Dense(24, activation='relu'))

		model.add(Dense(self.action_size, activation='linear'))

		model.compile(loss='mse',optimizer= Adam(lr=self.learning_rate))
		return model

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	def act(self, state):
		if np.random.rand() <self. epsilon:
			return env,.action_space.sample()
		act_values = self.model.predict(state)
		return np.argmax(act_values[0])

	def replay(self, batch_size):
		minibatch = random.sample(self.memory, batch_size)

		for state,action, reward, next_state, done in minibatch:
			target = reward

			if not done:
				target = reward+ self.gamma * np.amax(self.mode.predict(next_state)[0])

			target_f = self.mode.predict(state)
			target_f[0][action] = target
			self.mode.fit(state, target_f, epochs =1, verbose = 0)
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

if __name__ == "__main__":

	env = sc2_env.SC2Env(
        map_name="CollectMineralShards",
        step_mul=step_mul,
        visualize=True,
        screen_size_px=(16, 16),
        minimap_size_px=(16, 16))
	batch_size = 1
	action_size = len(sc2_actions)
	agent = DQNAgent(batch_size, action_size)

	for e in range(episodes):
		episode_rewards = [0.0]
		env.reset()
		state = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
		player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
  		screen = (player_relative == _PLAYER_NEUTRAL).astype(int) 
		state = np.reshape(screen,[1,4])

		for time_t in range(1000):
			env.render()
			action = agent.act(state)

			if _MOVE_SCREEN not in obs[0].observation["available_actions"]:
        		obs = env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])

      		new_action = [sc2_actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, coord])]
		    # else:
		    #   new_action = [sc2_actions.FunctionCall(_NO_OP, [])]
      		obs = env.step(actions=new_action)
      		player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
      		new_screen = (player_relative == _PLAYER_NEUTRAL).astype(int)
			next_state =new_screen
			next_state = np.reshape(next_state, [1,4])

			rew = obs[0].reward,
			done  =  obs[0].step_type == environment.StepType.LAST
			
			episode_rewards[-1] += rew
			reward = episode_rewards[-1]

			agent.remember(state,action, reward, next_state, done)

			state = next_state

			if done:
				obs = env.reset()
		        player_relative = obs[0].observation["screen"][_PLAYER_RELATIVE]
		        screent = (player_relative == _PLAYER_NEUTRAL).astype(int)

		        env.step(actions=[sc2_actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])])
        		episode_rewards.append(0.0)

