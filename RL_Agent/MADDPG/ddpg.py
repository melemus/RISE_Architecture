import tensorflow as tf
from keras import layers, models, optimizers
import numpy as np
from riseEnv import RiseEnv

class Actor(models.Model):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer1 = layers.Dense(64, activation='relu')
        self.layer2 = layers.Dense(64, activation='relu')
        #https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-torch-floattensor-3-1-which-is-output-0-of-tanhbackward-is-at-version-1-expected-version-0-instead/87630

        action_dim = action_dim.unsqueeze(-1)
        self.output_layer = layers.Dense(action_dim, activation='tanh')
        self.max_action = max_action

    def call(self, state):
        x = self.layer1(state)
        x = self.layer2(x)
        action = self.output_layer(x)
        return self.max_action * action

class Critic(models.Model):
    def __init__(self):
        super(Critic, self).__init__()
        self.layer1 = layers.Dense(64, activation='relu')
        self.layer2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(1)

    def call(self, state, action):
        x = tf.concat([state, action], axis=-1)
        x = self.layer1(x)
        x = self.layer2(x)
        value = self.output_layer(x)
        return value

class MultiAgentDDPG:
    def __init__(self, state_dim, action_dims, max_actions):
        self.agents = []
        for action_dim, max_action in zip(action_dims, max_actions):
            actor = Actor(action_dim, max_action)
            critic = Critic()
            actor_optimizer = optimizers.Adam(learning_rate=0.001)
            critic_optimizer = optimizers.Adam(learning_rate=0.002)
            self.agents.append({'actor': actor, 'critic': critic, 'actor_optimizer': actor_optimizer, 'critic_optimizer': critic_optimizer})

    def get_action(self, agent_id, state):
        state = np.expand_dims(state, axis=0)
        action = self.agents[agent_id]['actor'](state)
        return action.numpy()[0]

    def train(self, agent_id, state, action, reward, next_state, done):
        agent = self.agents[agent_id]

        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            predicted_action = agent['actor'](state)
            actor_loss = -agent['critic'](state, predicted_action)

            target_action = agent['actor'](next_state)
            target_value = reward + (1.0 - done) * 0.99 * agent['critic'](next_state, target_action)
            predicted_value = agent['critic'](state, action)
            critic_loss = tf.losses.mean_squared_error(target_value, predicted_value)

        actor_gradients = actor_tape.gradient(actor_loss, agent['actor'].trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, agent['critic'].trainable_variables)
        agent['actor_optimizer'].apply_gradients(zip(actor_gradients, agent['actor'].trainable_variables))
        agent['critic_optimizer'].apply_gradients(zip(critic_gradients, agent['critic'].trainable_variables))


state_dim = 5
action_dims = [1, 1, 1]
max_action_values = [1.0, 2.0, 0.5]  

multi_agent_ddpg = MultiAgentDDPG(state_dim, action_dims, max_action_values)
env = RiseEnv()

# Training loop
for episode in range(100):
    states = env.reset()
    total_reward = [0] * len(multi_agent_ddpg.agents)

    for _ in range(100):
        actions = [multi_agent_ddpg.get_action(agent_id, state) for agent_id, state in enumerate(states)]
        next_states, rewards, observations, dones = env.step(actions)

        for agent_id in range(len(multi_agent_ddpg.agents)):
            multi_agent_ddpg.train(agent_id, states[agent_id], actions[agent_id], rewards[agent_id], next_states[agent_id], dones[agent_id])
            total_reward[agent_id] += sum(rewards[agent_id])

        states = next_states

        if all(dones):
            break

    print(f"Episode: {episode + 1}, Total Rewards: {total_reward}")
