import numpy as np
import gymnasium as gym
import imageio
from IPython.display import Image, display
from gymnasium.utils import seeding
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Taxi-v3 Q-Learning Agent")
    parser.add_argument("--alpha", type=float, default=0.1, help="Learning rate (default: 0.1)")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor (default: 0.9)")
    parser.add_argument("--epsilon", type=float, default=0.9, help="Exploration rate (default: 0.9)")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes (default: 2000)")
    parser.add_argument("--max-actions", type=int, default=100, help="Max actions per episode (default: 100)")
    parser.add_argument("--render", action="store_true", help="Render agent behavior as a GIF")
    return parser.parse_args()

def update_q_table(q_table, state, action, reward, new_state, alpha, gamma):
    old_value = q_table[state, action]
    next_max = np.max(q_table[new_state])
    q_table[state, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

def epsilon_greedy(q_table, state, epsilon, action_space):
    if np.random.rand() < epsilon:
        return action_space.sample()
    return np.argmax(q_table[state, :])

def get_policy(q_table, num_states):
    return {state: np.argmax(q_table[state]) for state in range(num_states)}

def train_q_learning(env, q_table, alpha, gamma, epsilon, num_episodes, max_actions):
    episode_returns = []
    for episode in range(num_episodes):
        state, info = env.reset(seed=42)
        terminated = False
        episode_reward = 0
        episode_action = 0

        while not (terminated):
            action = epsilon_greedy(q_table, state, epsilon, env.action_space)
            new_state, reward, terminated, truncated, info = env.step(action)
            update_q_table(q_table, state, action, reward, new_state, alpha, gamma)
            episode_reward += reward
            episode_action += 1
            state = new_state
            if episode_action >= max_actions:
                break
        episode_returns.append(episode_reward)
    return episode_returns

def evaluate_policy(env, policy, max_actions, render=False):
    frames = []
    state, info = env.reset(seed=42)
    terminated = False
    episode_total_reward = 0
    episode_action = 0

    while not (terminated):
        action = policy[state]
        episode_action += 1
        new_state, reward, terminated, truncated, info = env.step(action)
        if render:
            frames.append(env.render())
        episode_total_reward += reward
        state = new_state
        if episode_action >= max_actions:
            break

    if render:
        imageio.mimsave('taxi_agent_behavior.gif', frames, fps=5, loop=0)
        gif_path = "taxi_agent_behavior.gif"
        display(Image(gif_path))

    return episode_total_reward

def main():
    args = parse_args()
    env = gym.make("Taxi-v3", render_mode='rgb_array')
    env.np_random, _ = seeding.np_random(42)
    env.action_space.seed(42)
    np.random.seed(42)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    q_table = np.zeros((num_states, num_actions))

    print(f"Training agent with alpha={args.alpha}, gamma={args.gamma}, epsilon={args.epsilon}")
    train_q_learning(env, q_table, args.alpha, args.gamma, args.epsilon, args.episodes, args.max_actions)
    policy = get_policy(q_table, num_states)
    total_reward = evaluate_policy(env, policy, args.max_actions, render=args.render)
    print(f"Total reward during evaluation: {total_reward}")

if __name__ == "__main__":
    main()