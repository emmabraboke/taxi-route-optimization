# Taxi Route Optimization with Reinforcement Learning

This project implements a simple Q-learning agent to solve the Taxi-v3 environment from OpenAI Gymnasium. The goal is to optimize the taxi’s route and decision-making to efficiently pick up and drop off passengers.

---

## 📌 Project Overview

- Implements **Q-learning** algorithm to learn an optimal policy.
- Trains the agent over 2000 episodes to maximize cumulative rewards.
- Evaluates the trained agent and optionally generates GIFs of the simulation.

---

## 🛠️ Requirements

- Python 3.x
- numpy
- gymnasium
- imageio
- pygame

---

## 🚀 Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/emmabraboke/taxi-route-optimization.git
    cd taxi-route-optimization
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ Running the Project
Train and evaluate the agent using:

```bash
python taxi_q_learning.py --alpha 0.1 --gamma 0.9 --epsilon 0.9 --episodes 2000 --max-actions 100 --render
```

**Arguments:**
- `--alpha`: Learning rate (default: 0.1)
- `--gamma`: Discount factor (default: 0.9)
- `--epsilon`: Exploration rate (default: 0.9)
- `--episodes`: Number of training episodes (default: 2000)
- `--max-actions`: Maximum actions per episode (default: 100)
- `--render`: Render the agent’s behavior and save it as a GIF (`taxi_agent_behavior.gif`)

Use the `--help` flag to see all configurable arguments:

```bash
python taxi_q_learning.py --help
```

---

**Alternatively, you can also run the project interactively:**

- Open the file `taxi_q_learning.ipynb` in Jupyter Notebook.
- Or upload `taxi_q_learning.ipynb` to [Google Colab](https://colab.research.google.com/) and run it in your browser.

---

## 📈 Training Progress

During training, the script prints progress and hyperparameters.

---

## 🎥 Agent Behavior

After training, if `--render` is specified, the agent’s behavior is saved as a GIF:

> Example of the trained agent navigating the Taxi-v3 environment.

![Trained Taxi-v3 Agent](taxi_agent_behavior.gif)

---