with open('federated_learning/config/config.py', 'r') as f:
    lines = f.readlines()

rl_config_lines = [
    "\n",
    "# ======================================\n",
    "# RL-BASED AGGREGATION CONFIGURATION\n",
    "# ======================================\n",
    "\n",
    "# Aggregation method selection\n",
    "# Options: 'dual_attention', 'rl_actor_critic', 'hybrid'\n",
    "# - 'dual_attention': Use only the dual attention mechanism\n",
    "# - 'rl_actor_critic': Use only the RL actor-critic approach\n",
    "# - 'hybrid': Start with dual attention, then gradually transition to RL\n",
    "RL_AGGREGATION_METHOD = 'dual_attention'  # Default to dual attention\n",
    "\n",
    "# RL Actor-Critic Parameters\n",
    "RL_ACTOR_HIDDEN_DIMS = [128, 64]       # Hidden layer dimensions for actor network\n",
    "RL_CRITIC_HIDDEN_DIMS = [128, 64]      # Hidden layer dimensions for critic network\n",
    "RL_LEARNING_RATE = 0.001               # Learning rate for both actor and critic\n",
    "RL_GAMMA = 0.99                        # Discount factor\n",
    "RL_ENTROPY_COEF = 0.01                 # Entropy coefficient for exploration\n",
    "RL_WARMUP_ROUNDS = 5                   # Rounds to use dual attention before RL\n",
    "RL_RAMP_UP_ROUNDS = 10                 # Rounds to blend dual attention with RL\n",
    "RL_INITIAL_TEMP = 5.0                  # Initial temperature for softmax\n",
    "RL_MIN_TEMP = 0.5                      # Minimum temperature after annealing\n",
    "RL_FALLBACK_THRESHOLD = 0.05           # Max allowed validation loss increase before fallback\n",
    "RL_PRETRAINING_EPISODES = 1000         # Number of episodes for pre-training\n",
    "RL_VALIDATION_MINIBATCH = 0.2          # Fraction of validation data to use for faster evaluation\n",
    "RL_SAVE_INTERVAL = 100                 # Interval for saving checkpoints during pre-training\n"
]

# Look for ATTACK CONFIGURATION section
for i, line in enumerate(lines):
    if "# ATTACK CONFIGURATION" in line and "======" in lines[i-1]:
        # Insert RL configuration before ATTACK CONFIGURATION
        lines[i-1:i-1] = rl_config_lines
        break

with open('federated_learning/config/config.py', 'w') as f:
    f.writelines(lines)

print("RL configuration parameters added successfully!") 