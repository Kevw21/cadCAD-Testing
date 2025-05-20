# 1. Imports
from cadCAD.configuration import Experiment
from cadCAD.configuration.utils import config_sim
from cadCAD.engine import ExecutionContext, Executor
from cadCAD import configs

import pandas as pd
import matplotlib.pyplot as plt

# 2. Initial State
initial_state = {
    'player_hp': 100,
    'monster_hp': 50
}

# 3. Policy: Fixed attack values each turn
def attack_policy(params, step, sL, s, **kwargs):
    # Skip attacks if either is already dead
    if s['player_hp'] <= 0 or s['monster_hp'] <= 0:
        return {'player_attack': 0, 'monster_attack': 0}
    return {'player_attack': 15, 'monster_attack': 10}

# 4. State Update Functions
def update_monster_hp(params, step, sL, s, inputs, **kwargs):
    current_hp = s['monster_hp']
    # Skip update if monster is already dead
    if current_hp <= 0:
        return ('monster_hp', current_hp)
    new_hp = max(0, current_hp - inputs['player_attack'])
    return ('monster_hp', new_hp)

def update_player_hp(params, step, sL, s, inputs, **kwargs):
    current_hp = s['player_hp']
    # Skip update if player is already dead
    if current_hp <= 0:
        return ('player_hp', current_hp)
    new_hp = max(0, current_hp - inputs['monster_attack'])
    return ('player_hp', new_hp)


# 5. PSUB
psubs = [
    {
        'policies': {
            'attack_policy': attack_policy
        },
        'variables': {
            'monster_hp': update_monster_hp,
            'player_hp': update_player_hp
        }
    }
]

# 6. Simulation Configuration
sim_config = {
    'N': 1,  # Number of Monte Carlo runs
    'T': range(100),  # Timesteps
    'M': {},  # Parameters
    'stop_condition': lambda state: state['player_hp'] <= 0 or state['monster_hp'] <= 0  # Early stop
}

# 7. Setup Experiment
exp = Experiment()
exp.append_configs(
    initial_state=initial_state,
    partial_state_update_blocks=psubs,
    sim_configs=[sim_config]
)

# 8. Run Simulation
executor = Executor(exec_context=ExecutionContext(), configs=exp.configs)
raw_result, _, _ = executor.execute()

# 9. Show Results and Plot
# Print the Table
df = pd.DataFrame(raw_result)
print("\nBattle Log:")
print(df[['timestep', 'player_hp', 'monster_hp']])

# Common Plotting Settings
def configure_plot(ax, title):
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Turn Number", fontsize=14)
    ax.set_ylabel("Hit Points", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['timestep'].unique())
    ax.set_yticks(range(0, max(df['player_hp'].max(), df['monster_hp'].max()) + 20, 10))
    ax.legend(fontsize=12)

# Shortened Plot (Only meaningful turns)
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Calculate when battle actually ends
last_meaningful_timestep = df[(df['player_hp'] > 0) & (df['monster_hp'] > 0)]['timestep'].max() + 1
df_trimmed = df[df['timestep'] <= last_meaningful_timestep]

# Plot only up to battle end
ax1.plot(df_trimmed['timestep'], df_trimmed['player_hp'], 
         marker='o', markersize=10, markeredgewidth=2,
         linewidth=2, label='Player HP', color='blue')
ax1.plot(df_trimmed['timestep'], df_trimmed['monster_hp'], 
         marker='s', markersize=10, markeredgewidth=2,
         linewidth=2, label='Monster HP', color='red')

# Set x-axis limits to just cover the battle
ax1.set_xlim(-0.5, last_meaningful_timestep + 0.5)
ax1.set_xticks(range(0, last_meaningful_timestep + 1))

# Add annotations
last_row = df_trimmed.iloc[-1]
ax1.annotate(f"Final: {last_row['player_hp']} HP",
             xy=(last_row['timestep'], last_row['player_hp']),
             xytext=(5, 10), textcoords='offset points',
             ha='left', va='bottom', fontsize=10)
ax1.annotate(f"Final: {last_row['monster_hp']} HP",
             xy=(last_row['timestep'], last_row['monster_hp']),
             xytext=(5, -25), textcoords='offset points',
             ha='left', va='top', fontsize=10)

if (df_trimmed['monster_hp'] == 0).any() or (df_trimmed['player_hp'] == 0).any():
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    defeat_turn = df_trimmed[df_trimmed['monster_hp'] == 0]['timestep'].min()
    ax1.annotate("Defeated!",
                xy=(defeat_turn, 0),
                xytext=(0, 15), textcoords='offset points',
                ha='center', va='bottom', fontsize=11, color='red',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

# Configure remaining plot elements
ax1.set_title("Battle Progression (Until Battle Ends)", fontsize=14)
ax1.set_xlabel("Turn Number", fontsize=12)
ax1.set_ylabel("Hit Points", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
plt.tight_layout()

# Full Plot (All timesteps)
fig2, ax2 = plt.subplots(figsize=(12, 8))
ax2.plot(df['timestep'], df['player_hp'], 
         marker='o', markersize=12, markeredgewidth=2,
         linewidth=3, label='Player HP', color='blue')
ax2.plot(df['timestep'], df['monster_hp'], 
         marker='s', markersize=12, markeredgewidth=2,
         linewidth=3, label='Monster HP', color='red')

# Reuse the same annotations logic
last_row = df.iloc[-1]
ax2.annotate(f"Final: {last_row['player_hp']} HP",
             xy=(last_row['timestep'], last_row['player_hp']),
             xytext=(5, 10), textcoords='offset points',
             ha='left', va='bottom', fontsize=12)
ax2.annotate(f"Final: {last_row['monster_hp']} HP",
             xy=(last_row['timestep'], last_row['monster_hp']),
             xytext=(5, -25), textcoords='offset points',
             ha='left', va='top', fontsize=12)

if (df['monster_hp'] == 0).any() or (df['player_hp'] == 0).any():
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.annotate("Defeated!",
                xy=(df[df['monster_hp'] == 0]['timestep'].min(), 0),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom', fontsize=12, color='red')

configure_plot(ax2, "Battle Progression (Full Simulation)")
plt.tight_layout()
plt.show()