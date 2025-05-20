# 1. Imports
from cadCAD.configuration import Experiment
from cadCAD.configuration.utils import config_sim
from cadCAD.engine import ExecutionContext, Executor
from cadCAD import configs
import pandas as pd

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
    'T': range(10),  # Timesteps
    'M': {}  # Parameters
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
df = pd.DataFrame(raw_result)
# Print the table
print("\nBattle Log:")
print(df[['timestep', 'player_hp', 'monster_hp']])

# Plotting
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plt.plot(df['timestep'], df['player_hp'], 
         marker='o', markersize=12, markeredgewidth=2, 
         linewidth=3, label='Player HP', color='blue')
plt.plot(df['timestep'], df['monster_hp'], 
         marker='s', markersize=12, markeredgewidth=2, 
         linewidth=3, label='Monster HP', color='red')

plt.title("Battle Progression Over Time", fontsize=16)
plt.xlabel("Turn Number", fontsize=14)
plt.ylabel("Hit Points", fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(df['timestep'].unique())
plt.yticks(range(0, max(df['player_hp'].max(), df['monster_hp'].max()) + 20, 10))
plt.legend(fontsize=12)
plt.tight_layout()

# Add annotations for the final HP values
last_row = df.iloc[-1]
plt.annotate(f"Final: {last_row['player_hp']} HP", 
             xy=(last_row['timestep'], last_row['player_hp']),
             xytext=(5, 10), textcoords='offset points',
             ha='left', va='bottom', fontsize=12)
plt.annotate(f"Final: {last_row['monster_hp']} HP", 
             xy=(last_row['timestep'], last_row['monster_hp']),
             xytext=(5, -25), textcoords='offset points',
             ha='left', va='top', fontsize=12)

# Highlight when one reaches zero
if (df['player_hp'] == 0).any() or (df['monster_hp'] == 0).any():
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.annotate("Defeated!", 
                 xy=(df[df['monster_hp'] == 0]['timestep'].min(), 0),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', va='bottom', fontsize=12, color='red')

plt.show()