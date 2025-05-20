# 1. Imports
from cadCAD.configuration import Experiment
from cadCAD.engine import ExecutionContext, Executor
import pandas as pd
import matplotlib.pyplot as plt

# 2. Initial State
initial_state = {
    'player_hp': 100,
    'monster_hp': 50,
    'next_attacker': 'player',
    'battle_active': True  # Flag to track battle state
}

# 3. Policy
def turn_based_policy(params, step, sL, s):
    if not s['battle_active']:
        return {'attacker': 'none', 'damage': 0}
    if s['next_attacker'] == 'player':
        return {'attacker': 'player', 'damage': 15}
    else:
        return {'attacker': 'monster', 'damage': 10}

# 4. State Updates
def update_player_hp(params, step, sL, s, inputs):
    if not s['battle_active']:
        return ('player_hp', s['player_hp'])
    if inputs['attacker'] == 'monster' and s['next_attacker'] == 'monster':
        new_hp = max(0, s['player_hp'] - inputs['damage'])
        return ('player_hp', new_hp)
    return ('player_hp', s['player_hp'])

def update_monster_hp(params, step, sL, s, inputs):
    if not s['battle_active']:
        return ('monster_hp', s['monster_hp'])
    if inputs['attacker'] == 'player' and s['next_attacker'] == 'player':
        new_hp = max(0, s['monster_hp'] - inputs['damage'])
        return ('monster_hp', new_hp)
    return ('monster_hp', s['monster_hp'])

def update_attacker(params, step, sL, s, inputs):
    if not s['battle_active']:
        return ('next_attacker', s['next_attacker'])
    if inputs['attacker'] == 'player':
        return ('next_attacker', 'monster')
    else:
        return ('next_attacker', 'player')

def update_battle_state(params, step, sL, s, inputs):
    # Battle ends when either reaches 0 HP
    battle_over = (s['player_hp'] <= 0) or (s['monster_hp'] <= 0)
    return ('battle_active', not battle_over)

# 5. PSUB
psubs = [{
    'policies': {'combat': turn_based_policy},
    'variables': {
        'player_hp': update_player_hp,
        'monster_hp': update_monster_hp,
        'next_attacker': update_attacker,
        'battle_active': update_battle_state
    }
}]

# 6. Simulation Config
sim_config = {
    'N': 1,
    'T': range(20),
    'M': {},
    'stop_condition': lambda s: not s['battle_active']
}

# 7. Run Simulation
exp = Experiment()
exp.append_configs(
    initial_state=initial_state,
    partial_state_update_blocks=psubs,
    sim_configs=[sim_config]
)

executor = Executor(ExecutionContext(), configs=exp.configs)
raw_result, _, _ = executor.execute()

# 8. Process Results
df = pd.DataFrame(raw_result)

# Filter to only active battle turns
active_battle_df = df[df['battle_active'] == True]

print("\n⚔️ Battle Log:")
print(active_battle_df[['timestep', 'player_hp', 'monster_hp']].to_string(index=False))

# 9. Generate Professional Combat Graph
plt.figure(figsize=(10, 6))

# Plot only active battle HP lines
plt.plot(active_battle_df['timestep'], active_battle_df['player_hp'], 
         marker='o', markersize=8, linewidth=2, 
         color='#3498db', label='Player HP')
plt.plot(active_battle_df['timestep'], active_battle_df['monster_hp'], 
         marker='s', markersize=8, linewidth=2, 
         color='#e74c3c', label='Monster HP')

# Add combat markers and damage text
for i in range(1, len(df)):
    if df['monster_hp'][i] < df['monster_hp'][i-1]:  # Player attack
        dmg = df['monster_hp'][i-1] - df['monster_hp'][i]
        plt.annotate(f'-{dmg}', 
                    (df['timestep'][i], df['monster_hp'][i]),
                    textcoords="offset points", xytext=(0,5),
                    ha='center', va='bottom', fontsize=10, color='#3498db',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))
    elif df['player_hp'][i] < df['player_hp'][i-1]:  # Monster attack
        dmg = df['player_hp'][i-1] - df['player_hp'][i]
        plt.annotate(f'-{dmg}', 
                    (df['timestep'][i], df['player_hp'][i]),
                    textcoords="offset points", xytext=(0,5),
                    ha='center', va='bottom', fontsize=10, color='#e74c3c',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.8))

# Style the plot
plt.title('Turn-Based Combat', fontsize=12)
plt.xlabel('Combat Timesteps', fontsize=10)
plt.ylabel('Hit Points', fontsize=10)
plt.xticks(active_battle_df['timestep'])
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()

# Mark battle end
if len(active_battle_df) < len(df):
    end_step = active_battle_df['timestep'].max()
    plt.axvline(x=end_step, color='red', linestyle=':', alpha=0.5)
    plt.text(end_step+0.2, max(active_battle_df['player_hp'].max(), 
             active_battle_df['monster_hp'].max())/2,
             'BATTLE ENDS', rotation=90, color='red')

plt.tight_layout()
plt.show()

# 10. Detailed Battle Sequence
print("\nBattle Sequence:")
turn_counter = 1
for i in range(1, len(df)):
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    
    # Skip if battle already ended
    if prev['monster_hp'] <= 0 or prev['player_hp'] <= 0:
        continue
        
    if curr['monster_hp'] < prev['monster_hp']:
        dmg = prev['monster_hp'] - curr['monster_hp']
        print(f"Turn {turn_counter}: Player attacks → Monster -{dmg} HP (Remaining: {curr['monster_hp']})")
        turn_counter += 1
        if curr['monster_hp'] <= 0:
            print(f"  Monster defeated at turn {turn_counter-1}!")
    elif curr['player_hp'] < prev['player_hp']:
        dmg = prev['player_hp'] - curr['player_hp']
        print(f"Turn {turn_counter}: Monster attacks → Player -{dmg} HP (Remaining: {curr['player_hp']})")
        turn_counter += 1