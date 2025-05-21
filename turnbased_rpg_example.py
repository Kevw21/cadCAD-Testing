# 1. Imports
import random
from cadCAD.configuration import Experiment
from cadCAD.engine import ExecutionContext, Executor
import pandas as pd
import matplotlib.pyplot as plt

# 2. Initial State
initial_state = {
    'player_hp': 250,
    'monster_hp': 200,
    'next_attacker': 'player',
    'battle_active': True,
    'last_crit': {'player': False, 'monster': False},  # Track crits
    'last_evade': {'player': False, 'monster': False},  # Track evasions
    'player_evasion': 0.25,
    'monster_evasion': 0.10
}

# 3. Policy
def turn_based_policy(params, step, sL, s):
    if not s['battle_active']:
        return {'attacker': 'none', 'damage': 0, 'crit': False, 'evaded': False}
    
    if s['next_attacker'] == 'player':
        # Set crit chance for player and look for monster evasion
        evaded = random.random() < s['monster_evasion']
        is_crit = random.random() < 0.35 if not evaded else False
        base_dmg = 15
        dmg = base_dmg * 2 if is_crit and not evaded else (0 if evaded else base_dmg)
        return {
            'attacker': 'player', 
            'damage': dmg,
            'crit': is_crit,
            'evaded': evaded,
            'base_damage': base_dmg
        }
    else:
        # Set crit chance for monster and look for player evasion
        evaded = random.random() < s['player_evasion']
        is_crit = random.random() < 0.5 if not evaded else False
        base_dmg = 15
        dmg = base_dmg * 2 if is_crit and not evaded else (0 if evaded else base_dmg)
        return {
            'attacker': 'monster',
            'damage': dmg,
            'crit': is_crit,
            'evaded': evaded,
            'base_damage': base_dmg
        }

# 4. State Update Functions
def update_player_hp(params, step, sL, s, inputs):
    if not s['battle_active']:
        return ('player_hp', s['player_hp'])
    if inputs['attacker'] == 'monster' and s['next_attacker'] == 'monster' and not inputs['evaded']:
        new_hp = max(0, s['player_hp'] - inputs['damage'])
        return ('player_hp', new_hp)
    return ('player_hp', s['player_hp'])

def update_monster_hp(params, step, sL, s, inputs):
    if not s['battle_active']:
        return ('monster_hp', s['monster_hp'])
    if inputs['attacker'] == 'player' and s['next_attacker'] == 'player' and not inputs['evaded']:
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
    battle_over = (s['player_hp'] <= 0) or (s['monster_hp'] <= 0)
    return ('battle_active', not battle_over)

def update_crit_status(params, step, sL, s, inputs):
    if inputs['attacker'] == 'player':
        return ('last_crit', {'player': inputs['crit'], 'monster': s['last_crit']['monster']})
    elif inputs['attacker'] == 'monster':
        return ('last_crit', {'player': s['last_crit']['player'], 'monster': inputs['crit']})
    return ('last_crit', s['last_crit'])

def update_evasion_status(params, step, sL, s, inputs):
    if inputs['attacker'] == 'player':
        return ('last_evade', {'player': False, 'monster': inputs['evaded']})
    elif inputs['attacker'] == 'monster':
        return ('last_evade', {'player': inputs['evaded'], 'monster': False})
    return ('last_evade', s['last_evade'])

# 5. PSUBs
psubs = [{
    'policies': {'combat': turn_based_policy},
    'variables': {
        'player_hp': update_player_hp,
        'monster_hp': update_monster_hp,
        'next_attacker': update_attacker,
        'battle_active': update_battle_state,
        'last_crit': update_crit_status,
        'last_evade': update_evasion_status
    }
}]

# 6. Simulation Config
sim_config = {
    'N': 1,
    'T': range(30),
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
active_battle_df = df[df['battle_active'] == True]

print("\n⚔️  Battle Log:")
print(active_battle_df[['timestep', 'player_hp', 'monster_hp']].to_string(index=False))

# Battle Sequence
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
        crit_msg = " (CRIT!)" if curr['last_crit']['player'] else ""
        print(f"Turn {turn_counter}: Player attacks → Monster -{dmg} HP{crit_msg} (Remaining: {curr['monster_hp']})")
        turn_counter += 1
    elif curr['last_evade']['monster']:
        print(f"Turn {turn_counter}: Player attacks → MONSTER EVADES!")
        turn_counter += 1
    elif curr['player_hp'] < prev['player_hp']:
        dmg = prev['player_hp'] - curr['player_hp']
        crit_msg = " (CRIT!)" if curr['last_crit']['monster'] else ""
        print(f"Turn {turn_counter}: Monster attacks → Player -{dmg} HP{crit_msg} (Remaining: {curr['player_hp']})")
        turn_counter += 1
    elif curr['last_evade']['player']:
        print(f"Turn {turn_counter}: Monster attacks → PLAYER EVADES!")
        turn_counter += 1

    # Check for victory
    if curr['monster_hp'] <= 0:
        print(f"  Monster defeated at turn {turn_counter-1}!")
    elif curr['player_hp'] <= 0:
        print(f"  Player defeated at turn {turn_counter-1}!")

# 9. Generate Plotting Data of combat sequence
plt.figure(figsize=(10, 6))

# Plot HP lines
player_line, = plt.plot(active_battle_df['timestep'], active_battle_df['player_hp'], 
                       color='#3498db', linewidth=2, label='Player HP')
monster_line, = plt.plot(active_battle_df['timestep'], active_battle_df['monster_hp'], 
                        color='#e74c3c', linewidth=2, label='Monster HP')

# Plot markers for combat events
for i in range(1, len(df)):
    if not df['battle_active'][i]:
        continue
        
    # Player attacks (monster HP changes)
    if df['last_evade'][i]['monster']:
        # Evaded attack
        plt.scatter(df['timestep'][i], df['monster_hp'][i-1],
                   s=200, marker='x', color='#2ecc71', linewidth=3,
                   zorder=4, label='Evasion' if i == 1 else "")
        plt.annotate('EVADED!',
                    (df['timestep'][i], df['monster_hp'][i-1]),
                    textcoords="offset points", xytext=(0,20),
                    ha='center', fontsize=11, color='#2ecc71',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    elif df['monster_hp'][i] < df['monster_hp'][i-1]:
        # Successful attack
        if df['last_crit'][i]['player']:
            # Critical hit
            plt.scatter(df['timestep'][i], df['monster_hp'][i],
                       s=200, marker='*', color='gold', edgecolor='#3498db', linewidth=2,
                       zorder=5, label='Critical Hit' if i == 1 else "")
            plt.annotate(f'CRIT! -{df["monster_hp"][i-1]-df["monster_hp"][i]}',
                        (df['timestep'][i], df['monster_hp'][i]),
                        textcoords="offset points", xytext=(0,25),
                        ha='center', fontsize=11, color='gold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.7))
        else:
            # Normal hit
            plt.scatter(df['timestep'][i], df['monster_hp'][i],
                       s=100, marker='o', facecolor='white', edgecolor='#3498db', linewidth=2,
                       zorder=3)
            plt.annotate(f'-{df["monster_hp"][i-1]-df["monster_hp"][i]}',
                        (df['timestep'][i], df['monster_hp'][i]),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=10, color='#3498db')
    
    # Monster attacks (player HP changes)
    if df['last_evade'][i]['player']:
        # Evaded attack
        plt.scatter(df['timestep'][i], df['player_hp'][i-1],
                   s=200, marker='x', color='#2ecc71', linewidth=3,
                   zorder=4)
        plt.annotate('EVADED!',
                    (df['timestep'][i], df['player_hp'][i-1]),
                    textcoords="offset points", xytext=(0,20),
                    ha='center', fontsize=11, color='#2ecc71',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    elif df['player_hp'][i] < df['player_hp'][i-1]:
        # Successful attack
        if df['last_crit'][i]['monster']:
            # Critical hit
            plt.scatter(df['timestep'][i], df['player_hp'][i],
                       s=200, marker='*', color='gold', edgecolor='#e74c3c', linewidth=2,
                       zorder=5)
            plt.annotate(f'CRIT! -{df["player_hp"][i-1]-df["player_hp"][i]}',
                        (df['timestep'][i], df['player_hp'][i]),
                        textcoords="offset points", xytext=(0,25),
                        ha='center', fontsize=11, color='gold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='black', alpha=0.7))
        else:
            # Normal hit
            plt.scatter(df['timestep'][i], df['player_hp'][i],
                       s=100, marker='s', facecolor='white', edgecolor='#e74c3c', linewidth=2,
                       zorder=3)
            plt.annotate(f'-{df["player_hp"][i-1]-df["player_hp"][i]}',
                        (df['timestep'][i], df['player_hp'][i]),
                        textcoords="offset points", xytext=(0,10),
                        ha='center', fontsize=10, color='#e74c3c')

# Style the plot
plt.title('Turn-Based Combat Simulation', fontsize=14, pad=20)
plt.xlabel('Combat Timesteps', fontsize=12)
plt.ylabel('Hit Points', fontsize=12)
plt.xticks(active_battle_df['timestep'])
plt.grid(True, linestyle='--', alpha=0.4)

# Create comprehensive legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='#3498db', lw=3, label='Player HP'),
    Line2D([0], [0], color='#e74c3c', lw=3, label='Monster HP'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='white', 
           markeredgecolor='#3498db', markersize=12, label='Player Attack'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='white', 
           markeredgecolor='#e74c3c', markersize=12, label='Monster Attack'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
           markeredgecolor='black', markersize=16, label='Critical Hit'),
    Line2D([0], [0], marker='x', color='#2ecc71', markerfacecolor='#2ecc71',
           markersize=12, label='Evaded Attack')
]
plt.legend(handles=legend_elements, loc='upper right', framealpha=1)

# Mark battle end
if len(active_battle_df) < len(df):
    end_step = active_battle_df['timestep'].max()
    plt.axvline(x=end_step, color='red', linestyle=':', alpha=0.7, linewidth=2)
    plt.text(end_step+0.2, max(active_battle_df['player_hp'].max(), 
             active_battle_df['monster_hp'].max())/2,
             'BATTLE ENDS', rotation=90, color='red', fontsize=12,
             bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))

plt.tight_layout()
plt.show()