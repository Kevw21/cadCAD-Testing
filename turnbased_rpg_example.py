# 1. Imports
import random
import numpy as np
from cadCAD.configuration import Experiment
from cadCAD.engine import ExecutionContext, Executor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 2. Initial State
initial_state = {
    'player_hp': 250,
    'monster_hp': 200,
    'player_base_damage': 15,
    'monster_base_damage': 15,
    'player_evasion': 0.25,
    'monster_evasion': 0.25,
    'player_crit_chance': 0.35,
    'monster_crit_chance': 0.50,
    'next_attacker': 'player',
    'battle_active': True,
    'last_crit': {'player': False, 'monster': False},
    'last_evade': {'player': False, 'monster': False}
}

# 3. Policy
def turn_based_policy(params, step, sL, s):
    if not s['battle_active']:
        return {'attacker': 'none', 'damage': 0, 'crit': False, 'evaded': False}
    
    if s['next_attacker'] == 'player':
        evaded = random.random() < s['monster_evasion']
        is_crit = random.random() < s['player_crit_chance'] if not evaded else False
        base_dmg = s['player_base_damage']
        dmg = base_dmg * 2 if is_crit and not evaded else (0 if evaded else base_dmg)
        return {
            'attacker': 'player', 
            'damage': dmg,
            'crit': is_crit,
            'evaded': evaded,
            'base_damage': base_dmg
        }
    else:
        evaded = random.random() < s['player_evasion']
        is_crit = random.random() < s['monster_crit_chance'] if not evaded else False
        base_dmg = s['monster_base_damage']
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

# 7. Monte Carlo Simulation Runs
num_simulations = 50
all_simulations = []

print(f"\nRunning {num_simulations} combat scenarios...")
for run in range(num_simulations):
    # Run simulation
    exp = Experiment()
    exp.append_configs(
        initial_state=initial_state,
        partial_state_update_blocks=psubs,
        sim_configs=[sim_config]
    )
    executor = Executor(ExecutionContext(), configs=exp.configs)
    raw_result, _, _ = executor.execute()
    df = pd.DataFrame(raw_result)
    all_simulations.append(df)

# 8. Process Results and Plot
plt.figure(figsize=(14, 8))

# Colors and styling
player_color = '#3498db'
monster_color = '#e74c3c'
crit_color = 'gold'
evade_color = '#2ecc71'
alpha = 0.15

# Plot all simulation runs
for df in all_simulations:
    active_df = df[df['battle_active'] == True]
    
    # Plot HP lines
    plt.plot(active_df['timestep'], active_df['player_hp'], 
             color=player_color, linewidth=1, alpha=alpha)
    plt.plot(active_df['timestep'], active_df['monster_hp'], 
             color=monster_color, linewidth=1, alpha=alpha)
    
    # Mark special events
    for i in range(1, len(active_df)):
        # Player attacks
        if active_df['last_evade'].iloc[i]['monster']:
            plt.scatter(active_df['timestep'].iloc[i], active_df['monster_hp'].iloc[i-1],
                       s=40, marker='x', color=evade_color, alpha=0.7)
        elif active_df['monster_hp'].iloc[i] < active_df['monster_hp'].iloc[i-1]:
            if active_df['last_crit'].iloc[i]['player']:
                plt.scatter(active_df['timestep'].iloc[i], active_df['monster_hp'].iloc[i],
                           s=60, marker='*', color=crit_color, alpha=0.7)
        
        # Monster attacks
        if active_df['last_evade'].iloc[i]['player']:
            plt.scatter(active_df['timestep'].iloc[i], active_df['player_hp'].iloc[i-1],
                       s=40, marker='x', color=evade_color, alpha=0.7)
        elif active_df['player_hp'].iloc[i] < active_df['player_hp'].iloc[i-1]:
            if active_df['last_crit'].iloc[i]['monster']:
                plt.scatter(active_df['timestep'].iloc[i], active_df['player_hp'].iloc[i],
                           s=60, marker='*', color=crit_color, alpha=0.7)

# Calculate and plot average HP trajectories
max_length = max(len(sim[sim['battle_active'] == True]) for sim in all_simulations)
avg_player = np.zeros(max_length)
avg_monster = np.zeros(max_length)
counts = np.zeros(max_length)

for sim in all_simulations:
    active_sim = sim[sim['battle_active'] == True]
    for t in range(len(active_sim)):
        avg_player[t] += active_sim['player_hp'].iloc[t]
        avg_monster[t] += active_sim['monster_hp'].iloc[t]
        counts[t] += 1

avg_player = avg_player / counts
avg_monster = avg_monster / counts

plt.plot(range(max_length), avg_player, color='darkblue', 
         linewidth=3, linestyle='--', label='Avg Player HP')
plt.plot(range(max_length), avg_monster, color='darkred', 
         linewidth=3, linestyle='--', label='Avg Monster HP')

# Calculate and plot win probabilities
player_wins = np.zeros(max_length)
monster_wins = np.zeros(max_length)

for sim in all_simulations:
    active_sim = sim[sim['battle_active'] == True]
    outcome = 'player' if active_sim['monster_hp'].iloc[-1] <= 0 else 'monster'
    for t in range(len(active_sim)):
        if outcome == 'player':
            player_wins[t] += 1
        else:
            monster_wins[t] += 1

plt.fill_between(range(max_length), player_wins/num_simulations*100, 
                 color='blue', alpha=0.1, label='Player Win %')
plt.fill_between(range(max_length), 100-(monster_wins/num_simulations*100), 
                 color='red', alpha=0.1, label='Monster Win %')

# Style the plot
plt.title(
    f'Combat Simulation ({num_simulations} Runs)\n'
    f"Player: {initial_state['player_crit_chance']*100:.0f}% Crit, {initial_state['player_evasion']*100:.0f}% Evade | "
    f"Monster: {initial_state['monster_crit_chance']*100:.0f}% Crit, {initial_state['monster_evasion']*100:.0f}% Evade",
    fontsize=14, 
    pad=20
)
plt.xlabel('Combat Timesteps', fontsize=12)
plt.ylabel('Hit Points / Win Probability (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.ylim(0, max(initial_state['player_hp'], initial_state['monster_hp']) + 20)

# Create comprehensive legend
legend_elements = [
    Line2D([0], [0], color='darkblue', lw=3, linestyle='--', label='Avg Player HP'),
    Line2D([0], [0], color='darkred', lw=3, linestyle='--', label='Avg Monster HP'),
    Line2D([0], [0], color='blue', alpha=0.2, lw=10, label='Player Win %'),
    Line2D([0], [0], color='red', alpha=0.2, lw=10, label='Monster Win %'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor=crit_color, 
           markersize=15, label='Critical Hit'),
    Line2D([0], [0], marker='x', color='w', markerfacecolor=evade_color,
           markersize=12, label='Evaded Attack')
]

plt.legend(handles=legend_elements, loc='upper right', framealpha=1)
plt.tight_layout()
plt.show()

# Print summary statistics
player_victories = sum(1 for sim in all_simulations 
                      if sim[sim['battle_active'] == True]['monster_hp'].iloc[-1] <= 0)
monster_victories = num_simulations - player_victories
avg_length = np.mean([len(sim[sim['battle_active'] == True]) for sim in all_simulations])

print("\n⚔️  Battle Statistics:")
print(f"Player Victory Rate: {player_victories}/{num_simulations} ({player_victories/num_simulations*100:.1f}%)")
print(f"Monster Victory Rate: {monster_victories}/{num_simulations} ({monster_victories/num_simulations*100:.1f}%)")
print(f"Average Battle Length: {avg_length:.1f} turns")
print(f"Player Avg Final HP: {np.mean([sim[sim['battle_active'] == True]['player_hp'].iloc[-1] for sim in all_simulations]):.1f}")
print(f"Monster Avg Final HP: {np.mean([sim[sim['battle_active'] == True]['monster_hp'].iloc[-1] for sim in all_simulations]):.1f}")