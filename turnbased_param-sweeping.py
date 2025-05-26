# 1. Imports
import random
import numpy as np
from cadCAD.configuration import Experiment
from cadCAD.engine import ExecutionContext, Executor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from itertools import product

# =======================================
# 2. Define Core Simulation Components
# =======================================

# Policy Function
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

# State Update Functions
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

# PSUBs (Partial State Update Blocks)
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

# Simulation Config
sim_config = {
    'N': 1,
    'T': range(30),
    'M': {},
    'stop_condition': lambda s: not s['battle_active']
}

# =============================================
# 3. Parameter Sweep Execution
# =============================================

# Define Parameter Ranges
param_sweep = {
    'player_crit_chance': [0.25, 0.5, 0.75],  # Low/Medium/High crit chance
    'monster_evasion': [0.1, 0.25, 0.4],     # Low/Medium/High evasion
}

# Generate All Combinations
param_combinations = list(product(*param_sweep.values()))
print(f"Total simulations to run: {len(param_combinations)}")

# Run Sweep
results = []

for i, (p_crit, m_evade) in enumerate(param_combinations):
    print(f"\nRunning simulation {i+1}/{len(param_combinations)}: player_crit={p_crit}, monster_evade={m_evade}")
    
    # Update Initial State
    initial_state = {
        'player_hp': 250,
        'monster_hp': 200,
        'player_base_damage': 15,
        'monster_base_damage': 15,
        'player_evasion': 0.25,
        'monster_evasion': m_evade,
        'player_crit_chance': p_crit,
        'monster_crit_chance': 0.50,
        'next_attacker': 'player',
        'battle_active': True,
        'last_crit': {'player': False, 'monster': False},
        'last_evade': {'player': False, 'monster': False}
    }

    # Monte Carlo Runs (50 per combination)
    num_simulations = 50
    battle_lengths = []
    player_wins = 0

    for _ in range(num_simulations):
        exp = Experiment()
        exp.append_configs(
            initial_state=initial_state,
            partial_state_update_blocks=psubs,
            sim_configs=[sim_config]
        )
        executor = Executor(ExecutionContext(), configs=exp.configs)
        raw_result, _, _ = executor.execute()
        df = pd.DataFrame(raw_result)
        active_df = df[df['battle_active'] == True]
        
        battle_lengths.append(len(active_df))
        if active_df['monster_hp'].iloc[-1] <= 0:
            player_wins += 1

    # Store Results
    results.append({
        'player_crit_chance': p_crit,
        'monster_evasion': m_evade,
        'player_win_rate': player_wins / num_simulations,
        'avg_battle_length': np.mean(battle_lengths)
    })

# =============================================
# 4. Analyze Results
# =============================================

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Heatmap: Player Win Rate
plt.figure(figsize=(10, 6))
pivot_win = results_df.pivot(index='player_crit_chance', columns='monster_evasion', values='player_win_rate')
sns.heatmap(pivot_win, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1)
plt.title("Player Win Rate by Parameter Combination")
plt.show()

# Heatmap: Battle Length
plt.figure(figsize=(10, 6))
pivot_length = results_df.pivot(index='player_crit_chance', columns='monster_evasion', values='avg_battle_length')
sns.heatmap(pivot_length, annot=True, fmt=".1f", cmap="YlOrRd")
plt.title("Average Battle Duration (Turns)")
plt.show()

# Tabular Results
print("\nParameter Sweep Results:")
print(results_df.sort_values('player_win_rate', ascending=False))