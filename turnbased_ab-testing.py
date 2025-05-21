# 1. Imports
import random
import numpy as np
from cadCAD.configuration import Experiment
from cadCAD.engine import ExecutionContext, Executor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 2. Define the two scenarios for A/B testing
scenario_a = {
    'name': 'High Crit Player',
    'initial_state': {
        'player_hp': 250,
        'monster_hp': 200,
        'player_base_damage': 15,
        'monster_base_damage': 15,
        'player_evasion': 0.15,  # Lower evasion
        'monster_evasion': 0.25,
        'player_crit_chance': 0.50,  # Higher crit chance
        'monster_crit_chance': 0.50,
        'next_attacker': 'player',
        'battle_active': True,
        'last_crit': {'player': False, 'monster': False},
        'last_evade': {'player': False, 'monster': False}
    }
}

scenario_b = {
    'name': 'High Evasion Player',
    'initial_state': {
        'player_hp': 250,
        'monster_hp': 200,
        'player_base_damage': 15,
        'monster_base_damage': 15,
        'player_evasion': 0.40,  # Higher evasion
        'monster_evasion': 0.25,
        'player_crit_chance': 0.20,  # Lower crit chance
        'monster_crit_chance': 0.50,
        'next_attacker': 'player',
        'battle_active': True,
        'last_crit': {'player': False, 'monster': False},
        'last_evade': {'player': False, 'monster': False}
    }
}

# 3. Policy and State Update Functions
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


# 4. PSUBs
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

# 5. Simulation Config
sim_config = {
    'N': 1,
    'T': range(30),
    'M': {},
    'stop_condition': lambda s: not s['battle_active']
}

# 6. Run A/B Testing
num_simulations = 100
results = {'scenario_a': [], 'scenario_b': []}

def run_scenario(scenario):
    scenario_results = []
    for run in range(num_simulations):
        exp = Experiment()
        exp.append_configs(
            initial_state=scenario['initial_state'],
            partial_state_update_blocks=psubs,
            sim_configs=[sim_config]
        )
        executor = Executor(ExecutionContext(), configs=exp.configs)
        raw_result, _, _ = executor.execute()
        df = pd.DataFrame(raw_result)
        scenario_results.append(df)
    return scenario_results

print(f"\nRunning A/B testing with {num_simulations} simulations per scenario...")
results['scenario_a'] = run_scenario(scenario_a)
results['scenario_b'] = run_scenario(scenario_b)

# 7. Comparative Analysis
def analyze_scenario(scenario, simulation_results):
    victories = {
        'player': 0,
        'monster': 0
    }
    battle_lengths = []
    player_final_hps = []
    monster_final_hps = []
    crit_rates = []
    evade_rates = []
    
    for sim in simulation_results:
        active_sim = sim[sim['battle_active'] == True]
        outcome = 'player' if active_sim['monster_hp'].iloc[-1] <= 0 else 'monster'
        victories[outcome] += 1
        battle_lengths.append(len(active_sim))
        player_final_hps.append(active_sim['player_hp'].iloc[-1])
        monster_final_hps.append(active_sim['monster_hp'].iloc[-1])
        
        # Calculate crit and evade rates
        total_attacks = len(active_sim) - 1  # Subtract initial state
        player_crits = sum(1 for i in range(1, len(active_sim)) 
                        if active_sim['last_crit'].iloc[i]['player'])
        monster_crits = sum(1 for i in range(1, len(active_sim)) 
                         if active_sim['last_crit'].iloc[i]['monster'])
        player_evades = sum(1 for i in range(1, len(active_sim)) 
                         if active_sim['last_evade'].iloc[i]['player'])
        monster_evades = sum(1 for i in range(1, len(active_sim)) 
                        if active_sim['last_evade'].iloc[i]['monster'])
        
        crit_rates.append((player_crits + monster_crits) / (total_attacks * 2))
        evade_rates.append((player_evades + monster_evades) / (total_attacks * 2))
    
    return {
        'name': scenario['name'],
        'victory_rate': victories['player'] / num_simulations,
        'avg_battle_length': np.mean(battle_lengths),
        'avg_player_final_hp': np.mean(player_final_hps),
        'avg_monster_final_hp': np.mean(monster_final_hps),
        'avg_crit_rate': np.mean(crit_rates),
        'avg_evade_rate': np.mean(evade_rates),
        'player_crit_chance': scenario['initial_state']['player_crit_chance'],
        'player_evasion': scenario['initial_state']['player_evasion']
    }

analysis_a = analyze_scenario(scenario_a, results['scenario_a'])
analysis_b = analyze_scenario(scenario_b, results['scenario_b'])

# 8. Print Comparative Results
print("\n=== Comparative Analysis ===")
print(f"\nScenario A: {scenario_a['name']}")
print(f"  Player Crit Chance: {analysis_a['player_crit_chance']*100:.0f}%")
print(f"  Player Evasion: {analysis_a['player_evasion']*100:.0f}%")
print(f"  Victory Rate: {analysis_a['victory_rate']*100:.1f}%")
print(f"  Average Battle Length: {analysis_a['avg_battle_length']:.1f} turns")
print(f"  Average Player Final HP: {analysis_a['avg_player_final_hp']:.1f}")
print(f"  Average Crit Rate: {analysis_a['avg_crit_rate']*100:.1f}%")
print(f"  Average Evade Rate: {analysis_a['avg_evade_rate']*100:.1f}%")

print(f"\nScenario B: {scenario_b['name']}")
print(f"  Player Crit Chance: {analysis_b['player_crit_chance']*100:.0f}%")
print(f"  Player Evasion: {analysis_b['player_evasion']*100:.0f}%")
print(f"  Victory Rate: {analysis_b['victory_rate']*100:.1f}%")
print(f"  Average Battle Length: {analysis_b['avg_battle_length']:.1f} turns")
print(f"  Average Player Final HP: {analysis_b['avg_player_final_hp']:.1f}")
print(f"  Average Crit Rate: {analysis_b['avg_crit_rate']*100:.1f}%")
print(f"  Average Evade Rate: {analysis_b['avg_evade_rate']*100:.1f}%")

# 9. Plot Comparative Results
plt.figure(figsize=(14, 10))

# Victory Rate Comparison
plt.subplot(2, 2, 1)
plt.bar(['High Crit', 'High Evade'], 
        [analysis_a['victory_rate']*100, analysis_b['victory_rate']*100],
        color=['#3498db', '#2ecc71'])
plt.title('Player Victory Rate Comparison')
plt.ylabel('Victory Rate (%)')
plt.ylim(0, 100)

# Battle Length Comparison
plt.subplot(2, 2, 2)
plt.bar(['High Crit', 'High Evade'], 
        [analysis_a['avg_battle_length'], analysis_b['avg_battle_length']],
        color=['#3498db', '#2ecc71'])
plt.title('Average Battle Length Comparison')
plt.ylabel('Turns')

# Final HP Comparison
plt.subplot(2, 2, 3)
bar_width = 0.35
index = np.arange(2)
plt.bar(index, [analysis_a['avg_player_final_hp'], analysis_b['avg_player_final_hp']],
        bar_width, label='Player', color=['#3498db', '#2ecc71'])
plt.bar(index + bar_width, [analysis_a['avg_monster_final_hp'], analysis_b['avg_monster_final_hp']],
        bar_width, label='Monster', color=['#e74c3c', '#e74c3c'])
plt.title('Average Final HP Comparison')
plt.xticks(index + bar_width/2, ['High Crit', 'High Evade'])
plt.ylabel('Hit Points')
plt.legend()

# Event Rate Comparison
plt.subplot(2, 2, 4)
plt.bar(index, [analysis_a['avg_crit_rate']*100, analysis_b['avg_crit_rate']*100],
        bar_width, label='Crit Rate', color='gold')
plt.bar(index + bar_width, [analysis_a['avg_evade_rate']*100, analysis_b['avg_evade_rate']*100],
        bar_width, label='Evade Rate', color='#9b59b6')
plt.title('Event Rates Comparison')
plt.xticks(index + bar_width/2, ['High Crit', 'High Evade'])
plt.ylabel('Rate (%)')
plt.legend()

plt.tight_layout()
plt.suptitle('A/B Testing: High Crit vs High Evasion Player Strategies', y=1.02, fontsize=14)
plt.show()

# 10. Statistical Significance Test
from scipy import stats

# Prepare data for t-tests
def get_outcomes(simulation_results):
    return [1 if sim[sim['battle_active'] == True]['monster_hp'].iloc[-1] <= 0 
            else 0 for sim in simulation_results]

a_outcomes = get_outcomes(results['scenario_a'])
b_outcomes = get_outcomes(results['scenario_b'])

# Perform chi-square test for victory rates
chi2, p_val, _, _ = stats.chi2_contingency([
    [sum(a_outcomes), num_simulations - sum(a_outcomes)],
    [sum(b_outcomes), num_simulations - sum(b_outcomes)]
])

print("\n=== Statistical Significance ===")
print(f"Chi-square test p-value: {p_val:.4f}")
if p_val < 0.05:
    print("The difference in victory rates is statistically significant (p < 0.05)")
else:
    print("The difference in victory rates is not statistically significant (p ≥ 0.05)")

# Perform t-test for battle lengths
a_lengths = [len(sim[sim['battle_active'] == True]) for sim in results['scenario_a']]
b_lengths = [len(sim[sim['battle_active'] == True]) for sim in results['scenario_b']]
t_stat, p_val = stats.ttest_ind(a_lengths, b_lengths)

print(f"\nT-test for battle lengths p-value: {p_val:.4f}")
if p_val < 0.05:
    print("The difference in battle lengths is statistically significant (p < 0.05)")
else:
    print("The difference in battle lengths is not statistically significant (p ≥ 0.05)")