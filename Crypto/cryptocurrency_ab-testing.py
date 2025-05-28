import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD.configuration import Experiment
import multiprocessing

# Policy Functions
def p_random_events(params, substep, state_history, previous_state):
    event_strength = np.random.normal(0, 0.002)
    if np.random.random() < 0.1:
        event_strength = np.random.normal(0, 0.01)
    return {'random_event': event_strength}

def p_update_momentum(params, substep, state_history, previous_state):
    history = previous_state['price_history']
    if len(history) >= 3:
        change = (history[-1] - history[-3]) / history[-3]
    else:
        change = 0
    # Modified in Scenario B through params
    new_momentum = params['momentum_decay'] * previous_state['momentum'] + (1 - params['momentum_decay']) * change
    return {'momentum_change': new_momentum}

def p_update_sentiment(params, substep, state_history, previous_state):
    price_history = previous_state['price_history']
    if len(price_history) >= 2:
        recent_change = (price_history[-1] - price_history[-2]) / price_history[-2]
        # Modified in Scenario B through params
        new_sentiment = previous_state['market_sentiment'] + params['sentiment_sensitivity'] * recent_change
        new_sentiment = max(0, min(1, new_sentiment))
    else:
        new_sentiment = previous_state['market_sentiment']
    return {'sentiment_change': new_sentiment}

# State Update Functions
def s_price(params, substep, state_history, previous_state, policy_input):
    current_price = previous_state['price']
    base_change = np.random.normal(0.0002, 0.001)
    sentiment_factor = 0.5 + previous_state['market_sentiment']
    momentum_effect = previous_state['momentum'] * params['momentum_impact'] * sentiment_factor
    event_effect = policy_input['random_event'] * (1.5 - previous_state['market_sentiment'])
    total_change = base_change + momentum_effect + event_effect
    total_change = np.clip(total_change, -0.015, 0.015)
    new_price = current_price * (1 + total_change)
    new_price = max(0.01, new_price)
    return ('price', new_price)

def s_update_history(params, substep, state_history, previous_state, policy_input):
    history = previous_state['price_history'].copy()
    history.append(previous_state['price'])
    return ('price_history', history)

def s_momentum(params, substep, state_history, previous_state, policy_input):
    return ('momentum', policy_input['momentum_change'])

def s_sentiment(params, substep, state_history, previous_state, policy_input):
    return ('market_sentiment', policy_input['sentiment_change'])

def run_scenario(scenario_name, params, runs=30):
    all_runs = []

    for run_id in range(runs):
        initial_state = {
            'price': 109670.53,
            'price_history': [109660.59, 109663.97, 109623.48],
            'momentum': 0,
            'market_sentiment': 0.5
        }

        experiment = Experiment()

        sys_config = {
            'initial_state': initial_state,
            'partial_state_update_blocks': [
                {
                    'policies': {
                        'random_events': p_random_events,
                        'momentum_calc': p_update_momentum,
                        'sentiment_update': p_update_sentiment
                    },
                    'variables': {
                        'price': s_price,
                        'price_history': s_update_history,
                        'momentum': s_momentum,
                        'market_sentiment': s_sentiment
                    }
                }
            ]
        }

        sim_config = {
            'N': 1,
            'T': range(24),
            'M': params
        }

        experiment.append_configs(
            sim_configs=[sim_config],
            initial_state=sys_config['initial_state'],
            partial_state_update_blocks=sys_config['partial_state_update_blocks']
        )

        exec_mode = ExecutionMode()
        exec_ctx = ExecutionContext(exec_mode.local_mode)
        executor = Executor(exec_ctx, experiment.configs)
        raw_result, tensor_field, sessions = executor.execute()

        df = pd.DataFrame(raw_result)
        df['run'] = run_id
        df['scenario'] = scenario_name
        all_runs.append(df)

    return pd.concat(all_runs)

def run_ab_test():
    # Scenario A
    params_a = {
        'momentum_decay': 0.7,
        'momentum_impact': 0.05,
        'sentiment_sensitivity': 0.5
    }
    
    # Scenario B
    params_b = {
        'momentum_decay': 0.3,
        'momentum_impact': 0.15,
        'sentiment_sensitivity': 0.1
    }
    
    runs = 30  # Number of Monte Carlo simulations per scenario
    
    print(f"Running A/B test with {runs} simulations per scenario...")
    df_a = run_scenario("Scenario A", params_a, runs)
    df_b = run_scenario("Scenario B", params_b, runs)
    
    combined_df = pd.concat([df_a, df_b])
    
    # Calculate average prices across runs for each scenario
    avg_prices = combined_df.groupby(['scenario', 'timestep'])['price'].mean().reset_index(name='avg_price')
    
    # Plotting both scenarios
    plt.figure(figsize=(14, 7))
    
    # Plot individual runs (light colors)
    for scenario, color in [('Scenario A', 'lightblue'), ('Scenario B', 'lightcoral')]:
        scenario_df = combined_df[combined_df['scenario'] == scenario]
        for run_id in scenario_df['run'].unique():
            subset = scenario_df[scenario_df['run'] == run_id]
            plt.plot(subset['timestep'], subset['price'], 
                    color=color, alpha=0.2, linewidth=0.8)
    
    # Plot average lines (bold colors)
    for scenario, color in [('Scenario A', 'blue'), ('Scenario B', 'red')]:
        subset = avg_prices[avg_prices['scenario'] == scenario]
        plt.plot(subset['timestep'], subset['avg_price'], 
                color=color, label=scenario, linewidth=2.5)
    
    plt.title(f'A/B Test: {runs}-Run Monte Carlo Simulation Comparison')
    plt.xlabel('Hourly Timesteps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Comparative statistics
    print("\nComparative Statistics:")
    
    # Final price comparison
    final_prices = combined_df[combined_df['timestep'] == combined_df['timestep'].max()]
    avg_final = final_prices.groupby('scenario')['price'].mean()
    std_final = final_prices.groupby('scenario')['price'].std()
    
    print("\nFinal Price Comparison:")
    print(f"Scenario A: ${avg_final['Scenario A']:,.2f} (±{std_final['Scenario A']:,.2f})")
    print(f"Scenario B: ${avg_final['Scenario B']:,.2f} (±{std_final['Scenario B']:,.2f})")
    print(f"Difference: ${(avg_final['Scenario B'] - avg_final['Scenario A']):,.2f} "
          f"({(avg_final['Scenario B']/avg_final['Scenario A']-1)*100:.2f}%)")

    # Volatility comparison (standard deviation of price changes)
    def calc_volatility(df):
        changes = df['price'].pct_change().dropna()
        return changes.std()
    
    volatility = combined_df.groupby(['scenario', 'run']).apply(calc_volatility).groupby('scenario').mean()
    print("\nAverage Volatility (Std Dev of Hourly Changes):")
    print(f"Scenario A: {volatility['Scenario A']:.4f}")
    print(f"Scenario B: {volatility['Scenario B']:.4f}")
    print(f"Difference: {volatility['Scenario B'] - volatility['Scenario A']:.4f}")

    # Hourly fluctuation comparison
    print("\nAverage Hourly Price Changes (%):")
    for scenario in ['Scenario A', 'Scenario B']:
        scenario_avg = avg_prices[avg_prices['scenario'] == scenario]
        prices = scenario_avg['avg_price'].values
        changes = [(prices[i] - prices[i-1])/prices[i-1]*100 for i in range(1, len(prices))]
        avg_change = np.mean(changes)
        print(f"{scenario}: {avg_change:.4f}%")
    
    return combined_df

if __name__ == '__main__':
    multiprocessing.freeze_support()
    results_df = run_ab_test()