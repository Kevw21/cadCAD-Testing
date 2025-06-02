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
        event_strength = np.random.normal(0.01, 0.01)
    return {'random_event': event_strength}

def p_update_momentum(params, substep, state_history, previous_state):
    history = previous_state['price_history']
    if len(history) >= 5:
        change = (history[-1] - history[-5]) / history[-5]
    else:
        change = 0
    new_momentum = 0.7 * previous_state['momentum'] + 0.3 * change
    return {'momentum_change': new_momentum}

def p_update_sentiment(params, substep, state_history, previous_state):
    price_history = previous_state['price_history']
    if len(price_history) >= 2:
        recent_change = (price_history[-1] - price_history[-2]) / price_history[-2]
        new_sentiment = previous_state['market_sentiment'] + 0.1 * recent_change
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

# Run simulation for a single momentum_impact value
def run_sweep(momentum_impact, runs=30):
    all_runs = []

    for run_id in range(runs):
        initial_state = {
            'price': 104745.90,
            'price_history': [104706.40, 104550.60, 104230.80, 104314.40, 104499.80],
            'momentum': 0,
            'market_sentiment': 0.5
        }

        experiment = Experiment()
        sys_config = {
            'initial_state': initial_state,
            'partial_state_update_blocks': [{
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
            }]
        }

        sim_config = {
            'N': 1,
            'T': range(24),
            'M': {'momentum_impact': momentum_impact}
        }

        experiment.append_configs(
            sim_configs=[sim_config],
            initial_state=sys_config['initial_state'],
            partial_state_update_blocks=sys_config['partial_state_update_blocks']
        )

        exec_mode = ExecutionMode()
        exec_ctx = ExecutionContext(exec_mode.local_mode)
        executor = Executor(exec_ctx, experiment.configs)
        raw_result, *_ = executor.execute()

        df = pd.DataFrame(raw_result)
        df['run'] = run_id
        df['momentum_impact'] = momentum_impact
        all_runs.append(df)

    return pd.concat(all_runs)

# Full parameter sweep
def parameter_sweep():
    momentum_values = np.round(np.linspace(0.01, 0.20, 10), 3)
    all_data = []

    for val in momentum_values:
        print(f"Running simulations for momentum_impact = {val:.3f}")
        df = run_sweep(momentum_impact=val, runs=30)
        all_data.append(df)

    combined_df = pd.concat(all_data)

    # Plot average prices for each value
    plt.figure(figsize=(14, 8))
    for val in momentum_values:
        subset = combined_df[combined_df['momentum_impact'] == val]
        avg_price = subset.groupby('timestep')['price'].mean()
        plt.plot(avg_price.index, avg_price.values, label=f'{val:.3f}')

    plt.title("Average Price Over Time for Different Momentum Impact Values")
    plt.xlabel("Timestep (Hours)")
    plt.ylabel("Price (USD)")
    plt.legend(title='Momentum Impact')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Final Price Boxplot
    plt.figure(figsize=(12, 6))
    final_prices = combined_df[combined_df['timestep'] == combined_df['timestep'].max()]
    final_prices.boxplot(column='price', by='momentum_impact')
    plt.title("Final Price Distribution by Momentum Impact")
    plt.suptitle("")
    plt.xlabel("Momentum Impact")
    plt.ylabel("Final Price (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Volatility vs. Momentum Impact
    def calc_vol(df):
        return df['price'].pct_change().dropna().std()

    volatility_df = combined_df.groupby(['momentum_impact', 'run']).apply(calc_vol).reset_index(name='volatility')
    mean_vol = volatility_df.groupby('momentum_impact')['volatility'].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(mean_vol.index, mean_vol.values, marker='o', linestyle='-')
    plt.title("Average Volatility vs. Momentum Impact")
    plt.xlabel("Momentum Impact")
    plt.ylabel("Volatility (Std Dev of % Change)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    parameter_sweep()
