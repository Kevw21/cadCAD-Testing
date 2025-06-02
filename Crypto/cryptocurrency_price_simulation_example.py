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
    momentum_effect = previous_state['momentum'] * 0.05 * sentiment_factor
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

# Main simulation runner
def run_simulation():
    runs = 30  # Number of Monte Carlo simulations
    all_runs = []

    for run_id in range(runs):
        # Setup experiment
        initial_state = {
            'price': 104745.90,
            'price_history': [104706.40, 104550.60, 104230.80, 104314.40, 104499.80],
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
            'M': {}
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
        all_runs.append(df)

    combined_df = pd.concat(all_runs)

    # Calculate average prices across runs
    avg_prices = combined_df.groupby('timestep')['price'].mean().reset_index(name='avg_price')

    # Plotting
    plt.figure(figsize=(12, 6))
    for run_id in range(runs):
        subset = combined_df[combined_df['run'] == run_id]
        plt.plot(subset['timestep'], subset['price'], color='gray', alpha=0.3, linewidth=1)

    plt.plot(avg_prices['timestep'], avg_prices['avg_price'], color='blue', label='Average Price', linewidth=2.5)
    plt.title(f'{runs}-Run Monte Carlo Simulation of Cryptocurrency Price')
    plt.xlabel('Hourly Timesteps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Hourly fluctuation from average
    fluctuations = []
    prices = avg_prices['avg_price'].values
    for i in range(1, len(prices)):
        prev_price = prices[i - 1]
        curr_price = prices[i]
        change_pct = ((curr_price - prev_price) / prev_price) * 100
        fluctuations.append({
            'From Hour': i - 1,
            'To Hour': i,
            'Price Change (%)': round(change_pct, 4)
        })

    fluctuation_df = pd.DataFrame(fluctuations)
    print("\nHourly Price Fluctuations Based on Average (%):")
    print(fluctuation_df.to_string(index=False))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run_simulation()
