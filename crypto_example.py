import numpy as np
import pandas as pd
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD.configuration import Experiment
import matplotlib.pyplot as plt

# Initial state
initial_state = {
    'price': 109670.53,
    'price_history': [109660.59, 109663.97, 109623.48],
    'momentum': 0,
    'market_sentiment': 0.5
}

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
    total_change = np.clip(total_change, -0.02, 0.02)

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

# cadCAD Experiment Setup
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

# Create DataFrame from raw results
df = pd.DataFrame(raw_result)

# Calculate Hourly Fluctuation %
fluctuations = []
prices = df['price'].values

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
print("\nHourly Price Fluctuations (%):")
print(fluctuation_df.to_string(index=False))

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['timestep'], df['price'])
plt.title('Cryptocurrency Price Prediction')
plt.xlabel('Hourly Timesteps')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()