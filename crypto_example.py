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
    # Lowered event strength to reflect more realistic behavior
    event_strength = np.random.normal(0, 0.002)  # ~0.2%
    if np.random.random() < 0.1:
        event_strength = np.random.normal(0, 0.01)  # ~1% rare event
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
    base_change = np.random.normal(0.0002, 0.001)  # Small average drift

    sentiment_factor = 0.5 + previous_state['market_sentiment']  # 0.5 to 1.5
    momentum_effect = previous_state['momentum'] * 0.05 * sentiment_factor  # Weakened

    # Panic sells if sentiment is very low
    event_effect = policy_input['random_event'] * (1.5 - previous_state['market_sentiment'])

    # Combine effects and clamp the change to [-2%, +2%]
    total_change = base_change + momentum_effect + event_effect
    total_change = np.clip(total_change, -0.02, 0.02)  # Clamp to ±2%

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
    'T': range(31),  # Simulate 31 timesteps
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

# Plotting
df = pd.DataFrame(raw_result)
plt.figure(figsize=(10, 5))
plt.plot(df['timestep'], df['price'])
plt.title('Realistic Cryptocurrency Price Prediction')
plt.xlabel('Time Step (e.g., hourly)')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()
