import pandas as pd
from cadCAD.engine import ExecutionMode, ExecutionContext, Executor
from cadCAD.configuration import Experiment
import numpy as np

# Initial state
initial_state = {
    'price': 109670.53,
    'price_history': [109660.59, 109663.97, 109623.48],
    'momentum': 0,
    'market_sentiment': 0.5
}

# Policy Functions
def p_random_events(params, substep, state_history, previous_state):
    event_strength = np.random.normal(0, 0.1)
    if np.random.random() < 0.1:
        event_strength = np.random.normal(0, 0.5)
    return {'random_event': event_strength}

def p_update_momentum(params, substep, state_history, previous_state):
    history = previous_state['price_history']
    if len(history) >= 3:
        change = (history[-1] - history[-3]) / history[-3]
    else:
        change = 0
    new_momentum = 0.7 * previous_state['momentum'] + 0.3 * change
    return {'momentum_change': new_momentum}

# State Update Functions
def s_price(params, substep, state_history, previous_state, policy_input):
    current_price = previous_state['price']
    base_change = np.random.normal(0.01, 0.02)
    momentum_effect = previous_state['momentum'] * 0.3
    event_effect = policy_input['random_event']
    new_price = current_price * (1 + base_change + momentum_effect + event_effect)
    new_price = max(0.01, new_price)
    return ('price', new_price)

def s_update_history(params, substep, state_history, previous_state, policy_input):
    history = previous_state['price_history'].copy()
    history.append(previous_state['price'])
    return ('price_history', history)

def s_momentum(params, substep, state_history, previous_state, policy_input):
    return ('momentum', policy_input['momentum_change'])

# Create Experiment
experiment = Experiment()

# System Configuration
sys_config = {
    'initial_state': initial_state,
    'partial_state_update_blocks': [
        {
            'policies': {
                'random_events': p_random_events,
                'momentum_calc': p_update_momentum
            },
            'variables': {
                'price': s_price,
                'price_history': s_update_history,
                'momentum': s_momentum
            }
        }
    ]
}

# Simulation Configuration
sim_config = {
    'N': 1,
    'T': range(5), 
    'M': {}
}

# Add configurations to experiment
experiment.append_configs(
    sim_configs=[sim_config],
    initial_state=sys_config['initial_state'],
    partial_state_update_blocks=sys_config['partial_state_update_blocks']
)

# Execute simulation
exec_mode = ExecutionMode()
exec_ctx = ExecutionContext(exec_mode.local_mode)
executor = Executor(exec_ctx, experiment.configs)
raw_result, tensor_field, sessions = executor.execute()

# Process results
df = pd.DataFrame(raw_result)
print("Simulation Results:")
print(df[['timestep', 'price', 'momentum']].tail())

# Plot results
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(df['timestep'], df['price'])
plt.title('Cryptocurrency Price Prediction')
plt.xlabel('Time Step')
plt.ylabel('Price (USD)')
plt.grid(True)
plt.show()