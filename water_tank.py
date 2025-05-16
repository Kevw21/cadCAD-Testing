# 1. Imports
from cadCAD.configuration import Experiment
from cadCAD.configuration.utils import config_sim
from cadCAD.engine import ExecutionContext, ExecutionMode, Executor
import pandas as pd

# 2. Initial State
initial_state = {
    'water_level': 100
}

# 3. Policy Function
def pump_policy(params, step, sL, s, **kwargs):
    return {'pump_out': 10}

# 4. State Update Function
def update_water_level(params, step, sL, s, inputs, **kwargs):
    return ('water_level', max(0, s['water_level'] - inputs['pump_out']))

# 5. Partial State Update Blocks
psubs = [
    {
        'policies': {
            'pump_policy': pump_policy
        },
        'variables': {
            'water_level': update_water_level
        }
    }
]

# 6. Simulation Config
sim_config = {
    'N': 1,               # Number of runs
    'T': range(15),       # Timesteps
    'M': {}               # No parameter sweeping
}

# 7. Set up Experiment
exp = Experiment()
exp.append_configs(
    initial_state=initial_state,
    partial_state_update_blocks=psubs,
    sim_configs=sim_config
)

# 8. Execute Simulation
exec_mode = ExecutionMode()
exec_context = ExecutionContext(exec_mode.local_mode)
executor = Executor(exec_context=exec_context, configs=exp.configs)
raw_result = executor.execute()

# 9. Process and Show Results
# The raw_result is a list containing one list of state dictionaries
states_list = raw_result[0]

# Extract the data we need
results = []
for state in states_list:
    results.append({
        'timestep': state['timestep'],
        'water_level': state['water_level']
    })

# Create a proper DataFrame
result_df = pd.DataFrame(results)
print(result_df[['timestep', 'water_level']])