from cadcad.configuration import Experiment
from cadcad.engine import ExecutionMode, ExecutionContext, Executor
from cadcad.configuration.utils import config_sim
import pandas as pd

# 1. Initial State
initial_state = {
    'counter': 0
}

# 2. Simulation Configuration
sim_config = config_sim(
    {
        'T': range(5),  # Simulate 5 time steps
        'N': 1,         # One run
        'M': {}
    }
)

# 3. Policy Function
def increase_policy(_params, step, sL, s, **kwargs):
    return {'increase': 1}

# 4. State Update Function
def update_counter(_params, step, sL, s, inputs, **kwargs):
    return ('counter', s['counter'] + inputs['increase'])

partial_state_update_blocks = [
    {
        'policies': {
            'increase_policy': increase_policy
        },
        'variables': {
            'counter': update_counter
        }
    }
]

# 5. Create Experiment
exp = Experiment()
exp.append_model(
    initial_state=initial_state,
    partial_state_update_blocks=partial_state_update_blocks,
    sim_config=sim_config
)

# 6. Run the simulation
exec_context = ExecutionContext(context=ExecutionMode().single_threaded)
executor = Executor(exec_context=exec_context, configs=exp.configs)
raw_result, _ = executor.execute()

# 7. Show results
df = pd.DataFrame(raw_result)
print(df[['timestep', 'counter']])
