import numpy as np
import ruamel.yaml
from other.models.models_handler import MODELS_COUNT, NAMES
from other.parsing.parsing_utils import *


current_dir = os.path.dirname(os.path.abspath(__file__))
y_path = os.path.join(current_dir, '..', '..', 'configs', 'metrix.yaml')
yaml = ruamel.yaml.YAML(typ='rt')


# Load YAML file
with open(y_path, 'r', encoding='utf-8') as f:
    ydict = yaml.load(f)

metrix_data_path = is_type_of(ydict['data']['metrix'])
metrix_data_path = construct_full_path(current_dir, '..', '..', *split_path(metrix_data_path))

model_id = is_range(ydict['model']['id'], 0, MODELS_COUNT, int, req=False)
model_name = is_type_of(ydict['model']['name'], req=False)
# Determine model name based on id or provided name.
if model_id is not None:
    model_name = NAMES[model_id]
elif model_name is not None:
    if model_name not in NAMES:
        raise ValueError(f"Model name must be one of: {NAMES}")
else:
    raise ValueError(f"Model name or id has to be declared")

load_from = is_type_of(ydict['model']['weights'], req=True)
load_from = construct_full_path(current_dir, '..', '..', *split_path(load_from))

num_workers = is_range(ydict['metrix_evaluation']['workers'], 0, 32, int)
batch_size = is_range(ydict['metrix_evaluation']['batch'], 1, 2 ** 15, int)

thresholds = np.linspace(*parse_linspace(ydict['threshold_settings']['thresholds'], [0, 1], [0, 1], [1, 1000, int]))
