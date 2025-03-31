import os
import numpy as np
import ruamel.yaml
from other.parsing.parsing_utils import *
from other.models.models_handler import MODELS_COUNT, NAMES

# ---------------------------
# FILE LOADING SETUP
# ---------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
y_path = os.path.join(current_dir, '..', '..', 'configs', 'train.yaml')
yaml = ruamel.yaml.YAML(typ='rt')  # Preserve YAML comments/formatting

# Load YAML configuration with round-trip loader
with open(y_path,'r', encoding='utf-8') as f:
    ydict = yaml.load(f)

# ---------------------------
# DATA PATHS CONFIGURATION
# ---------------------------
train_coco_data_dir = is_type_of(ydict['data']['directories']['train_coco_data_dir'])
train_coco_data_dir = construct_full_path(current_dir, '..', '..', *split_path(train_coco_data_dir))

val_coco_data_dir = is_type_of(ydict['data']['directories']['val_coco_data_dir'])
val_coco_data_dir = construct_full_path(current_dir, '..', '..', *split_path(val_coco_data_dir))

test_safe_to_desk = is_type_of(ydict['data']['directories']['test_safe_to_desk'], bool)

result_dir = is_type_of(ydict['data']['directories']['result_dir'])
result_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main', *split_path(result_dir))

background_dir = is_type_of(ydict['data']['directories']['background_dir'])
background_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main', *split_path(background_dir))

objects_dir = is_type_of(ydict['data']['directories']['objects_dir'])
objects_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main', *split_path(objects_dir))

# ---------------------------
# DATA PROCESSING CONFIG
# ---------------------------
# OUTPUT SETTINGS (data.output_settings)
merge_outputs = is_type_of(ydict['data']['output_settings']['merge_outputs'],bool)
package_by_background = is_type_of(ydict['data']['output_settings']['package_by_background'],bool)
# OBJECT PLACEMENT (data.object_placement_settings)

allow_out_of_bounds = is_type_of(ydict['data']['object_placement_settings']['allow_out_of_bounds'],bool)
out_of_bounds_range = parse_range(ydict['data']['object_placement_settings']['out_of_bounds_range'],[0,1],[0,1])
placement_distribution = is_type_of(ydict['data']['object_placement_settings']['placement_distribution'])

# IMAGE GENERATION (data.image_generation_settings)
images_per_combination = is_range(ydict['data']['image_generation_settings']['images_per_combination'],fr=1,to=2**32,tp=int)
objects_per_image_range = parse_range(ydict['data']['image_generation_settings']['objects_per_image_range'],[1,len(os.listdir(objects_dir))],[1,len(os.listdir(objects_dir))])
object_scale_range = parse_range(ydict['data']['image_generation_settings']['object_scale_range'],[10**-2,2],[10**-2,2])
# BACKGROUND BLUR (data.background_blur_settings)
bg_blur_type = is_type_of(ydict['data']['background_blur_settings']['bg_blur_type'],str)
bg_blur_probability = is_range(ydict['data']['background_blur_settings']['bg_blur_probability'],0,1)
bg_blur_kernel_range = parse_range(ydict['data']['background_blur_settings']['bg_blur_kernel_range'],[10**-2,10],[10**-2,10])
# blur_intensity_range = parse_range(ydict['data']['blur_settings']['blur_intensity_range'],[10**-2,100],[10**-2,100])

# ALPHA BLUR (data.alpha_blur_settings)
alpha_blur_type = is_type_of(ydict['data']['alpha_blur_settings']['alpha_blur_type'],str)
alpha_blur_kernel_range = parse_range(ydict['data']['alpha_blur_settings']['alpha_blur_kernel_range'],[10**-2,10],[10**-2,10])
# alpha_blur_intensity_range = parse_range(ydict['data']['alpha_blur_settings']['alpha_blur_intensity_range'],[0,100],[0,100])

# RESIZE SETTINGS (data.resize_setting)
resize_width = is_range(ydict['data']['resize_setting']['resize_width'],fr=160, to=640, tp=int)
resize_height = is_range(ydict['data']['resize_setting']['resize_height'],fr=90, to=360, tp=int)

# NOISE SETTINGS (data.noise_settings)
noise_type = is_type_of(ydict['data']['noise_settings']['noise_type'],str)
noise_level_range = parse_range(ydict['data']['noise_settings']['noise_level_range'],[0,1],[0,1])

# TRANSFORMATIONS (data.transformations)
flip_probability = is_type_of(ydict['data']['transformations']['flip_probability'],float)
# OUTPUT FORMAT (data.output_format)
output_format = is_type_of(ydict['data']['output_format']['output_format'],str)
# MISCELLANEOUS (data.miscellaneous)
depth = is_type_of(ydict['data']['miscellaneous']['depth'],bool)
# ROTATION SETTINGS (data.rotation_settings)
foreground_rotation = is_type_of(ydict['data']['rotation_settings']['foreground_rotation'],bool)
background_rotation = is_type_of(ydict['data']['rotation_settings']['background_rotation'],bool)
background_rotation_angle = parse_range(ydict['data']['rotation_settings']['background_rotation_angle'],[-30,30],[-30,30])
foreground_rotation_angle = parse_range(ydict['data']['rotation_settings']['foreground_rotation_angle'],[-30,30],[-30,30])
# COLOR JITTER (data.color_jitter_settings)

jitter_probability = is_type_of(ydict['data']['color_jitter_settings']['jitter_probability'],float)
brightness_range = parse_range(ydict['data']['color_jitter_settings']['brightness_range'],[0,2],[0,2])
contrast_range = parse_range(ydict['data']['color_jitter_settings']['contrast_range'],[0,2],[0,2])
saturation_range = parse_range(ydict['data']['color_jitter_settings']['saturation_range'],[0,2],[0,2])
hue_range = parse_range(ydict['data']['color_jitter_settings']['hue_range'],[-1,0],[0,1])
# DATASET SETTINGS (data.dataset_settings)
scale_factor = is_range(ydict['data']['dataset_settings']['scale_factor'],1,2)



# ---------------------------
# MODEL CONFIGURATION
# ---------------------------
model_id = is_range(ydict['model']['id'], 0, MODELS_COUNT, int, req=False)
model_name = is_type_of(ydict['model']['name'], req=False)
# Model identification logic
if model_id is not None:
    model_name = NAMES[model_id]
elif model_name is not None:
    if model_name not in NAMES:
        raise ValueError(f"Model name must be one of: {NAMES}")
else:
    raise ValueError(f"Model name or id has to be declared")

saves_count = is_range(ydict['model']['saves_count'], 0, 100, int)

create_new_model = is_type_of(ydict['model']['create_new_model'], bool, req=False)
train_res_dir = is_type_of(ydict['model']['directory'])
weights_load_from = is_type_of(ydict['model']['weights'], req=False)

# ---------------------------
# TRAINING CONFIG
# ---------------------------
available_types_of_computing = ['loss', 'dataloader']

# Synthetic mask handling
synthetic_mask_pre_calculation_mode = is_type_of(ydict['train']['synthetic_mask_pre_calculation_mode'], req=True)
if synthetic_mask_pre_calculation_mode not in available_types_of_computing:
    raise ValueError(f'synthetic_mask_pre_calculation_mode must be one of: {available_types_of_computing}')


# # Natural data mask handling
natural_data_mask_saving = is_type_of(ydict['train']['natural_data_mask_saving'], bool, req=False)
natural_data_mask_pre_calculation_mode = is_type_of(ydict['train']['natural_data_mask_pre_calculation_mode'], req=False)


if natural_data_mask_pre_calculation_mode not in available_types_of_computing:
    natural_data_mask_pre_calculation_mode = None
#   natural_data_mask_pre_calculation_mode->  Kam loss/ dataloader kam None | natural_data_mask_saving kam None kam True kam false
if not natural_data_mask_saving :
   if not natural_data_mask_pre_calculation_mode:
        raise ValueError("Either natural_data_mask_saving or natural_data_mask_pre_calculation_mode has to be declared")
if natural_data_mask_saving and natural_data_mask_pre_calculation_mode:
    raise ValueError('Only one of them can be declared')

# # Training hyperparameters
lr = 10 ** is_range(ydict['train']['lr'], -10, 10)
do_epoches = is_range(ydict['train']['epoch'], 0, 1000, int)
num_workers = is_range(ydict['train']['workers'], 0, 32, int)
batch_size = is_range(ydict['train']['batch'], 1, 2 ** 15, int)
accumulation_steps = is_range(ydict['train']['n_accum'], 1, 2 ** 15, int)

# # Loss components weights
alpha = is_range(ydict['train']['alpha'], 10**-6, 10**2, float)
beta = is_range(ydict['train']['beta'], 10**-6, 10**2, float)
gamma = is_range(ydict['train']['gamma'], 10**-6, 10**2, float)

# # Checkpoint saving logic
if saves_count == 0:
    saves_count = do_epoches
elif saves_count > do_epoches:
    raise ValueError(f"Saves count must be less than epoches count to do: {do_epoches}")

# # Generate save intervals using linear spacing
save_frames = np.linspace(do_epoches / saves_count, do_epoches, saves_count, dtype=int)

# ---------------------------
# VALIDATION CONFIG
# ---------------------------
val_every = is_range(ydict['val']['every'], 0, 1000, int)
val_num_workers = is_range(ydict['val']['workers'], 0, 32, int)
val_batch_size = is_range(ydict['val']['mini_batch'], 1, 2 ** 15, int)

# ---------------------------
# VERBOSITY SETTINGS
# ---------------------------
plot = is_type_of(ydict['verbose']['plot'], bool)
print_level = is_range(ydict['verbose']['print'], 0, 2, int)
threshold = is_range(ydict['verbose']['threshold'], 0, 1, float)
# n_examples = is_range(ydict['verbose']['n_examples'], 0, 1000, int)
#

# ---------------------------
# CONFIG PERSISTENCE FUNCTION
# ---------------------------
def model_has_been_saved():
    global ydict
    ydict['model']['weights'] = None
    ydict['model']['create_new_model'] = False

    try:
        # Write modified config back to file
        with open(y_path, 'w') as f:
            yaml.dump(ydict, f)
    except IOError:
        print("WARNING: Couldn't change yaml file content")
