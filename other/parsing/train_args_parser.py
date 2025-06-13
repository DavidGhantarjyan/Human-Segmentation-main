import os
import numpy as np
import ruamel.yaml
from other.parsing.parsing_utils import *
from other.models.models_handler import MODELS_COUNT, NAMES

# ---------------------------
# FILE LOADING SETUP
# ---------------------------
# Determine the current directory and build the path to the train.yaml configuration file.
current_dir = os.path.dirname(os.path.abspath(__file__))
y_path = os.path.join(current_dir, '..', '..', 'configs', 'train.yaml')
# Initialize a YAML parser with round-trip capability to preserve comments and formatting.
yaml = ruamel.yaml.YAML(typ='rt')

# Load the YAML configuration into a dictionary (ydict) using a round-trip loader.
with open(y_path, 'r', encoding='utf-8') as f:
    ydict = yaml.load(f)

# ---------------------------
# DATA PATHS CONFIGURATION
# ---------------------------
# Parse and construct full paths for the training and validation COCO datasets.
train_coco_data_dir = is_type_of(ydict['data']['directories']['train_coco_data_dir'])
train_coco_data_dir = construct_full_path(current_dir, '..', '..', *split_path(train_coco_data_dir))

train_mapillary_data_dir = is_type_of(ydict['data']['directories']['train_mapillary_data_dir'])
train_mapillary_data_dir = construct_full_path(current_dir, '..', '..', *split_path(train_mapillary_data_dir))

val_coco_data_dir = is_type_of(ydict['data']['directories']['val_coco_data_dir'])
val_coco_data_dir = construct_full_path(current_dir, '..', '..', *split_path(val_coco_data_dir))

# Determine whether synthetic data should be persisted to disk.
test_safe_to_desk = is_type_of(ydict['data']['directories']['test_safe_to_desk'], bool)

# Construct paths for synthetic data generation outputs and directories for backgrounds and objects.
result_dir = is_type_of(ydict['data']['directories']['result_dir'])
result_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main', *split_path(result_dir))

background_dir = is_type_of(ydict['data']['directories']['background_dir'])
background_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main', *split_path(background_dir))

objects_dir = is_type_of(ydict['data']['directories']['objects_dir'])
objects_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main', *split_path(objects_dir))

greenscreen_obj_dir = is_type_of(ydict['data']['directories']['greenscreen_obj_dir'])
greenscreen_obj_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main', *split_path(greenscreen_obj_dir))

tiktok_obj_dir = is_type_of(ydict['data']['directories']['tiktok_obj_dir'])
tiktok_obj_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main', *split_path(tiktok_obj_dir))

tiktok_background_dir = is_type_of(ydict['data']['directories']['tiktok_background_dir'])
tiktok_background_dir = construct_full_path(current_dir, '..', '..', 'SyntheticData_main',*split_path(tiktok_background_dir))


# Output settings for generated data.
merge_outputs = is_type_of(ydict['data']['output_settings']['merge_outputs'],bool)
package_by_background = is_type_of(ydict['data']['output_settings']['package_by_background'],bool)

# ---------------------------
# SYNTHETIC DATA PROCESSING CONFIGURATION
# ---------------------------
# Object placement configuration.
tiktok_application_chance = is_range(ydict['data']['synthetic_data_settings']['image_generation_application_settings']['tiktok_application_chance'],0,1)
greenscreen_application_chance = is_range(ydict['data']['synthetic_data_settings']['image_generation_application_settings']['greenscreen_application_chance'],0,1)

allow_out_of_bounds = is_type_of(ydict['data']['synthetic_data_settings']['object_placement_settings']['allow_out_of_bounds'],bool)
out_of_bounds_range = parse_range(ydict['data']['synthetic_data_settings']['object_placement_settings']['out_of_bounds_range'],[0,1],[0,1])
placement_distribution = is_type_of(ydict['data']['synthetic_data_settings']['object_placement_settings']['placement_distribution'])

# Image generation configuration.
images_per_combination = is_range(ydict['data']['synthetic_data_settings']['image_generation_settings']['images_per_combination'],fr=1,to=2**32,tp=int)
objects_per_image_range = parse_range(ydict['data']['synthetic_data_settings']['image_generation_settings']['objects_per_image_range'],[1,len(os.listdir(objects_dir))],[1,len(os.listdir(objects_dir))])
object_scale_range = parse_range(ydict['data']['synthetic_data_settings']['image_generation_settings']['object_scale_range'],[10**-2,2],[10**-2,2])

# Background blur configuration.
synthetic_bg_blur_type = is_type_of(ydict['data']['synthetic_data_settings']['background_blur_settings']['bg_blur_type'],str)
synthetic_bg_blur_probability = is_range(ydict['data']['synthetic_data_settings']['background_blur_settings']['bg_blur_probability'],0,1)
synthetic_bg_blur_kernel_range = parse_range(ydict['data']['synthetic_data_settings']['background_blur_settings']['bg_blur_kernel_range'],[3,100],[3,100])
real_bg_blur_type = is_type_of(ydict['data']['real_data_settings']['background_blur_settings']['bg_blur_type'],str)
real_bg_blur_probability = is_range(ydict['data']['real_data_settings']['background_blur_settings']['bg_blur_probability'],0,1)
real_bg_blur_kernel_range = parse_range(ydict['data']['real_data_settings']['background_blur_settings']['bg_blur_kernel_range'],[3,100],[3,100])
# blur_intensity_range = parse_range(ydict['data']['blur_settings']['blur_intensity_range'],[10**-2,100],[10**-2,100])

# Alpha blur configuration.
alpha_blur_type = is_type_of(ydict['data']['synthetic_data_settings']['alpha_blur_settings']['alpha_blur_type'],str)
video_matte_alpha_blur_kernel_range = parse_range(ydict['data']['synthetic_data_settings']['alpha_blur_settings']['video_matte_alpha_blur_kernel_range'],[1,100],[1,100])
tiktok_alpha_blur_kernel_range = parse_range(ydict['data']['synthetic_data_settings']['alpha_blur_settings']['tiktok_alpha_blur_kernel_range'],[1,100],[1,100])
synthetic_alpha_blur_kernel_range = parse_range(ydict['data']['synthetic_data_settings']['alpha_blur_settings']['synthetic_alpha_blur_kernel_range'],[1,100],[1,100])

# alpha_blur_intensity_range = parse_range(ydict['data']['alpha_blur_settings']['alpha_blur_intensity_range'],[0,100],[0,100])

# Resize settings for images.
synthetic_resize_width = is_range(ydict['data']['synthetic_data_settings']['resize_setting']['resize_width'],fr=160, to=640, tp=int)
synthetic_resize_height = is_range(ydict['data']['synthetic_data_settings']['resize_setting']['resize_height'],fr=90, to=360, tp=int)
real_resize_width = is_range(ydict['data']['real_data_settings']['resize_setting']['resize_width'],fr=160, to=640, tp=int)
real_resize_height = is_range(ydict['data']['real_data_settings']['resize_setting']['resize_height'],fr=90, to=360, tp=int)

# Noise configuration.
synthetic_noise_type = is_type_of(ydict['data']['synthetic_data_settings']['noise_settings']['noise_type'],str)
synthetic_noise_level_range = parse_range(ydict['data']['synthetic_data_settings']['noise_settings']['noise_level_range'],[0,1],[0,1])
real_noise_type = is_type_of(ydict['data']['real_data_settings']['noise_settings']['noise_type'],str)
real_noise_level_range = parse_range(ydict['data']['real_data_settings']['noise_settings']['noise_level_range'],[0,1],[0,1])


# Transformation settings.
synthetic_flip_probability = is_range(ydict['data']['synthetic_data_settings']['transformations']['flip_probability'],0,1)
real_flip_probability = is_range(ydict['data']['real_data_settings']['transformations']['flip_probability'],0,1)
# Output format for generated data.
output_format = is_type_of(ydict['data']['synthetic_data_settings']['output_format']['output_format'],str)
# Miscellaneous settings.
depth = is_type_of(ydict['data']['synthetic_data_settings']['miscellaneous']['depth'],bool)
# Rotation settings.
foreground_rotation = is_type_of(ydict['data']['synthetic_data_settings']['rotation_settings']['foreground_rotation'],bool)
foreground_rotation_angle = parse_range(ydict['data']['synthetic_data_settings']['rotation_settings']['foreground_rotation_angle'],[-30,30],[-30,30])

synthetic_background_rotation = is_type_of(ydict['data']['synthetic_data_settings']['rotation_settings']['background_rotation'],bool)
synthetic_background_rotation_angle = parse_range(ydict['data']['synthetic_data_settings']['rotation_settings']['background_rotation_angle'],[-30,30],[-30,30])

real_background_rotation = is_type_of(ydict['data']['real_data_settings']['rotation_settings']['background_rotation'],bool)
real_background_rotation_angle = parse_range(ydict['data']['real_data_settings']['rotation_settings']['background_rotation_angle'],[-30,30],[-30,30])


# Color jitter settings.
synthetic_jitter_probability = is_range(ydict['data']['synthetic_data_settings']['color_jitter_settings']['jitter_probability'],0,1)
real_jitter_probability = is_range(ydict['data']['real_data_settings']['color_jitter_settings']['jitter_probability'],0,1)

# brightness_range = parse_range(ydict['data']['color_jitter_settings']['brightness_range'],[0,2],[0,2])
# contrast_range = parse_range(ydict['data']['color_jitter_settings']['contrast_range'],[0,2],[0,2])
synthetic_saturation_range = parse_range(ydict['data']['synthetic_data_settings']['color_jitter_settings']['saturation_range'],[0,2],[0,2])
synthetic_hue_range = parse_range(ydict['data']['synthetic_data_settings']['color_jitter_settings']['hue_range'],[-1,0],[0,1])
real_saturation_range = parse_range(ydict['data']['real_data_settings']['color_jitter_settings']['saturation_range'],[0,2],[0,2])
real_hue_range = parse_range(ydict['data']['real_data_settings']['color_jitter_settings']['hue_range'],[-1,0],[0,1])


synthetic_clahe_probability = is_type_of(ydict['data']['synthetic_data_settings']['albumentations_settings']['clahe_probability'], float)
synthetic_tile_grid_size = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['contrast_tile_grid_size'], [1, 16], [1, 16])
synthetic_clip_limit = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['contrast_clip_limit'], 2, 6, float)
real_clahe_probability = is_type_of(ydict['data']['real_data_settings']['albumentations_settings']['clahe_probability'], float)
real_tile_grid_size = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['contrast_tile_grid_size'], [1, 16], [1, 16])
real_clip_limit = is_range(ydict['data']['real_data_settings']['albumentations_settings']['contrast_clip_limit'], 2, 6, float)


# albumentations_settings
x_range = ((0, 1), (0, 1))
y_range = ((0, 1), (0, 1))
synthetic_full_img_shadow_roi = parse_four_element_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_shadow_roi'],x_range,y_range)
synthetic_full_img_num_shadows_limit = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_num_shadows_limit'], [1,10], [1,10],int)
synthetic_full_shadow_dimension = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_shadow_dimension'], 1, 10, tp=int)
synthetic_full_shadow_intensity_range = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_shadow_intensity_range'], [0,1], [0,1], float)
synthetic_full_shadow_transform_probability = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_shadow_transform_probability'], 0, 1)

synthetic_cropped_img_shadow_roi = parse_four_element_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['cropped_img_shadow_roi'],x_range,y_range)
synthetic_cropped_img_num_shadows_limit = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['cropped_img_num_shadows_limit'], [1,10], [1,10],int)
synthetic_cropped_img_shadow_dimension = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['cropped_img_shadow_dimension'], 1, 10, tp=int)
synthetic_cropped_img_shadow_intensity_range = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['cropped_img_shadow_intensity_range'], [0,1], [0,1], float)
synthetic_cropped_img_shadow_transform_probability = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['cropped_img_shadow_transform_probability'], 0, 1)

real_shadow_roi = parse_four_element_range(ydict['data']['real_data_settings']['albumentations_settings']['shadow_roi'],x_range,y_range)
real_num_shadows_limit = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['num_shadows_limit'], [1,10], [1,10],int)
real_shadow_dimension = is_range(ydict['data']['real_data_settings']['albumentations_settings']['shadow_dimension'], 1, 10, tp=int)
real_shadow_intensity_range = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['shadow_intensity_range'], [0,1], [0,1], float)
real_shadow_transform_probability = is_range(ydict['data']['real_data_settings']['albumentations_settings']['shadow_transform_probability'], 0, 1)

synthetic_crop_probability = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['crop_probability'], 0, 1)
synthetic_max_part_shift = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['max_part_shift'], [0,1], [0,1], float)

synthetic_gamma_probability = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['gamma_probability'],0,1)
synthetic_gamma_limit = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['gamma_limit'],[0,200],[0,200])
real_gamma_probability = is_range(ydict['data']['real_data_settings']['albumentations_settings']['gamma_probability'],0,1)
real_gamma_limit = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['gamma_limit'],[0,200],[0,200])


synthetic_sigmoid_probability = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['sigmoid_probability'],0,1)
synthetic_sigmoid_limit = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['sigmoid_limit'],[0,10],[0,10])
real_sigmoid_probability = is_range(ydict['data']['real_data_settings']['albumentations_settings']['sigmoid_probability'],0,1)
real_sigmoid_limit = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['sigmoid_limit'],[0,10],[0,10])


synthetic_contrast_brightness_probability = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['contrast_brightness_probability'],0,1)
synthetic_brightness_range = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['brightness_range'],[0,2],[0,2])
synthetic_contrast_range = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['contrast_range'],[0,2],[0,2])
real_contrast_brightness_probability = is_range(ydict['data']['real_data_settings']['albumentations_settings']['contrast_brightness_probability'],0,1)
real_brightness_range = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['brightness_range'],[0,2],[0,2])
real_contrast_range = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['contrast_range'],[0,2],[0,2])


synthetic_motion_blur_probability = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['motion_blur_probability'],0,1)
synthetic_motion_blur_limit = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['motion_blur_limit'],[3,100],[3,100])

real_motion_blur_probability = is_range(ydict['data']['real_data_settings']['albumentations_settings']['motion_blur_probability'],0,1)
real_motion_blur_limit = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['motion_blur_limit'],[3,100],[3,100])

# contrast_brightness_probability = is_range(ydict['data']['albumentations_settings']['contrast_brightness_probability'],0,1)
# brightness_limit = is_range(ydict['data']['albumentations_settings']['brightness_limit'],0,1)
# contrast_limit = is_range(ydict['data']['albumentations_settings']['contrast_limit'],0,1)

synthetic_full_img_dropout_probability = is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_dropout_probability'],0,1)
synthetic_full_img_dropout_holes_range= parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_dropout_holes_range'],[0,16],[0,16])
synthetic_full_img_dropout_height_range =parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_dropout_height_range'],[0,100],[0,100])
synthetic_full_img_dropout_width_range = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_dropout_width_range'],[0,100],[0,100])
synthetic_full_img_dropout_mask_fill_value = is_type_of(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_dropout_mask_fill_value'], int)
synthetic_full_img_dropout_roi = parse_four_element_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['full_img_dropout_roi'],x_range,y_range)

synthetic_crop_img_dropout_probability =  is_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['crop_img_dropout_probability'],0,1)
synthetic_crop_img_dropout_holes_range = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['crop_img_dropout_holes_range'],[0,16],[0,16])
synthetic_crop_img_dropout_height_range = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['crop_img_dropout_height_range'],[0,100],[0,100])
synthetic_crop_img_dropout_width_range = parse_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['crop_img_dropout_width_range'],[0,100],[0,100])
synthetic_crop_img_dropout_mask_fill_value = is_type_of(ydict['data']['synthetic_data_settings']['albumentations_settings']['crop_img_dropout_mask_fill_value'], int)
synthetic_crop_img_dropout_roi = parse_four_element_range(ydict['data']['synthetic_data_settings']['albumentations_settings']['crop_img_dropout_roi'],x_range,y_range)

real_dropout_probability = is_range(ydict['data']['real_data_settings']['albumentations_settings']['dropout_probability'],0,1)
real_dropout_holes_range= parse_range(ydict['data']['real_data_settings']['albumentations_settings']['dropout_holes_range'],[0,16],[0,16])
real_dropout_height_range =parse_range(ydict['data']['real_data_settings']['albumentations_settings']['dropout_height_range'],[0,100],[0,100])
real_dropout_width_range = parse_range(ydict['data']['real_data_settings']['albumentations_settings']['dropout_width_range'],[0,100],[0,100])
real_dropout_mask_fill_value = is_type_of(ydict['data']['real_data_settings']['albumentations_settings']['dropout_mask_fill_value'], int)
real_dropout_roi = parse_four_element_range(ydict['data']['real_data_settings']['albumentations_settings']['dropout_roi'],x_range,y_range)

# Dataset scaling factor.
scale_factor = is_range(ydict['data']['dataset_settings']['scale_factor'],1,3)


# ---------------------------
# REAL DATA PROCESSING CONFIGURATION
# ---------------------------

# ---------------------------
# MODEL CONFIGURATION
# ---------------------------
# Parse model id and name from configuration.
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

saves_count = is_range(ydict['model']['saves_count'], 0, 100, int)

create_new_model = is_type_of(ydict['model']['create_new_model'], bool, req=False)
train_res_dir = is_type_of(ydict['model']['directory'])
weights_load_from = is_type_of(ydict['model']['weights'], req=False)

# ---------------------------
# TRAINING CONFIG
# ---------------------------
available_types_of_computing = ['loss', 'dataloader']

# Synthetic mask handling configuration.
synthetic_mask_pre_calculation_mode = is_type_of(ydict['train']['synthetic_mask_pre_calculation_mode'], req=True)
if synthetic_mask_pre_calculation_mode not in available_types_of_computing:
    raise ValueError(f'synthetic_mask_pre_calculation_mode must be one of: {available_types_of_computing}')


# Natural data mask handling configuration.
natural_data_mask_saving = is_type_of(ydict['train']['natural_data_mask_saving'], bool, req=False)
natural_data_mask_pre_calculation_mode = is_type_of(ydict['train']['natural_data_mask_pre_calculation_mode'], req=False)


if natural_data_mask_pre_calculation_mode not in available_types_of_computing:
    natural_data_mask_pre_calculation_mode = None
if not natural_data_mask_saving :
   if not natural_data_mask_pre_calculation_mode:
        raise ValueError("Either natural_data_mask_saving or natural_data_mask_pre_calculation_mode has to be declared")
if natural_data_mask_saving and natural_data_mask_pre_calculation_mode:
    raise ValueError('Only one of them can be declared')

# Training hyperparameters.
lr = 5 * 10 ** is_range(ydict['train']['lr'], -10, 10)
enable_lr_reset = is_type_of(ydict['train']['enable_lr_reset'], bool)
scheduler_available = is_type_of(ydict['train']['scheduler_available'], bool)
if enable_lr_reset:
    scheduler_available = False
do_epoches = is_range(ydict['train']['epoch'], 0, 1000, int)
num_workers = is_range(ydict['train']['workers'], 0, 32, int)
batch_size = is_range(ydict['train']['batch'], 1, 2 ** 15, int)
accumulation_steps = is_range(ydict['train']['n_accum'], 1, 2 ** 15, int)

# Loss component weights.
alpha = is_range(ydict['train']['alpha'], 0.0, 10**2, float)
beta = is_range(ydict['train']['beta'], 0.0, 10**2, float)
gamma = is_range(ydict['train']['gamma'], 0.0, 10**2, float)
delta = is_range(ydict['train']['delta'], 0.0, 10**2, float)
eta = is_range(ydict['train']['eta'], 0.0, 10**2, float)
sigma = is_range(ydict['train']['sigma'], 0.0, 10**2, float)
epsilon = is_range(ydict['train']['epsilon'], 0.0, 10**2, float)


# Checkpoint saving logic.
if saves_count == 0:
    saves_count = do_epoches
elif saves_count > do_epoches:
    raise ValueError(f"Saves count must be less than epoches count to do: {do_epoches}")

# Compute save intervals using linear spacing.
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
        # Overwrite the YAML configuration file with the updated settings.
        with open(y_path, 'w', encoding='utf-8') as f:
            yaml.dump(ydict, f)
    except IOError:
        print("WARNING: Couldn't change yaml file content")
