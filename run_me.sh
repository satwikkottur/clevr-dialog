# Step 1: Generate the CLEVR-Dialog dataset.
ROOT='data/clevr/CLEVR_v1.0/json/full/'
DATA_ROOT='data/clevr/CLEVR_v1.0/'
#  python -u generate_dataset.py \
#    --scene_path=${DATA_ROOT}"scenes/CLEVR_train_scenes.json" \
#    --num_beams=50 \
#    --num_workers=30 \
#    --save_path=${DATA_ROOT}"json/v3/clevr_train_raw_70k.json"
# --save_path=${DATA_ROOT}"json/new_cap_fix/clevr_train_raw.json"

# python -u generate_dataset.py \
#   --scene_path=${DATA_ROOT}"scenes/CLEVR_val_scenes.json" \
#   --num_beams=50 \
#   --num_workers=30 \
#   --save_path=${DATA_ROOT}"json/v3/clevr_test_raw_70k.json"

# DATA_PARTS=${DATA_ROOT}"json/new_cap_fix/clevr_train_raw.json,"
# DATA_PARTS+=${DATA_ROOT}"json/new_cap_fix/clevr_train_missing_raw.json"
# python -u util/merge_dataset_parts.py \
#   --dataset_files=${DATA_PARTS} \
#   --save_path=${DATA_ROOT}"json/new_cap_fix/clevr_train_merge_raw.json"


# Step 2: Create lighter version of CLEVR-Dialog datasets.
# Remove the history/graph annotations and save only dialog text and metainfo.
# DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# for SPLIT in "train" "test"; do
#   echo "Creating light dataset: "$SPLIT
#   python -u util/create_light_dataset.py \
#     --load_path=${DATA_ROOT}"clevr_"$SPLIT"_raw_70k.json"
# done 


# Step 3: Creating a vdformat version of CLEVR-Dialog datasets.
# DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# for SPLIT in "train" "test"; do
#   echo "Converting dataset: "$SPLIT
#   python util/convert_json.py \
#     --load_path=${DATA_ROOT}"clevr_"$SPLIT"_raw_70k_light.json" \
#     --save_path=${DATA_ROOT}"clevr_"$SPLIT"_raw_70k_vdformat.json"
# done 


# Step 4: Split the train set into train and val (first 1000 dialogs)
# DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# echo 'Splitting train into train and val'
# python util/split_train_val.py \
#   --load_path=${DATA_ROOT}"clevr_train_raw_70k_vdformat.json" \
#   --train_save_path=${DATA_ROOT}"clevr_train_split_vdformat.json" \
#   --val_save_path=${DATA_ROOT}"clevr_val_split_vdformat.json"


# Step 5: Extract annotations (coreference and bounding box).
DATA_ROOT='data/clevr/CLEVR_v1.0/'
# For train dataset.
# python util/generate_annotations.py \
#   --dialog_file=${DATA_ROOT}"json/v3/clevr_train_raw_70k.json" \
#   --dataset_file=${DATA_ROOT}"scenes/CLEVR_train_scenes.json" \
#   --bbox_file=${DATA_ROOT}"bboxes/annotations_train.json" \
#   --template_root="templates/" \
#   --save_path=${DATA_ROOT}"json/v3/clevr_dialog_annotations_train.json"
# For test dataset.
# python util/generate_annotations.py \
#   --dialog_file=${DATA_ROOT}"json/v3/clevr_test_raw_70k.json" \
#   --dataset_file=${DATA_ROOT}"scenes/CLEVR_val_scenes.json" \
#   --bbox_file=${DATA_ROOT}"bboxes/annotations_test.json" \
#   --template_root="templates/" \
#   --save_path=${DATA_ROOT}"json/v3/clevr_dialog_annotations_test.json"


# Step 6: Visualizing annotations via a html page.
ROOT="data/clevr/CLEVR_v1.0/"
DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# rm results.zip
# rm results/images/*
# python util/visualize_annotations.py \
#   --dataset_path=${DATA_ROOT}"clevr_train_raw_70k_light.json" \
#   --annotation_path=${DATA_ROOT}"clevr_dialog_annotations_train.json" \
#   --html_path="util/visualize_table_template.html" \
#   --image_root=${ROOT}"images/train" \
#   --save_path="results"
# python util/visualize_annotations.py \
#   --dataset_path=${DATA_ROOT}"clevr_test_raw_70k_light.json" \
#   --annotation_path=${DATA_ROOT}"clevr_dialog_annotations_test.json" \
#   --html_path="util/visualize_table_template.html" \
#   --image_root=${ROOT}"images/val" \
#   --save_path="results/"
# zip -r results.zip results/


# Step 7: Plotting dataset statistics.
# DATA_ROOT='data/clevr/CLEVR_v1.0/json/v3/'
# python util/visualize_dataset_stats.py \
#   --train_path=${DATA_ROOT}"clevr_train_raw_70k.json" \
#   --test_path=${DATA_ROOT}"clevr_test_raw_70k.json"
#   --train_path=${DATA_ROOT}"clevr_test_raw_70k_light.json" \
#   --test_path=${DATA_ROOT}"clevr_test_raw_70k_light.json"


# Step 8: Create VisDial datamats
ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# python util/prepro.py \
#   -input_json_train $ROOT"clevr_train_split_vdformat.json" \
#   -input_json_val $ROOT"clevr_val_split_vdformat.json" \
#   -input_json_test $ROOT"clevr_test_split_vdformat.json" \
#   -output_h5 $ROOT"clevr_data.h5" \
#   -output_json $ROOT"clevr_params.json"

# DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# python util/prepro_test.py \
#   -input_json_test $DATA_ROOT'clevr_test_split_vdformat.json' \
#   -input_param_json $DATA_ROOT'clevr_params.json' \
#   -output_h5 $DATA_ROOT'clevr_testdata.h5'


# Step 9: Get accuracy per question type
DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# python util/compute_accuracy_per_type.py \
#   --rank_json="visdial-clevr/model_pool.json" \
#   --dataset_path=${DATA_ROOT}"clevr_test_raw_70k.json"

DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# METRIC_ROOT="/nethome/skottur3/repos/copies/clevr-dialog/visdial-nmn/results/"
METRIC_ROOT="/nethome/skottur3/repos/visdial-nmn/results/"
python util/visualize_grounding_metrics.py \
  --metric_path=${METRIC_ROOT}"grounding_metrics.npy" \
  --dataset_path=${DATA_ROOT}"clevr_train_raw_70k_light.json"
# -----------------------------------------------------------------------------

# Fixing the errors in question generation.
# DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# python util/fix_question_reference_problem.py \
#   --load_path=${DATA_ROOT}"clevr_train_raw_70k.json" \
#   --save_path=${DATA_ROOT}"clevr_train_raw_70k_fixed.json"
 #  --load_path=${DATA_ROOT}"clevr_test_raw_70k.json" \
 #  --save_path=${DATA_ROOT}"clevr_test_raw_70k_fixed.json"

# python util/splitTrainVal.py
#python util/readBestModel.py
#python util/statistics.py
# python checkAnswerDistribution.py
#python util/modelAccuracyVsCoref.py
#python util/statistics.py

# Figure out the obj-relation caption group situation.
# --dialog_file=${DATA_ROOT}"clevr_dialogs_debug_test_raw.json"\
# --dialog_file='data/clevr/CLEVR_v1.0/json/new_cap_fix/clevr_train_merge_raw.json' \
# DATA_ROOT='data/clevr/CLEVR_v1.0/json/new_cap_fix/'
# DATA_ROOT='data/clevr/CLEVR_v1.0/json/full/'
# DATA_ROOT='data/clevr/CLEVR_v1.0/json/v3/'
# python util/fix_caption_group_problem.py\
#   --dialog_file=${DATA_ROOT}"clevr_train_raw_70k.json" \
#   --template_root="templates/"

# Fix the numerical inconsistency with annotations.
# DATA_ROOT="data/clevr/CLEVR_v1.0/json/v3/"
# for SPLIT in "train" "test"; do
#   echo "Fixing numerical key issues for: "$SPLIT
#   python util/fix_numerical_key_problem.py \
#     --annotation_file=${DATA_ROOT?}"clevr_dialog_annotations_"$SPLIT".json"
# done 

#python identifyCorefChains.py
#python computeInfoGain.py
#python util/convertImageFeatures.py
#python util/convertJSON.py
