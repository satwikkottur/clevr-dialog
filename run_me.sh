# Generate the CLEVR-Dialog dataset.
DATA_ROOT='data/CLEVR_v1.0/'
python -u generate_dataset.py \
 --scene_path=${DATA_ROOT}"scenes/CLEVR_train_scenes.json" \
 --num_beams=100 \
 --num_workers=1 \
 --save_path=${DATA_ROOT}"clevr_dialog_train_raw.json" \
 --num_images=10

# python -u generate_dataset.py \
#   --scene_path=${DATA_ROOT}"scenes/CLEVR_val_scenes.json" \
#   --num_beams=100 \
#   --num_workers=12 \
#   --save_path=${DATA_ROOT}"clevr_dialog_val_raw.json"
