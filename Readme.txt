python run_pipeline.py --list-models                                    # see model choices
python run_pipeline.py --image_dir Mechanical_input --model claude-sonnet-5
python run_pipeline.py --image_dir Mechanical_input --skip-existing     # resume; reuse finished OCR/tables
python run_pipeline.py --image_dir Mechanical_input --ocr-timeout 600   # 10 min/image instead of 5