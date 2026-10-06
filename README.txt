Replace:
ml\model.py
ml\train.py
app\inference.py

Then run:
python ml\train.py --manifest "D:\SkinCancerData\combined\combined_manifest.csv" --epochs 3 --batch-size 16 --workers 0 --output-dir "D:\SkinCancerData\models"

After training:
python run.py
