"""A CLI run proves warm-start training never decodes old test images in skip mode."""
import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'ml'))
from model import create_model,CLASS_NAMES
from splits import split_three_way

class FineTuneTests(unittest.TestCase):
    def test_warm_start_and_skip_test(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=[]
            for label_index,label in enumerate(CLASS_NAMES):
                for i in range(15):
                    image=root/f'{label}_{i}.png'
                    Image.new('RGB',(64,64),(label_index*40,i*12,90)).save(image)
                    rows.append(dict(image_path=str(image),label=label,source='PAD-UFES-20',
                                     patient_id=f'{label}_{i}',original_id=f'{label}_{i}',
                                     image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
            frame=pd.DataFrame(rows);_,_,test=split_three_way(frame)
            # Keep metadata and hashes but remove test files; decoding them would fail.
            for path in test.image_path:Path(path).unlink()
            frame.to_csv(root/'manifest.csv',index=False)
            checkpoint=root/'initial.pt'
            torch.save(dict(model_state=create_model(pretrained=False).state_dict(),
                            architecture='mobilenet_v3_small',class_names=CLASS_NAMES,
                            image_size=64,training_mode='synthetic'),checkpoint)
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NO_ALBUMENTATIONS_UPDATE='1')
            result=subprocess.run([sys.executable,str(Path(__file__).resolve().parents[1]/'ml/train.py'),
                '--manifest',str(root/'manifest.csv'),'--init-checkpoint',str(checkpoint),
                '--fine-tune','--skip-test','--train-source','PAD-UFES-20','--selection-source','PAD-UFES-20',
                '--epochs','1','--size','64','--batch-size','8','--workers','0',
                '--device','cpu','--output-dir',str(root/'output')],env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue((root/'output/metrics.json').exists())
            self.assertFalse((root/'output/test_metrics.json').exists())
            import json
            metrics=json.loads((root/'output/metrics.json').read_text())
            self.assertFalse(metrics['test_evaluated'])
            self.assertEqual(metrics['fitting_source'],'PAD-UFES-20')
            self.assertEqual(metrics['fitting_records'],len(pd.read_csv(root/'output/train_split.csv')))
            self.assertEqual(metrics['history'][0]['epoch'],0)

if __name__ == '__main__':unittest.main()
