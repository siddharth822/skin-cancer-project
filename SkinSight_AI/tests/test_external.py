import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
from external_data import audit_external
from calibrate import fit_temperature

class ExternalTests(unittest.TestCase):
    def fixture(self, root):
        paths=[]
        for i in range(4):
            image=root/f'{i}.png';Image.new('RGB',(64,64),(i*30,80,90)).save(image)
            paths.append(dict(image_path=str(image),label='NEV',source='source',original_id=str(i),
                              patient_id=f'p{i}',lesion_id=f'l{i}',image_sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
        for name,row in zip(['train','validation','test'],paths[:3]):
            pd.DataFrame([row]).to_csv(root/f'{name}_split.csv',index=False)
        pd.DataFrame([paths[3]]).to_csv(root/'external.csv',index=False)
        return paths

    def test_accepts_novel_images_and_rejects_old_test_and_calibration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=self.fixture(root)
            _,audit=audit_external(root/'external.csv',root)
            self.assertFalse(audit['known_group_overlap'])
            with self.assertRaisesRegex(ValueError,'overlaps'):
                audit_external(root/'test_split.csv',root)
            pd.DataFrame([rows[3]]).to_csv(root/'calibration.csv',index=False)
            with self.assertRaisesRegex(ValueError,'overlaps'):
                audit_external(root/'external.csv',root,root/'calibration.csv')

    def test_rejects_changed_image_and_patient_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);rows=self.fixture(root)
            rows[3]['patient_id']=rows[0]['patient_id']
            pd.DataFrame([rows[3]]).to_csv(root/'external.csv',index=False)
            with self.assertRaisesRegex(ValueError,'overlaps'):
                audit_external(root/'external.csv',root)
            Image.new('RGB',(64,64),'red').save(root/'3.png')
            with self.assertRaisesRegex(ValueError,'checksum'):
                audit_external(root/'external.csv',root)

    def test_calibration_cli_and_external_test_overlap_guard(self):
        import subprocess,os,json
        from model import create_model,CLASS_NAMES
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);self.fixture(root)
            checkpoint=root/'model.pt'
            torch.save(dict(model_state=create_model(pretrained=False).state_dict(),
                            architecture='mobilenet_v3_small',class_names=CLASS_NAMES,
                            image_size=64),checkpoint)
            base=Path(__file__).resolve().parents[1]/'ml'
            env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NO_ALBUMENTATIONS_UPDATE='1')
            calibrated=root/'calibrated.pt'
            result=subprocess.run([sys.executable,str(base/'calibrate.py'),
                '--manifest',str(root/'external.csv'),'--checkpoint',str(checkpoint),
                '--reference-dir',str(root),'--output-checkpoint',str(calibrated)],
                env=env,text=True,capture_output=True,timeout=60)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertTrue(calibrated.with_suffix('.calibration.csv').exists())
            from app import inference
            from unittest.mock import patch
            with patch.object(inference,'CHECKPOINT',calibrated):
                predictor=inference.Predictor()
            self.assertGreater(predictor.temperature,0)
            result=subprocess.run([sys.executable,str(base/'evaluate_external.py'),
                '--manifest',str(root/'external.csv'),'--checkpoint',str(calibrated),
                '--reference-dir',str(root),'--output',str(root/'report.json')],
                env=env,text=True,capture_output=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('overlaps',result.stderr)
            self.assertFalse((root/'report.json').exists())

    def test_temperature_reduces_nll_without_changing_classes(self):
        logits=torch.tensor([[9.,0.],[9.,0.],[0.,9.],[0.,9.]])
        labels=torch.tensor([0,1,1,0])
        temperature,before,after=fit_temperature(logits,labels)
        self.assertGreater(temperature,1)
        self.assertLess(after,before)
        self.assertTrue(torch.equal(logits.argmax(1),(logits/temperature).argmax(1)))

if __name__ == '__main__':unittest.main()
