"""Pipeline regression tests use artificial images, never medical accuracy claims."""
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'ml'))
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import pandas as pd
import torch
from PIL import Image
from model import create_model
from train import split_df, evaluate, tfms
from dataset import ManifestDataset, CLASS_NAMES
from prepare_combined import build_pad_manifest, build_isic_manifest
from torch.utils.data import DataLoader
from app import inference

torch.set_num_threads(2)

class WorkflowTests(unittest.TestCase):
    def test_training_checkpoint_loads_and_predicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'model.pt'
            model = create_model(pretrained=False, freeze_backbone=False)
            torch.save(dict(model_state=model.state_dict(), class_names=CLASS_NAMES,
                            architecture='mobilenet_v3_small', image_size=64,
                            validation_report={'value': 0.2}), checkpoint)
            with patch.object(inference, 'CHECKPOINT', checkpoint):
                predictor = inference.Predictor()
            raw = io.BytesIO()
            Image.new('RGB', (64, 64), 'brown').save(raw, format='PNG')
            result = predictor.predict_bytes(raw.getvalue())
            self.assertTrue(predictor.ready)
            self.assertIn(result['label'], CLASS_NAMES)
            self.assertEqual(len(result['top_predictions']), 3)
            self.assertTrue(0 <= result['confidence'] <= 100)

    def test_patient_split_no_overlap(self):
        rows = [dict(label=CLASS_NAMES[i % 6], source='PAD-UFES-20',
                     patient_id=f'PAD::{i // 2}', original_id=str(i)) for i in range(60)]
        train, val = split_df(pd.DataFrame(rows))
        self.assertFalse(set(train.patient_id) & set(val.patient_id))
        self.assertEqual(len(train) + len(val), 60)

    def test_recursive_manifest_mapping_and_missing_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); nested = root / 'imgs_part_1'; nested.mkdir()
            Image.new('RGB', (64,64)).save(nested / 'PAD_1.png')
            Image.new('RGB', (64,64)).save(root / 'ISIC_1.jpg')
            pd.DataFrame([dict(img_id='PAD_1.png', diagnostic='NEV', patient_id=7)]).to_csv(root/'pad.csv',index=False)
            pd.DataFrame([dict(image='ISIC_1', NV=1, MEL=0, DF=0),dict(image='excluded', NV=0, MEL=0, DF=1)]).to_csv(root/'isic.csv',index=False)
            pad=build_pad_manifest(root/'pad.csv',root)
            isic=build_isic_manifest(root/'isic.csv',root)
            self.assertEqual(pad.iloc[0].patient_id, 'PAD::7')
            self.assertEqual(isic.iloc[0].label, 'NEV')
            self.assertEqual(len(isic),1)
            (nested/'PAD_1.png').unlink()
            with self.assertRaises(ValueError): build_pad_manifest(root/'pad.csv',root)

    def test_isic_metadata_uses_lesions_and_rejects_dummy_patients(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            Image.new('RGB',(64,64)).save(root/'ISIC_1.jpg')
            pd.DataFrame([dict(image='ISIC_1', NV=1)]).to_csv(root/'gt.csv',index=False)
            pd.DataFrame([dict(isic_id='ISIC_1',patient_id='dummy_0',lesion_id='lesion_7')]).to_csv(root/'meta.csv',index=False)
            manifest=build_isic_manifest(root/'gt.csv',root,root/'meta.csv')
            self.assertEqual(manifest.iloc[0].patient_id,'')
            self.assertEqual(manifest.iloc[0].lesion_id,'lesion_7')
            self.assertEqual(manifest.iloc[0].grouping_level,'lesion')
            self.assertEqual(len(manifest.iloc[0].image_sha256),64)

    def test_conflicting_isic_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pd.DataFrame([dict(image='ISIC_1', NV=1, DF=1)]).to_csv(root/'gt.csv',index=False)
            with self.assertRaisesRegex(ValueError,'one-hot'):
                build_isic_manifest(root/'gt.csv',root)

    def test_evaluation_handles_absent_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            image=Path(tmp)/'one.png'; Image.new('RGB',(64,64)).save(image)
            ds=ManifestDataset(pd.DataFrame([dict(image_path=str(image),label='NEV')]),tfms(False,64))
            loss,score,report=evaluate(create_model(pretrained=False),DataLoader(ds,batch_size=1),torch.device('cpu'))
            self.assertIn('MEL',report)
            self.assertTrue(loss >= 0)

if __name__ == '__main__': unittest.main()
