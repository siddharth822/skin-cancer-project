import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
from splits import linked_groups, split_three_way
from evaluation import summarize

class EvaluationTests(unittest.TestCase):
    def test_transitive_patient_lesion_and_duplicate_grouping(self):
        frame = pd.DataFrame([
            dict(source='PAD', original_id='a',patient_id='p1',lesion_id='l1',image_sha256='hashA'),
            dict(source='PAD', original_id='b',patient_id='p1',lesion_id='l2',image_sha256='hashB'),
            dict(source='ISIC',original_id='c',patient_id='',lesion_id='l3',image_sha256='hashB'),
            dict(source='ISIC',original_id='d',patient_id='',lesion_id='l3',image_sha256='hashD'),
            dict(source='ISIC',original_id='e',patient_id='',lesion_id='',image_sha256='unique'),
        ])
        groups = linked_groups(frame)
        self.assertEqual(len(set(groups[:4])),1)
        self.assertNotEqual(groups.iloc[0],groups.iloc[4])

    def test_three_partitions_are_disjoint_and_keep_all_strata(self):
        frame=pd.DataFrame([dict(source=source,label=label,original_id=f'{source}_{label}_{i}',
                                 patient_id=f'{source}_{label}_{i//2}',image_sha256=f'{source}_{label}_{i}')
                            for source in ['PAD','ISIC'] for label in ['MEL','NEV'] for i in range(30)])
        parts=split_three_way(frame)
        for i,part in enumerate(parts):
            self.assertEqual(set(part.label),{'MEL','NEV'})
            self.assertEqual(set(part.source),{'PAD','ISIC'})
            for other in parts[:i]:
                self.assertFalse(set(part._group)&set(other._group))
        self.assertEqual(sum(map(len,parts)),len(frame))
        again=split_three_way(frame)
        self.assertEqual(parts[2].original_id.tolist(),again[2].original_id.tolist())

    def test_specificity_sensitivity_auc_and_confusion_matrix(self):
        probabilities=np.array([[.8,.2,0,0,0,0],[.1,.9,0,0,0,0],[.7,.3,0,0,0,0]])
        report=summarize([0,1,1],probabilities)
        self.assertEqual(report['confusion_matrix'][1][0],1)
        self.assertEqual(report['per_class']['BCC']['sensitivity'],.5)
        self.assertEqual(report['per_class']['ACK']['specificity'],.5)
        self.assertIsNone(report['per_class']['MEL']['sensitivity'])
        self.assertIsNone(report['per_class']['MEL']['roc_auc'])
        self.assertGreaterEqual(report['expected_calibration_error_10_bins'],0)

if __name__ == '__main__': unittest.main()
