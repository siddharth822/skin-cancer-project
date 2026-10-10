import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import auth,reporting
from app.guidance import GUIDANCE,stage_info

class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.dbpatch=patch.object(auth,'DB_PATH',Path(self.temp.name)/'accounts.sqlite3');self.dbpatch.start()
        auth.register('owner','a-long-test-password');auth.register('other','a-long-test-password')
    def tearDown(self):self.dbpatch.stop();self.temp.cleanup()
    def result(self,label,score):
        return {'label':label,'confidence':score,'top_predictions':[{'label':label,'probability':score}], 'guidance':GUIDANCE[label],'stage':stage_info(label),'medical_disclaimer':'Educational only.'}
    def test_real_aggregates_exclude_other_accounts(self):
        a=reporting.save_report('owner',self.result('MEL',80))
        b=reporting.save_report('owner',self.result('NEV',40))
        c=reporting.save_report('other',self.result('SCC',100))
        data=reporting.dashboard_data('owner')
        self.assertEqual(data['stats']['total'],2);self.assertEqual(data['stats']['average'],60)
        self.assertEqual(data['stats']['cancer_classes'],1);self.assertEqual(data['stats']['today'],2)
        self.assertEqual(sum(m['count'] for m in data['months']),2)
        self.assertEqual({r['report_id'] for r in data['reports']},{a,b})
        self.assertNotIn(c,{r['report_id'] for r in data['reports']})
        self.assertEqual(sum(r['count'] for r in data['frequencies']),2)
    def test_empty_account_has_no_example_results(self):
        data=reporting.dashboard_data('owner')
        self.assertEqual(data['reports'],[]);self.assertEqual(data['stats']['total'],0)
        self.assertEqual(len(data['months']),6)
        self.assertTrue(all(month['height']==0 for month in data['months']))

if __name__=='__main__':unittest.main()
