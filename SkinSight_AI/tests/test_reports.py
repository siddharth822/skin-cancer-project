import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import auth,reporting
from app.guidance import GUIDANCE,stage_info

class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.dbpatch=patch.object(auth,'DB_PATH',Path(self.temp.name)/'accounts.sqlite3');self.dbpatch.start()
        auth.register('owner','long-test-password','owner@example.com');auth.register('other','other-test-password','other@example.com')
        self.result={'label':'MEL','confidence':80.,'top_predictions':[{'label':'MEL','probability':80.}], 'guidance':GUIDANCE['MEL'],'stage':stage_info('MEL'),'medical_disclaimer':'Educational screening, not a medical diagnosis.'}
    def tearDown(self):self.dbpatch.stop();self.temp.cleanup()

    def test_private_pdf_and_delivery_requires_verified_recipient(self):
        report_id=reporting.save_report('owner',self.result)
        self.assertIsNone(reporting.get_report('other',report_id))
        data=reporting.get_report('owner',report_id);pdf=reporting.make_pdf(reporting.report_text(data))
        self.assertTrue(pdf.startswith(b'%PDF-1.4'));self.assertIn(b'Doctor guidance',pdf);self.assertIn(b'Not determinable',pdf)
        with patch.object(reporting,'send_mail') as sender:
            reporting.deliver_report('owner',report_id);sender.assert_not_called()
        self.assertEqual(reporting.get_report('owner',report_id)['email_status'],'verification_required')
        code=auth.verification_code('owner');auth.verify_email('owner',code)
        with patch.object(reporting,'send_mail',return_value='accepted') as sender:
            reporting.deliver_report('owner',report_id)
            args=sender.call_args.args;self.assertEqual(args[0],'owner@example.com');self.assertIn('dermatologist',args[2]);self.assertTrue(args[3].startswith(b'%PDF'))
        self.assertEqual(reporting.get_report('owner',report_id)['email_status'],'accepted')

    def test_verification_expiry_wrong_code_and_address_reset(self):
        code=auth.verification_code('owner')
        with self.assertRaises(ValueError):auth.verify_email('owner','wrong')
        auth.verify_email('owner',code);self.assertTrue(auth.email_profile('owner')['email_verified'])
        auth.set_email('owner','new@example.com');self.assertFalse(auth.email_profile('owner')['email_verified'])
        code=auth.verification_code('owner')
        with patch.object(auth.time,'time',return_value=auth.time.time()+601):
            with self.assertRaises(ValueError):auth.verify_email('owner',code)
        with self.assertRaises(ValueError):auth.set_email('owner','evil\r\nBcc: other@example.com')

    def test_gmail_tls_attachment_and_failure(self):
        env={'SKINSIGHT_SMTP_HOST':'smtp.gmail.com','SKINSIGHT_SMTP_PORT':'465','SKINSIGHT_SMTP_USER':'sender@example.com','SKINSIGHT_SMTP_PASSWORD':'fake-test-only','SKINSIGHT_MAIL_FROM':'sender@example.com'}
        with patch.dict(os.environ,env),patch.object(reporting.smtplib,'SMTP_SSL') as smtp:
            smtp.return_value.__enter__.return_value.send_message.return_value={}
            self.assertEqual(reporting.send_mail('owner@example.com','Report','Doctor guidance',b'%PDF-1.4'),'accepted')
            server=smtp.return_value.__enter__.return_value
            server.login.assert_called_once();message=server.send_message.call_args.args[0]
            self.assertEqual(message['To'],'owner@example.com');self.assertEqual(len(list(message.iter_attachments())),1)
            self.assertEqual(smtp.call_args.kwargs['timeout'],8)
            self.assertTrue(smtp.call_args.kwargs['context'].check_hostname)
        with patch.dict(os.environ,env),patch.object(reporting.smtplib,'SMTP_SSL',side_effect=OSError('offline')):
            self.assertEqual(reporting.send_mail('owner@example.com','Report','test'),'failed')
        with patch.dict(os.environ,env),patch.object(reporting.smtplib,'SMTP_SSL',side_effect=reporting.smtplib.SMTPAuthenticationError(535,b'test error')):
            self.assertEqual(reporting.send_mail('owner@example.com','Report','test'),'auth_failed')
            self.assertIn('App Password',reporting.delivery_message('auth_failed'))
        with patch.dict(os.environ,{},clear=True):self.assertEqual(reporting.send_mail('owner@example.com','Report','test'),'not_configured')

    def test_migrates_existing_account_database(self):
        path=Path(self.temp.name)/'legacy.sqlite3'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, password_hash TEXT NOT NULL)')
        with patch.object(auth,'DB_PATH',path):
            auth.register('legacy','long-test-password','legacy@example.com')
            self.assertEqual(auth.email_profile('legacy')['email'],'legacy@example.com')

if __name__=='__main__':unittest.main()
