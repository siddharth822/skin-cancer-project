import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import HTTPException
from starlette.requests import Request
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import auth
from app.guidance import stage_info


def request(token='',csrf='test',origin='http://localhost'):
    return Request({'type':'http','headers':[(b'cookie',f'skinsight_session={token}; skinsight_csrf={csrf}'.encode()),(b'host',b'localhost'),(b'origin',origin.encode())]})


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.patch=patch.object(auth,'DB_PATH',Path(self.temp.name)/'accounts.sqlite3');self.patch.start()
        auth.register('Teammate','a-unique-password')

    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def test_hashed_password_persistent_login_expiry_and_logout(self):
        with auth.connect() as db:
            row=db.execute('SELECT * FROM users').fetchone()
            self.assertNotEqual(row['password_hash'],'a-unique-password')
            self.assertEqual(row['username'],'teammate')
        token=auth.login('TEAMMATE','a-unique-password','local')
        self.assertEqual(auth.require_user(request(token)),'teammate')
        auth.logout(request(token));self.assertIsNone(auth.current_user(request(token)))
        token=auth.login('teammate','a-unique-password','local')
        with patch.object(auth.time,'time',return_value=time.time()+auth.SESSION_SECONDS+1):
            self.assertIsNone(auth.current_user(request(token)))

    def test_bad_credentials_rate_limit_and_duplicate(self):
        with self.assertRaises(ValueError):auth.register('TEAMMATE','another-password')
        for _ in range(10):
            with self.assertRaisesRegex(ValueError,'Incorrect'):auth.login('teammate','wrong-password','bad')
        with self.assertRaisesRegex(ValueError,'Too many'):auth.login('teammate','a-unique-password','bad')
        with self.assertRaises(ValueError):auth.register('short','short')

    def test_csrf_checks_token_and_origin(self):
        auth.check_csrf(request(),'test')
        with self.assertRaises(HTTPException):auth.check_csrf(request(),'wrong')
        with self.assertRaises(HTTPException):auth.check_csrf(request(origin='https://attacker.example'),'test')

    def test_stage_is_never_assigned_from_prediction(self):
        self.assertIn('Not determinable',stage_info('MEL')['status'])
        self.assertTrue(stage_info('MEL')['education'])
        for label in ['BCC','SCC']:self.assertIn('Not determinable',stage_info(label)['status'])
        self.assertIn('Not applicable',stage_info('NEV')['status'])

if __name__=='__main__':unittest.main()
