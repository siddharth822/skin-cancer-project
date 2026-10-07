"""Run the actual HTTP API with a temporary artificial checkpoint."""
import io
import http.cookiejar
import re
import urllib.parse
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ml'))
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import numpy as np
import torch
import uvicorn
from PIL import Image
from model import create_model, CLASS_NAMES
from app import inference, main, auth

class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory()
        cls.auth_patch=patch.object(auth,'DB_PATH',Path(cls.temp.name)/'accounts.sqlite3');cls.auth_patch.start()
        checkpoint=Path(cls.temp.name)/'test.pt'
        torch.save(dict(model_state=create_model(pretrained=False).state_dict(),
                        class_names=CLASS_NAMES,architecture='mobilenet_v3_small',image_size=64),checkpoint)
        with patch.object(inference,'CHECKPOINT',checkpoint):
            predictor=inference.Predictor()
        cls.predictor_patch=patch.object(main,'predictor',predictor);cls.predictor_patch.start()
        cls.sock=socket.socket();cls.sock.bind(('127.0.0.1',0))
        cls.url=f'http://127.0.0.1:{cls.sock.getsockname()[1]}'
        cls.server=uvicorn.Server(uvicorn.Config(main.app,log_level='error'))
        cls.thread=threading.Thread(target=cls.server.run,kwargs={'sockets':[cls.sock]},daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started: break
            time.sleep(.02)
        if not cls.server.started: raise RuntimeError('API did not start')
        cls.opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        html=cls.opener.open(cls.url+'/register').read().decode()
        cls.csrf=re.search(r'name="csrf" value="([^"]+)"',html).group(1)
        form=urllib.parse.urlencode(dict(username='testuser',password='unique-test-password',csrf=cls.csrf)).encode()
        cls.opener.open(cls.url+'/register',form).close()
        cls.opener.open(cls.url+'/login',form).close()

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit=True;cls.thread.join(timeout=5)
        cls.sock.close();cls.predictor_patch.stop();cls.auth_patch.stop();cls.temp.cleanup()

    def upload(self,raw,content_type='image/png'):
        boundary='skinsighttest'
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="image.png"\r\nContent-Type: {content_type}\r\n\r\n').encode()+raw+f'\r\n--{boundary}--\r\n'.encode()
        return self.opener.open(urllib.request.Request(self.url+'/api/predict',data=body,headers={'Content-Type':f'multipart/form-data; boundary={boundary}', 'X-CSRF-Token':self.csrf}))

    def test_prediction_and_html(self):
        with self.opener.open(self.url+'/') as response:
            self.assertIn(b'MobileNet',response.read())
        with urllib.request.urlopen(self.url+'/api/health') as response:
            self.assertTrue(json.load(response)['model_ready'])
        raw=io.BytesIO();Image.fromarray(np.random.default_rng(42).integers(50,200,(128,128,3),dtype=np.uint8)).save(raw,format='PNG')
        with self.upload(raw.getvalue()) as response:
            result=json.load(response)
            self.assertIn(result['label'],CLASS_NAMES)
            self.assertIn('guidance',result)
            self.assertIn('stage',result)
            self.assertIn('not a medical diagnosis',result['medical_disclaimer'])

    def test_avatar_rejected_before_inference(self):
        from PIL import ImageDraw
        avatar=Image.new("RGB",(128,128),(128,140,140))
        ImageDraw.Draw(avatar).text((35,45),"SP",fill="white")
        raw=io.BytesIO();avatar.save(raw,format="PNG")
        with patch.object(main.predictor,"predict_bytes") as prediction:
            with self.assertRaises(urllib.error.HTTPError) as error: self.upload(raw.getvalue())
            self.assertEqual(error.exception.code,422)
            self.assertIn("graphic",json.load(error.exception)["detail"])
            prediction.assert_not_called()

    def test_unusable_photos_rejected(self):
        for image in [Image.new("RGB",(32,32),"brown"), Image.new("RGB",(128,128),"black"), Image.new("RGB",(128,128),"white")]:
            with self.subTest(size=image.size):
                raw=io.BytesIO();image.save(raw,format="PNG")
                with patch.object(main.predictor,"predict_bytes") as prediction:
                    with self.assertRaises(urllib.error.HTTPError) as error: self.upload(raw.getvalue())
                    self.assertEqual(error.exception.code,422)
                    prediction.assert_not_called()

    def test_authentication_required(self):
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(self.url+"/api/predict",data=b"image=invalid"))
        self.assertEqual(error.exception.code,401)

    def test_invalid_image_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as error: self.upload(b'invalid')
        self.assertEqual(error.exception.code,400)

    def test_untrained_model_returns_503(self):
        with patch.object(main.predictor,'model',None):
            with self.assertRaises(urllib.error.HTTPError) as error: self.upload(b'invalid')
        self.assertEqual(error.exception.code,503)

if __name__ == '__main__': unittest.main()
