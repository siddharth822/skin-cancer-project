"""Local accounts with salted password hashes and revocable server-side sessions."""
from contextlib import contextmanager
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path
from urllib.parse import urlsplit
from fastapi import HTTPException, Request

DB_PATH = Path(os.environ.get('SKINSIGHT_AUTH_DB', str(Path(__file__).resolve().parents[1] / 'data/accounts.sqlite3')))
SECURE_COOKIE = os.environ.get('SKINSIGHT_SECURE_COOKIES') == '1'
SESSION_SECONDS = 8 * 60 * 60


@contextmanager
def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, username TEXT UNIQUE NOT NULL, salt TEXT NOT NULL, password_hash TEXT NOT NULL)')
    columns={row[1] for row in db.execute('PRAGMA table_info(users)')}
    for name,definition in [('email','TEXT'),('email_verified','INTEGER NOT NULL DEFAULT 0'),('verification_hash','TEXT'),('verification_expires','REAL'),('verification_attempts','INTEGER NOT NULL DEFAULT 0')]:
        if name not in columns:db.execute(f'ALTER TABLE users ADD COLUMN {name} {definition}')
    db.execute('CREATE UNIQUE INDEX IF NOT EXISTS users_email_unique ON users(email) WHERE email IS NOT NULL')
    db.execute('CREATE TABLE IF NOT EXISTS reports (id TEXT PRIMARY KEY, username TEXT NOT NULL, payload TEXT NOT NULL, email_status TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires REAL NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS attempts (address TEXT PRIMARY KEY, failures INTEGER NOT NULL, started REAL NOT NULL)')
    db.commit()
    try:
        with db:
            yield db
    finally:
        db.close()


def password_hash(password, salt):
    return hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 600000).hex()


def register(username, password, email=None):
    email=validate_email(email) if email else None
    username = username.strip().lower()
    if not re.fullmatch(r'[a-z0-9_]{3,32}', username):
        raise ValueError('Username must have 3–32 letters, numbers or underscores.')
    if not 12 <= len(password) <= 128:
        raise ValueError('Use a password with 12–128 characters.')
    salt = secrets.token_hex(16)
    hashed = password_hash(password, salt)
    try:
        with connect() as db:
            db.execute('INSERT INTO users(username,salt,password_hash,email) VALUES(?,?,?,?)', (username,salt,hashed,email))
    except sqlite3.IntegrityError:
        raise ValueError('That username or email is already registered.')


def login(username, password, address):
    now = time.time()
    with connect() as db:
        attempt = db.execute('SELECT * FROM attempts WHERE address=?',(address,)).fetchone()
        if attempt and now - attempt['started'] < 900 and attempt['failures'] >= 10:
            raise ValueError('Too many failed attempts. Try again in 15 minutes.')
        user = db.execute('SELECT * FROM users WHERE username=? OR email=?',(username.strip().lower(),username.strip().lower())).fetchone()
        valid = hmac.compare_digest(password_hash(password[:129],user['salt'] if user else '00'*16),user['password_hash'] if user else '0'*64)
        if not user or not valid or len(password)>128:
            failures = attempt['failures']+1 if attempt and now-attempt['started']<900 else 1
            started = attempt['started'] if attempt and now-attempt['started']<900 else now
            db.execute('INSERT OR REPLACE INTO attempts VALUES(?,?,?)',(address,failures,started))
            raise_after = True
        else:
            db.execute('DELETE FROM attempts WHERE address=?',(address,))
            db.execute('DELETE FROM sessions WHERE expires < ?',(now,))
            token = secrets.token_urlsafe(32)
            db.execute('INSERT INTO sessions VALUES(?,?,?)',(hashlib.sha256(token.encode()).hexdigest(),user['id'],now+SESSION_SECONDS))
            raise_after = False
    if raise_after:
        raise ValueError('Incorrect username or password.')
    return token


def current_user(request: Request):
    token = request.cookies.get('skinsight_session')
    if not token:
        return None
    with connect() as db:
        user = db.execute('SELECT users.username FROM sessions JOIN users ON users.id=sessions.user_id WHERE token_hash=? AND expires>?', (hashlib.sha256(token.encode()).hexdigest(),time.time())).fetchone()
    return user['username'] if user else None


def require_user(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=401,detail='Please log in before analyzing an image.')
    return user


def check_csrf(request, token):
    expected = request.cookies.get('skinsight_csrf','')
    origin = request.headers.get('origin')
    if (origin and urlsplit(origin).netloc != request.headers.get('host')) or not expected or not token or not hmac.compare_digest(expected,token):
        raise HTTPException(status_code=403,detail='Session verification failed. Refresh the page and try again.')


def logout(request):
    token = request.cookies.get('skinsight_session','')
    with connect() as db:
        db.execute('DELETE FROM sessions WHERE token_hash=?',(hashlib.sha256(token.encode()).hexdigest(),))


def validate_email(email):
    email=email.strip().lower()
    if len(email)>254 or not re.fullmatch(r"[a-z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?\.[a-z]{2,}",email):
        raise ValueError('Enter a valid email address.')
    return email


def email_profile(username):
    with connect() as db:
        row=db.execute('SELECT email,email_verified FROM users WHERE username=?',(username,)).fetchone()
    return dict(row)


def set_email(username,email):
    email=validate_email(email)
    try:
        with connect() as db:
            db.execute('UPDATE users SET email=?,email_verified=0,verification_hash=NULL,verification_expires=NULL,verification_attempts=0 WHERE username=?',(email,username))
    except sqlite3.IntegrityError:raise ValueError('That email is already registered.')


def verification_code(username):
    with connect() as db:
        row=db.execute('SELECT verification_expires FROM users WHERE username=?',(username,)).fetchone()
        if row['verification_expires'] and row['verification_expires']-time.time()>540:
            raise ValueError('Wait one minute before requesting another code.')
        code=f'{secrets.randbelow(1000000):06d}'
        db.execute('UPDATE users SET verification_hash=?,verification_expires=?,verification_attempts=0 WHERE username=?',(hashlib.sha256(code.encode()).hexdigest(),time.time()+600,username))
    return code


def verify_email(username,code):
    with connect() as db:
        row=db.execute('SELECT * FROM users WHERE username=?',(username,)).fetchone()
        valid=(row['verification_hash'] and row['verification_expires']>time.time() and row['verification_attempts']<5 and hmac.compare_digest(row['verification_hash'],hashlib.sha256(code.strip().encode()).hexdigest()))
        if valid:db.execute('UPDATE users SET email_verified=1,verification_hash=NULL,verification_expires=NULL WHERE username=?',(username,))
        else:db.execute('UPDATE users SET verification_attempts=verification_attempts+1 WHERE username=?',(username,))
    if not valid:raise ValueError('Code is incorrect or expired. Request a new code.')
