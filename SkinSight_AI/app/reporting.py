"""Owner-only screening reports and TLS SMTP delivery."""
import json
import os
import secrets
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from textwrap import wrap
from . import auth


def smtp_ready():
    return all(os.environ.get(k) for k in ['SKINSIGHT_SMTP_HOST','SKINSIGHT_SMTP_USER','SKINSIGHT_SMTP_PASSWORD','SKINSIGHT_MAIL_FROM'])


def send_mail(recipient, subject, body, attachment=None):
    if not smtp_ready():
        return 'not_configured'
    message=EmailMessage();message['From']=os.environ['SKINSIGHT_MAIL_FROM'];message['To']=recipient;message['Subject']=subject
    message.set_content(body)
    if attachment:message.add_attachment(attachment,maintype='application',subtype='pdf',filename='skinsight-screening-report.pdf')
    try:
        host=os.environ['SKINSIGHT_SMTP_HOST'];port=int(os.environ.get('SKINSIGHT_SMTP_PORT','587'))
        context=ssl.create_default_context()
        if port==465:
            connection=smtplib.SMTP_SSL(host,port,timeout=8,context=context)
        else:
            connection=smtplib.SMTP(host,port,timeout=8);connection.ehlo();connection.starttls(context=context);connection.ehlo()
        with connection as server:
            server.login(os.environ['SKINSIGHT_SMTP_USER'],os.environ['SKINSIGHT_SMTP_PASSWORD'])
            refused=server.send_message(message)
        return 'failed' if refused else 'accepted'
    except smtplib.SMTPAuthenticationError:
        return 'auth_failed'
    except smtplib.SMTPRecipientsRefused:
        return 'recipient_refused'
    except smtplib.SMTPSenderRefused:
        return 'sender_refused'
    except ssl.SSLError:
        return 'tls_failed'
    except (OSError,smtplib.SMTPException,ValueError):
        return 'failed'


def report_text(result):
    return '\n'.join([
        'SkinSight AI - Educational screening report',
        'Report ID: '+result['report_id'], 'Created (UTC): '+result['created_at'],
        '', 'Image-model top class: '+result['guidance']['name'],
        f"Model score: {result['confidence']:.1f}% (not a probability of disease)",
        'Class scores: '+', '.join(f"{p['label']} {p['probability']:.1f}%" for p in result['top_predictions']),
        '', 'Doctor guidance:',result['guidance']['message'],result['guidance']['urgency'],
        'Consult a qualified dermatologist. Bring this report and describe when the lesion appeared and any changes.',
        '', 'Cancer stage: '+result['stage']['status'],result['stage']['explanation'],
        *result['stage'].get('education',[]), '', result['medical_disclaimer'],
        'Smartphone validation balanced accuracy: 56.9%; independent clinical validation is pending.',
    ])


def make_pdf(text):
    """Small portable text-only PDF; no user HTML, uploads or external renderer."""
    lines=[]
    for line in text.splitlines():lines.extend(wrap(line,92) or [''])
    objects=[]
    def add(data):objects.append(data);return len(objects)
    add(b'');add(b'');font=add(b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>');pages=[]
    for offset in range(0,len(lines),47):
        commands=['BT /F1 10 Tf 48 790 Td 15 TL']
        for line in lines[offset:offset+47]:
            safe=line.replace('–','-').replace('—','-').replace('’',"'").encode('ascii','replace').decode().replace('\\','\\\\').replace('(','\\(').replace(')','\\)')
            commands.append('('+safe+') Tj T*')
        commands.append('ET');stream='\n'.join(commands).encode()
        content=add(b'<< /Length '+str(len(stream)).encode()+b' >>\nstream\n'+stream+b'\nendstream')
        page=add(f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>'.encode());pages.append(page)
    objects[0]=b'<< /Type /Catalog /Pages 2 0 R >>';objects[1]=f"<< /Type /Pages /Count {len(pages)} /Kids [{' '.join(str(p)+' 0 R' for p in pages)}] >>".encode()
    output=bytearray(b'%PDF-1.4\n');positions=[0]
    for i,obj in enumerate(objects,1):positions.append(len(output));output.extend(f'{i} 0 obj\n'.encode()+obj+b'\nendobj\n')
    xref=len(output);output.extend(f'xref\n0 {len(objects)+1}\n0000000000 65535 f \n'.encode())
    for p in positions[1:]:output.extend(f'{p:010d} 00000 n \n'.encode())
    output.extend(f'trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode());return bytes(output)


def save_report(username,result):
    result['report_id']=secrets.token_urlsafe(18);result['created_at']=datetime.now(timezone.utc).isoformat()
    with auth.connect() as db:
        db.execute('INSERT INTO reports VALUES(?,?,?,?)',(result['report_id'],username,json.dumps(result),'pending'))
    return result['report_id']


def get_report(username,report_id):
    with auth.connect() as db:
        row=db.execute('SELECT payload,email_status FROM reports WHERE id=? AND username=?',(report_id,username)).fetchone()
    if not row:return None
    data=json.loads(row['payload']);data['email_status']=row['email_status'];return data


def deliver_report(username,report_id):
    result=get_report(username,report_id)
    with auth.connect() as db:user=db.execute('SELECT email,email_verified FROM users WHERE username=?',(username,)).fetchone()
    if not user or not user['email'] or not user['email_verified']:status='verification_required'
    else:status=send_mail(user['email'],'Your SkinSight screening report',report_text(result),make_pdf(report_text(result)))
    with auth.connect() as db:db.execute('UPDATE reports SET email_status=? WHERE id=? AND username=?',(status,report_id,username))


def delivery_message(status):
    return {
        'auth_failed': 'Gmail rejected the sender login. Restart start_with_gmail.bat and enter the Gmail address that created the App Password, followed by that 16-character App Password (not your normal password).',
        'recipient_refused': 'Gmail rejected the recipient address. Check the Report email address.',
        'sender_refused': 'Gmail rejected the sender address. Use the same Gmail address for sender and login.',
        'tls_failed': 'The secure Gmail connection failed. Check your computer date/time and network; TLS verification remains enabled.',
        'not_configured': 'Gmail sending is not configured. Start the app with start_with_gmail.bat.',
        'failed': 'Could not send through Gmail. Check internet access and whether the network permits smtp.gmail.com on port 465, then try again.',
    }.get(status, 'Unable to send the email. Your report can still be downloaded.')
