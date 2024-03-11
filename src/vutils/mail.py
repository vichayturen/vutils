from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from typing import List, Optional, Union


class MailSender:
    def __init__(self, host: str, port: int, email: str, password: str):
        self.server = smtplib.SMTP_SSL(host, port)
        self.email = email
        self.password = password

    def send(self, title: str, msg: str, target_emails: Optional[Union[str, List[str]]]=None):
        self.server.login(user=self.email, password=self.password)
        if target_emails is None:
            target_emails = self.email
        email_body = MIMEMultipart('mixed')
        email_body['Subject'] = title
        email_body['From'] = self.email
        email_body['To'] = target_emails
        text_plain = MIMEText(msg, 'plain', 'utf-8')
        email_body.attach(text_plain)
        self.server.sendmail(self.email, target_emails, email_body.as_string())
        self.server.quit()
