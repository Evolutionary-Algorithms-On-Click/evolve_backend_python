from mailer.mailer import Mailer, Message

from dotenv import load_dotenv

load_dotenv()
import os

host = os.getenv("MAILER_HOST", "smtp.gmail.com")

port = os.getenv("MAILER_PORT", 465)

user = os.getenv("MAILER_USER", "<mailer_user>")

password = os.getenv("MAILER_PASS", "<mailer_password>")

mailerInstance = Mailer(host, port, user, password)

class MailMessage(Message):
    def __init__(self, receiver_email, subject, textBody=None, htmlBody=None, attachmentPath=None):
        super().__init__(receiver_email, subject, textBody, htmlBody, attachmentPath, sender_email=user)

