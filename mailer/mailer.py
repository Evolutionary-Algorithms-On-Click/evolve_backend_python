import smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from os.path import basename

class Message:
    def __init__(
        self,
        receiver_email,
        subject,
        textBody=None,
        htmlBody=None,
        attachmentPath=None,
        sender_email=None,
    ):
        self.message = MIMEMultipart("alternative")
        self.message["Subject"] = subject
        self.message["From"] = sender_email
        self.message["To"] = receiver_email
        self.plainText = textBody
        self.htmlBody = htmlBody
        self.attachmentPath = attachmentPath

        if self.plainText:
            self.addPlainText()
        if self.htmlBody:
            self.addHtml()
        if self.attachmentPath:
            self.addAttachment()

    def addPlainText(self):
        plainText = MIMEText(self.plainText, "plain")
        self.message.attach(plainText)

    def addHtml(self):
        htmlBody = MIMEText(self.htmlBody, "html")
        self.message.attach(htmlBody)

    def addAttachment(self):
        filename = self.attachmentPath
        basename_ = basename(filename)
        with open(filename, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {basename_}",
        )
        self.message.attach(part)

class Mailer:
    def __init__(self, host, port, sender_email, password):
        self.host = host
        self.port = port
        self.sender_email = sender_email
        self.password = password
        self.context = ssl.create_default_context()

        if port == '587':
            self.server = smtplib.SMTP(self.host, self.port)
            self.server.starttls(context=self.context)
        elif port == '465':
            self.server = smtplib.SMTP_SSL(self.host, self.port, context=self.context)
        else:
            raise ValueError("Unsupported port. Use 465 for SSL or 587 for STARTTLS.")

        self.server.login(self.sender_email, self.password)

    def sendMail(self, receiver_email, message):
        self.server.sendmail(
            self.sender_email, receiver_email, message.message.as_string()
        )

    def close(self):
        self.server.quit()
