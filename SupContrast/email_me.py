"""
Notification module to send an email to the user when the script is done.

Example:
from email_me import notifyUser

notify_me = notifyUser()
notify_me({'elapsed_days': 1, 'elapsed_hours': 15})

"""
import smtplib
import os
from email.message import EmailMessage
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class NotifyUser(object):
    def __init__(self) -> None:
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587  # SMTP server port number
        self.username = 'soledadd1974@gmail.com'  # email address
        self.password = os.getenv('EMAIL_PASSWORD')  # email password from environment variable
        
        if not self.password:
            raise ValueError("EMAIL_PASSWORD environment variable not found. Please check your .env file.")

        self.msg = EmailMessage()
        self.msg['Subject'] = 'Script Execution Completed'
        self.msg['From'] = 'soledadd1974@gmail.com'  # user's email address, as the sender
        self.msg['To'] = 'dvdsosa@gmail.com'  # destination email address

    def __call__(self, message: str) -> None:
        try:
            self.msg.set_content(message)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(self.msg)
                print("Notification sent to " + self.msg['To'] + " successfully!")
        except smtplib.SMTPException as e:
            print("SMTP error occurred: " + str(e))
        except Exception as e:
            print("Error occurred: " + str(e))