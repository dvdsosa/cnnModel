"""
Notification module to send an email to the user when the script is done.

Example:
from email_me import notifyUser

notify_me = notifyUser()
notify_me({'elapsed_days': 1, 'elapsed_hours': 15})

"""
import smtplib
from email.message import EmailMessage

class NotifyUser(object):
    def __init__(self) -> None:
        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587  # SMTP server port number
        self.username = 'soledadd1974@gmail.com'  # email address
        self.password = 'twok dskr aqdb abdt'  # email password

        self.msg = EmailMessage()
        self.msg['Subject'] = 'Script Execution Completed'
        self.msg['From'] = 'soledadd1974@gmail.com'  # user's email address, as the sender
        self.msg['To'] = 'dvdsosa@gmail.com'  # destination email address

    def __call__(self, tiempo):
        try:
            self.msg.set_content(
                f'Script execution completed!\n'
                f'Elapsed time: {tiempo["elapsed_days"]} days {tiempo["elapsed_hours"]} hours\n'
            )

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.username, self.password)
                server.send_message(self.msg)
                print("Notification sent to " + self.msg['To'] + " successfully!")
        except smtplib.SMTPException as e:
            print("SMTP error occurred: " + str(e))
        except Exception as e:
            print("Error occurred: " + str(e))