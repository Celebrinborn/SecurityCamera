from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io
import logging
import os
import smtplib
from typing import List, Optional
import unicodedata

from email.mime.image import MIMEImage

from senders.message_sender import Sender

# remove later
from pathlib import Path

logger = logging.getLogger(__name__)
def load_image_for_inline(path:Path) -> MIMEImage:
        import cv2
        # Load an image using OpenCV

        image = cv2.imread(path.as_posix())

        # Encode the image to JPEG format in memory
        # jpg vs png
        if path.suffix == '.jpg':
            success, encoded_image = cv2.imencode('.jpg', image)
        else :
            success, encoded_image = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Could not encode image")

        # Return the encoded image as MIMEImage
        return MIMEImage(encoded_image.tobytes())


class SMTPSender(Sender):
    @staticmethod
    def Send(subject:str, content:str, attachments:Optional[List[io.BytesIO]]):
        """
        Sends an email with Gmail using credentials retrieved from the system.

        :param subject: The subject line of the email.
        :param body: The HTML body of the email.
        :param inline_images: List of BytesIO objects to embed in the email.
        :return: True if the email was sent successfully, False otherwise.
        """
        logger.debug('pulling creds')
        try:
            gmail_username = os.environ['GMAIL_USERNAME']
            gmail_password = os.environ['GMAIL_PASSWORD']
            gmail_to = os.environ['GMAIL_TO']
        except ValueError as e:
            logger.exception(f'unable to retrieve credential {e}')
            return False

        # Create the email message with HTML body
        message = MIMEMultipart('related')
        message['Subject'] = subject
        message['From'] = gmail_username
        message['To'] = gmail_to

        body_part = MIMEText(content, 'html')
        message.attach(body_part)

        # Embed images
        for i, img_bytes in enumerate(attachments):
            img = MIMEImage(img_bytes.read())
            img.add_header('Content-ID', f'<image{i}>')
            img.add_header('Content-Disposition', 'inline')
            message.attach(img)

        # Convert the message to string format
        raw_message = message.as_string()

        try:
            # Log in to the Gmail account and send the message
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                logger.debug('logging into gmail')
                smtp.login(gmail_username, gmail_password)
                logger.debug('sending gmail')
                smtp.sendmail(gmail_username, gmail_to, raw_message)

            logger.info(f"Email sent to {gmail_to} with subject: {subject}")
            return True
        except Exception as e:
            logger.exception(f"Failed to send email with subject: {subject}. Error message: {str(e)}")
            return False

if __name__ == '__main__':
    import datetime
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            color: #333;
        }}
        .container {{
            width: 80%;
            margin: 20px auto;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 7px;
            background-color: #f9f9f9;
        }}
        h1 {{
            color: #444;
        }}
        .alert-image {{
            width: 100%;
            max-width: 600px;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Security Alert: Person Detected</h1>
        <p>A person has been detected by your security camera. Please review the image and take appropriate action.</p>
        <img src="cid:image0" alt="Security Alert Image" class="alert-image">
        <p>Time of Detection: {datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")}</p>
        <p>If you recognize this activity as normal, no further action is needed. If not, please check your premises or contact authorities.</p>
    </div>
</body>
</html>
"""

    from log_config import configure_logging
    configure_logging()
    logger = logging.getLogger(__name__)
    import dotenv
    dotenv.load_dotenv('.env')

    image_file_path = Path('data', 'Capture.PNG')
    if not image_file_path.exists():
        # print current working directory
        print(os.getcwd())
        # print files and directories in current directory
        print(os.listdir())
        raise FileNotFoundError(f"image file not found at {image_file_path}")
    inline_image = load_image_for_inline(image_file_path)
    sender = SMTPSender()
    sender.Send('Person Detected at Backdoor Camera', html_template, [inline_image])