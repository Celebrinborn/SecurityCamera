import io
import logging
import os
import smtplib
from typing import Optional
import unicodedata

from email.mime.image import MIMEImage

logger = logging.getLogger(__name__)

def _unicode_to_utf8(text):
    normalized_text = unicodedata.normalize('NFKD', text)
    return normalized_text.encode('ascii', 'ignore').decode('utf-8')

def _send_email_with_gmail(subject: str, body: str) -> bool:
    """
    Sends an email with Gmail using credentials retrieved from the system.

    :param subject: The subject line of the email.
    :param body: The body of the email.
    :return: True if the email was sent successfully, False otherwise.
    """
    logger.debug('pulling creds')
    try:
        gmail_username = os.environ.get('GMAIL_USERNAME')
        gmail_password = os.environ.get('GMAIL_PASSWORD')
        gmail_to = os.environ.get('GMAIL_TO')
    except ValueError as e:
        logger.exception(f'unable to retrieve credential {e}')
        return False

    # Create the email message
    raw_message = f"Subject: {subject}\n\n{body}"
    message = _unicode_to_utf8(raw_message)
    logger.info(f'sending message {subject} to {gmail_to} with message {" | ".join(message.splitlines())}')


    try:
        # Log in to the Gmail account and send the message
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            logger.debug('logging into gmail')
            smtp.login(gmail_username, gmail_password)
            logger.debug('sending gmail')
            smtp.sendmail(gmail_username, gmail_to, message)

        logger.info(f"Email sent to {gmail_to} with subject: {subject} and body {body}")
    except smtplib.SMTPAuthenticationError as e:
        logger.exception(f"Failed to authenticate with Gmail. Error message: {str(e)}")
        logger.error(f'raw message is {raw_message}')
        return False
    except smtplib.SMTPServerDisconnected as e:
        logger.exception(f"Failed to connect to the SMTP server. Error message: {str(e)}")
        logger.error(f'raw message is {raw_message}')
        return False
    except smtplib.SMTPException as e:
        logger.exception(f"SMTP error occurred while trying to send email. Error message: {str(e)}")
        logger.error(f'raw message is {raw_message}')
        return False
    except UnicodeEncodeError as e:
        logger.exception(f'message has unrecognized unicode character.')
        logger.error(f'raw message is {raw_message}')
    except Exception as e:
        logger.exception(f"Failed to send email with subject: {subject} and body {body}. Message is {message} Error message: {str(e)}")
        logger.error(f'raw message is {raw_message}')
        raise e
    return True



def Send(content:str, attachments:Optional[list[io.BytesIO]]):
    _send_email_with_gmail(subject='test', body=content)


if __name__ == '__main__':
    from log_config import configure_logging
    configure_logging()
    logger = logging.getLogger(__name__)
    import dotenv
    dotenv.load_dotenv('.env')
    _send_email_with_gmail(subject='test', body='test')