import keyring
import os
import logging
import argparse

logging.basicConfig(filename=os.path.join('logs', 'sighthound_email_patch_set_creds.log'), level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()

parser.add_argument('--from_email', type=str, help='email to send from')
parser.add_argument('--to_email', type=str, help='emails to send to')
parser.add_argument('--email_password', type=str, help='email to send from')

args = parser.parse_args()

logger.info('adding cred')
keyring.set_password("sighthound_email_patch_creds", "email_address", args.from_email)
keyring.set_password("sighthound_email_patch_creds", "email_password", args.to_email)
keyring.set_password("sighthound_email_patch_creds", "send_to_email_address", args.email_password)