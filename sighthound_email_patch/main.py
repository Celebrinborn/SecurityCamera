import logging
import argparse
import sys
import os
from datetime import datetime
import time

import keyring
import smtplib, ssl 
from email.message import EmailMessage
import mimetypes

import logging
logger = logging.getLogger(__name__)

# Create handlers
c_handler = logging.StreamHandler()
f_handler = logging.FileHandler(os.path.join('logs', 'sighthound_email_patch.log'))
c_handler.setLevel(logging.DEBUG) # Change me to change console messages
f_handler.setLevel(logging.INFO) # Change me to change file messages

# Create formatters and add it to handlers
c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(c_format)
f_handler.setFormatter(f_format)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)


parser = argparse.ArgumentParser()

parser.add_argument('-p', '--path', type=str, help='path to camera directory (c:/users/adminsistrator/appdata/local/sighthound videos/videos/archive/front east looking west)')
parser.add_argument('-n', '--camera_name', type=str, help='camera name for email')
parser.add_argument('-s', '--subject_line', type=str, help='email subject line', default=f'sighthound person detected')
parser.add_argument('-d', '--delay', type=int, help='time in seconds to wait prior to pulling event', default=0)
parser.add_argument('-c', '--count', type=int, help='number of screenshots to send', default=13)
parser.add_argument('--destination_email', type=str, help='who to send the email to')

add_creds_args = parser.add_mutually_exclusive_group()
add_creds_args.add_argument('--add_creds', action='store_true')

args = parser.parse_args()

# load test variables because I can't get vscode to use the parser
if os.path.exists(os.path.join('sighthound_email_patch', '_settings.py')):
    logger.warning('loading test parser file')
    import _settings
    args = _settings.test_parser()
    print(os.listdir())
else:
    print('running without test vars')
    print(os.listdir())

logger.info('system args')
logger.info(dir(args))

def AddCreds():
    logger.info('adding cred')
    email_address = input('email address: ')
    keyring.set_password("sighthound_email_patch_creds", "email_address", email_address)
    password = input('password: ')
    keyring.set_password("sighthound_email_patch_creds", "email_password", password)

# def SendMail(Port, UserName, UserPassword, sent_from, send_to, attachment_path, subject, body):
#     msg = MIMEMultipart()
#     msg['Subject'] = subject
#     msg['From'] = smtp_user
#     msg['To'] = recipient
#     msg.add_header('Content-Type','text/html')
#     msg.attach(MIMEText(message, 'html'))
#     df.to_csv(textStream,index=False)
#     msg.attach(MIMEApplication(textStream.getvalue(), Name=filename))


def Send_Email(filename):
    
    smtp_server = "smtp.gmail.com" #name of smptp server 
    port = 587  # For starttls 

    message = EmailMessage()
    message['Subject'] = args.subject_line
    message['From'] = keyring.get_password('sighthound_email_patch_creds', 'email_address')
    message['To'] = args.destination_email
    message.set_content(f"""People detected by {args.camera_name} at {datetime.now().strftime(r"%Y%m%d_%H%M%S")}""")

    print(f'file is {filename}')
    print(f'does file exist: {os.path.isfile(filename)}')
    with open(filename, 'rb') as file:
        _content = file.read()
        message.add_attachment(_content, maintype='image', subtype='jpg', filename=filename)
    
    server = smtplib.SMTP(smtp_server, port)
    # s = smtplib.SMTP('smtp.sendgrid.net', 587)
    try:
        server.connect(smtp_server, port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(keyring.get_password('sighthound_email_patch_creds', 'email_address')
            , keyring.get_password('sighthound_email_patch_creds', 'email_password'))
        server.send_message(message)
    except BaseException as e:
        raise e
    finally:
        server.quit()
    

def Main():
    logger.info('sending email')   
    # find first screenshot after event (maybe add delay)
    if not os.path.exists(args.path):
        logger.critical(f'path {str(args.path)} does not exist')
        raise Exception('invalid path')
    base_path = args.path

    # sighthound throws videos into numbered buckets based on first 5 characters of the epoch
    _newest_date = 0 # some date in 1940, just need something older then anything in the directory
    _newest_directory = None
    _second_newest_directory = None
    print(f'list dir: {os.listdir(base_path)}')
    for directory in os.listdir(base_path):
        _modified_date = os.path.getmtime(os.path.join(base_path,directory))
        if _modified_date > _newest_date:
            _second_newest_directory = _newest_directory
            _newest_date = _modified_date
            _newest_directory = directory
            print(f'directory: {directory}, _modified_date: {_modified_date}, _newest_date: {_newest_date}, _newest_directory: {_newest_directory}')

    print(_newest_directory, _second_newest_directory)
    thumbnails_folder_path = os.path.join(base_path, _newest_directory, 'thumbs')
    second_thumbnails_folder_path = os.path.join(base_path, _second_newest_directory, 'thumbs')

    # veriify path is real
    if not os.path.exists(thumbnails_folder_path):
        logger.critical(f'path {str(thumbnails_folder_path)} does not exist')
        raise Exception('invalid thumbnails_folder_path')


    # get a list of files in the directoryl sort by title (they are unix timestamps so sorting in ascending order is fine)
    # and filter to make sure the extention is always .jpg
    print("thumbshot_filename_list = [os.path.join(thumbnails_folder_path, jpg) for jpg in os.listdir(thumbnails_folder_path).sort() if jpg[-4:].lower() == '.jpg']")
    print(thumbnails_folder_path, os.listdir(thumbnails_folder_path))

    thumbshot_filename_list = [os.path.join(thumbnails_folder_path, jpg) for jpg in sorted(os.listdir(thumbnails_folder_path)) if jpg[-4:].lower() == '.jpg']
    second_thumbshot_filename_list = [os.path.join(second_thumbnails_folder_path, jpg) for jpg in sorted(os.listdir(second_thumbnails_folder_path)) if jpg[-4:].lower() == '.jpg']

    # get list of image paths to attach to email
    image_paths_to_send_list = []
    if len(thumbshot_filename_list) > int(args.count): # make sure you don't try to request more files then exist
        image_paths_to_send_list = thumbshot_filename_list[-int(args.count):]
    else: # if you would request too many files, grab the previous folder and add those too
        image_paths_to_send_list = thumbshot_filename_list
        _count_remaining = int(args.count) - len(thumbshot_filename_list)
        if len(second_thumbshot_filename_list) > _count_remaining:
            image_paths_to_send_list = image_paths_to_send_list + second_thumbshot_filename_list[-_count_remaining:]
        else: # make sure you don't grab more files then exist from the backup folder
            image_paths_to_send_list = image_paths_to_send_list + second_thumbshot_filename_list



    # ogger.info('adding cred')
    # email_address = input('email address: ')
    # keyring.set_password("sighthound_email_patch_creds", "email_address", email_address)
    # password = input('password: ')
    # keyring.set_password("sighthound_email_patch_creds", "email_password", password)
    
    #smtp_server = "smtp.gmail.com" #name of smptp server 
    port = 587  # For starttls 

    #send_email()
    _screenshot_to_sent = image_paths_to_send_list[0]
    print('screenshot_path', _screenshot_to_sent)

    Send_Email(_screenshot_to_sent)

    # find {count} screenshots after event

    # write email

    # send email

    # send email

if __name__ == '__main__':
    print('running code')
    if args.add_creds == True:
        print('running add creds')
        AddCreds()
    else:
        print('sending email')
        Main()