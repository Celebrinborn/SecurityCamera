import sys
import os
import logging
import time
from datetime import datetime
from collections import namedtuple

import smtplib
from email.message import EmailMessage


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler(sys.stdout)

file_handler = logging.FileHandler(filename=os.path.join('logs', 'sighthound_monitor.log'))

formatter = logging.Formatter('%(asctime)s - {%(pathname)s:%(lineno)d} - %(levelname)s - %(funcName)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

Screenshot_and_Path = namedtuple('Screenshot_and_Path', ['screenshot_file_name', 'screenshot_file_path'])
def GetNewestScreenshot(root_path) -> Screenshot_and_Path:
    # find newest timebucket
    _newest_folder = 0
    # check for thumbs folder
    if not os.path.exists(os.path.join(root_path, 'thumbs')):
        logger.debug(f'folder {root_path} does not have a thumbs directory, skipping')
        return None
    # find newest photo
    _newest_screenshot_int = 0
    logger.debug(f"list of files in {os.path.join(root_path, 'thumbs')}: {os.listdir(os.path.join(root_path, 'thumbs'))}")
    for screenshot in os.listdir(os.path.join(root_path, 'thumbs')):
        _screenshot_unix_time_name = screenshot.split('.')[0]
        if _screenshot_unix_time_name.isdigit():
            if int(_screenshot_unix_time_name) > _newest_screenshot_int:
                _newest_screenshot_name = screenshot
                _newest_screenshot_int = int(_screenshot_unix_time_name)
    if _newest_screenshot_int == 0:
        logger.debug(f'unable to find any thumbnails in directory {os.path.join(root_path, "thumbs")}')
        return None
    # return results
    # logger.debug(f'path is: {root_path}, thumbs, {_newest_screenshot_name}') # remove me
    return Screenshot_and_Path(_newest_screenshot_int, os.path.join(root_path, 'thumbs', _newest_screenshot_name))

def GetXScreenshots(count:int, screenshot_path:Screenshot_and_Path) -> list:
    path, _ = os.path.split(screenshot_path.screenshot_file_path)
    logger.debug(f'count is: {count} path is {path}')
    screenshots = sorted(os.listdir(path), reverse=True)
    logger.debug(f'screenshots are: {screenshots}')

    # add full path back to screenshots:
    screenshots = [os.path.join(path, x) for x in screenshots]

    if len(screenshots) < count:
        logger.debug(f'folder {path} only has {len(screenshots)} screenshots, {count} rquested')
        results = screenshots
        # todo: get files from previous folder
    else:
        results = screenshots[0:count]
        logger.debug(f'returning {results} from {screenshots}')
    return results

def Send_Email(list_of_files:list, camera_name:str, *_, _isFirst = True):
    # preset values that never change
    smtp_server = "smtp.gmail.com"
    port = 587 

    # create email message
    message = EmailMessage()
    try:
        message['Subject'] = f'Sighthound Camera Event Detected at {camera_name}'
        message['From'] = secrets['login']
        message['To'] = secrets['destination_email']
        message.set_content(f"""People detected by {camera_name} at {datetime.now().strftime(r"%Y%m%d_%H%M%S")}""")
    except KeyError as e:
        logger.critical(f'unable to send email. missing secret {e}, please check "login" and "destination_email" enviorn vars')

    # add each attachment
    for filename in list_of_files:
        if os.path.isfile(filename):
            logger.debug(f'adding {filename}')
            with open(filename, 'rb') as file:
                _content = file.read()
                message.add_attachment(_content, maintype='image', subtype='jpg', filename=filename)
        else:
            logger.error(f'file {filename} does not exist. skipping')

    # initate email server connection
    server = smtplib.SMTP(smtp_server, port)

    # connect to server and send email
    try:
        server.connect(smtp_server, port)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(secrets['login'], secrets['password'])
        server.send_message(message)
    except KeyError as e:
        # log environ key not existing
        logger.error(f'secret login or password keys do not exist or are invalid, {e}', exc_info=True, stack_info=True)
    except BaseException as e:
        # if this is the first time wait 30 seconds then try again
        if _isFirst:
            logger.error(f'unable to send email for unknown reason. waiting 30 seconds then attempting again {e}', exc_info=True, stack_info=True)
            time.sleep(30)
            logger.error('trying email again...')
            Send_Email(list_of_files, camera_name, _isFirst=False)
        # else log the issue as unable to send mail
        else:
            logger.critical(f'unable to send email for second attempt {e}', exc_info=True, stack_info=True)
    finally:
        # cleanups
        server.quit()

# get secrets
secrets = {}
_secrets_path = os.path.join("..", "run", "secrets")
_secrets_list = os.listdir(_secrets_path)
for f in _secrets_list:
    logger.info(f'loading secret {f}')
    with open (os.path.join(_secrets_path, f)) as _file:
        secrets[f] = _file.readline().strip()

_sleeptime = 60
try:
    _sleeptime = int(os.environ['sleeptime'])
    logger.info(f'sleeptime is {_sleeptime} seconds')
except KeyError as e:
    logger.warning('sleeptime os environ not present, defaulting to checking every 60 seconds')
except ValueError as e:
    logger.warning(f'sleeptime os environ variable cannot be cast to int. value is {os.environ["sleeptime"]}. defaulting to 60 seconds')
logger.debug('waiting 60 seconds before checking again')

logger.info('moving to archive folder')
os.chdir('archive')
logger.info(f'folders in directory: {os.listdir()}')

# get list of camera folders and save as a dictionary
_CameraList = os.listdir()
Cameras = {}
for camera in _CameraList:
    if not os.path.isdir(camera):
        os.info(f'skipping file {camera} as this is not a directory')
        continue
    logger.info(f'checking folder {camera}')
    screenshot_path = GetNewestScreenshot(camera)
    if screenshot_path:
        logger.info(f'adding to camera list: {camera} at {screenshot_path}')
        Cameras[camera] = screenshot_path
    else:
        logger.info(f'no results from GetNewestScreenshot returned. not adding {camera} to camera list')
logger.info(f'cameras are {Cameras.keys()}')

logger.info('beginning to start main loop')
while True:
    for camera in Cameras.keys():
        logger.debug(f'checking for new files in archives/{camera}')
        if GetNewestScreenshot(camera) != Cameras[camera]:
            logger.info(f'new file detected in {camera}')
            _count = 15
            if 'image_count' in os.environ:
                if os.environ['image_count'].isdigit():
                    _count = int(os.environ['image_count'])
                else:
                    logger.warning(f'image count environment variable is NOT a digit, is: {os.environ["image_count"]}')
            screenshots = GetXScreenshots(_count, Cameras[camera])
            if len(screenshots) == 0:
                logger.error(f'GetXScreenshots failed to get any screenshots at time {int(time.time())}')
                continue
            logger.debug(f'screenshots sending for email are: {screenshots}')
            Send_Email(screenshots, camera)
        else:
            logger.debug(f'no new files detected in camera {camera}')
    
    # raise Exception('breaking loop for testing purposes')
    time.sleep(_sleeptime)

