import time


import datetime

print(
    time.strftime('%Y%m%d_%H%M%S_%Z', time.localtime())
)