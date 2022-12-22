from datetime import datetime
import time

print(datetime.now().strftime(r"%Y%m%d_%H%M%S"))

video_start_time = datetime.now()

time.sleep(5)

delta = datetime.now() - video_start_time

print(datetime.now().strftime(r"%Y%m%d_%H%M%S"))
print(f'seconds is {delta}')