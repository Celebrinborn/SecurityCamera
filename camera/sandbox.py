from ImageRecognitionSender import ImageRecognitionSender
import asyncio
import numpy as np


print('creating object')
with ImageRecognitionSender(ip_address='http://localhost:8000') as sender:
        print(sender._event_loop)
        print(sender._task)
        sender.Send_screenshot(np.ndarray(5,2), 'test')
        #sender.Abort(force=True)
        print('closing object')