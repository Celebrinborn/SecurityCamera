import datetime
import numpy as np
import cv2
import os


def get_first_primes(upper_limit):
   number = 1
   while number <= upper_limit:
       if is_prime(number):
           yield number
       number += 1


def is_prime(number):
   if number > 1:
       if number == 2:
           return True
       if number % 2 == 0:
           return False
       for current in range(3, ((number // 2) + 1), 2):
           if number % current == 0:
               return False
       return True
   return False

if __name__ == '__main__':
    print("Your number: ")
    n = 14
    generator = get_first_primes(int(n))
    #    for next_value in generator:
    #        print(next_value, end=', ')
    print('starting next')
    print(next(generator))

    print('time')
    print(datetime.datetime.timestamp(datetime.datetime.now()))
    print(datetime.datetime.now().hour)
    print(datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S_%f'))

    test_width = 1280
    test_height= 720
    test_depth = 3
    frames = [np.random.rand(test_width, test_height,test_depth), np.random.rand(test_height,test_width,test_depth), np.random.rand(test_height,test_width,test_depth)]

    fourcc = cv2.VideoWriter_fourcc(*'FMP4')

    print(os.getcwd())
    path = os.path.join('D','Documents','.vscode','SecurityCamera','tests','test_output')

    filename = 'testoutput'
    file_extention = 'avi'

    fqfn = f'{os.path.join(path, filename)}.{file_extention}'

    print(fqfn)
    videowriter = cv2.VideoWriter(fqfn, fourcc, 25, (test_width, test_height))
    for frame in frames:
        videowriter.write
    videowriter.release()




