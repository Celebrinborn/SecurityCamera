import datetime

current_datetime = datetime.datetime.now()
unix_time = int(datetime.datetime.timestamp(current_datetime))

print(current_datetime)
print(unix_time)

print(current_datetime.strftime(r'%Y%m%d_%H%M%S_%f'))