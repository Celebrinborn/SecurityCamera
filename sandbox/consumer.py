from kafka import KafkaAdminClient
from kafka.errors import KafkaError

import time

kafka_ip = 'localhost:9092'

for i in range(10):
    try:
        admin_client = KafkaAdminClient(bootstrap_servers=kafka_ip)
        print("Connection successful")
        break
    except KafkaError as e:
        print(f"Failed to connect to Kafka: {e}")
        import sys
        if i == 9:
            sys.exit()
        else :
            time.sleep(5)

from kafka import KafkaConsumer

consumer = KafkaConsumer(
    'test-topic',
    bootstrap_servers=kafka_ip,
    auto_offset_reset='earliest'
)

for message in consumer:
    print(f"Received message: {message.value.decode('utf-8')}")
