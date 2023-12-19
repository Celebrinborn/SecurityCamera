from kafka import KafkaProducer
import time

producer = KafkaProducer(bootstrap_servers='localhost:9092')

topic = 'test-topic'

for i in range(100):
    message = f"Message {i}"
    producer.send(topic, message.encode('utf-8'))
    print(f"Sent: {message}")
    time.sleep(1)

producer.close()
