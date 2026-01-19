from kafka import KafkaProducer
import json
import time
import random

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

while True:
    message = {'sensor': 'temp', 'value': random.randint(20, 35)}
    producer.send('test_topic', message)
    print("Sent:", message)
    time.sleep(1)
