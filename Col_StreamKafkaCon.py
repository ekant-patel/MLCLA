from kafka import KafkaConsumer
import json

consumer = KafkaConsumer(
    'test_topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='latest',
    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
)

for msg in consumer:
    print("Received:", msg.value)
