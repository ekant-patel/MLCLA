from flask import Flask, jsonify

app = Flask(__name__)

orders = [
    {"order_id": 1, "product": "Laptop", "amount": 1200.50},
    {"order_id": 2, "product": "Keyboard", "amount": 80.75},
    {"order_id": 3, "product": "Mouse", "amount": 25.50}
]

@app.route('/orders', methods=['GET'])
def get_orders():
    return jsonify(orders)

if __name__ == "__main__":
    app.run(debug=True)
