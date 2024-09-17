from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.get_json()
    x = int(data['x'])
    y = int(data['y'])
    result = x + y
    return jsonify({'sum': result})

if __name__ == '__main__':
    app.run(debug=True)
