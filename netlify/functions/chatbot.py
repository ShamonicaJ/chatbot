from flask import Flask, request, jsonify
from serverless_wsgi import handle_request  # ✅ Correct import

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    return jsonify({"response": "📚 Recommendation: The Silent Patient by Alex Michaelides"})

# Netlify function handler
def handler(event, context):
    from netlify_adapter import serverless_wsgi
    return serverless_wsgi.handle_request(app, event, context)
