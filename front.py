# serve the front end
from flask import Flask, render_template

app = Flask(__name__,
            static_folder='templates')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/<path:path>')
def static_file(path):
    print(path)
    return app.send_static_file(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
