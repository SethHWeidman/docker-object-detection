from __future__ import print_function

from flask import Flask, request, render_template, send_file
from detect_objects import detect_objects_filename, detect_objects_file
from PIL import Image
app = Flask(__name__)

@app.route('/detect_objects_filename/', methods=['GET', 'POST'])
def detect_objects_func():
    filename = request.args['filename']
    filename_return = detect_objects_filename(filename)
    return send_file(filename_return)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
