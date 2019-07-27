# -*- coding:utf-8 -*-

# @Time    : 2019-03-05 11:25

# @Author  : Swing


from flask import Flask, render_template, request, jsonify
from flask_uploads import UploadSet, configure_uploads, IMAGES
from model.inference_interface import inference
import os

app = Flask(__name__)

app.config['UPLOADED_PHOTO_DEST'] = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOADED_PHOTO_ALLOW'] = IMAGES

image = UploadSet('PHOTO')
configure_uploads(app, image)


# TODO: 参数配置
# 模型checkpoint文件夹
ckpt_dir = ''
# word_counts.txt路径
word_counts = ''


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['post', 'get'])
def image_upload():
    if 'image' in request.files:
        try:
            filename = image.save(request.files['image'], folder='image')

            results = inference(image.path(filename),
                                ckpt_dir,
                                word_counts)

            return jsonify({'success': True, 'results': results})
        except Exception as err:
            print(err)
            return jsonify({'success': False, 'message': 'Unknown error!'})

    return jsonify({'success': False, 'message': 'No file found!'})


if __name__ == '__main__':
    app.run(debug=True)
