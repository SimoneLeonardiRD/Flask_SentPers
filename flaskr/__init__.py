from flask import Flask, request, render_template
from bert_serving.client import BertClient
import os
import torch
import torch.nn as nn
import random
import sentperslib as spl

PATH = "../BERT/big5/Models/"
PATHbhv = "../BERT/bhv/"


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/')
    def my_form():
        myCmd = 'rm flaskr/static/images/*'
        os.system(myCmd)
        big5plot = False
        # big5plot is needed in flaskr/templates/main_page.html
        return render_template('main_page.html')

    @app.route('/', methods=['POST'])
    def my_form_post():
        text = request.form['text']
        big5_scores = []
        print("start bert client")
        bc = BertClient(ip='0.0.0.0')
        # ip to connect with bert-as-a-service server
        print("encoding")
        tweetEmbeddings = bc.encode([text])
        for label in ["O", "C", "E", "A", "N"]:
            print("model loading")
            model = nn.Sequential(nn.Linear(768, 300),
                                  nn.ReLU(), nn.Linear(300, 1))
            model.load_state_dict(torch.load(PATH+"SentPers_"+label))
            model.eval()
            print(label, " score predicting")
            result = model(torch.from_numpy(tweetEmbeddings[0]))
            # big5 are on a scale 1 to 5
            # big5_scores.append(round(result[0],3))
            big5_scores.append(round(result.item(), 3))
            # big5 in percentage
            # big5_scores.append(round(((result[0]-1)*100/4),2))
        print("OCEAN: ", big5_scores)

        filename = str(random.randint(1, 100))
        big5_mean = []
        big5_mean = spl.mean_big5_5lines("cls_nn_statistics_multilanguage.txt")
        spl.radar_plot(handle="big5", big5=big5_scores,
                       filename=filename, big5_mean=big5_mean)
        bhv_scores = spl.compute_bhv(text=text)
        bc.close()  # close bert-as-a-service client
        return render_template('main_page.html', bhvscores=bhv_scores,
                               big5scores=big5_scores, insertedtext=text,
                               big5plot=True, filename=filename)

    @app.route('/twitter')
    def login_and_download():
        return "twitter login and posts download"

    @app.route('/questionnaire')
    def questionnaire_form():
        return "do questionnaire and compare results"

    return app
