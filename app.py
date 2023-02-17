from flask import Flask, render_template, request, jsonify
from prediction_service.prediction import form_response, api_response
import os
import pickle
import yaml
import numpy

params_path = 'params.yaml'
webapp_root = 'webapp'

static_dir =  os.path.join(webapp_root, 'static')
template_dir  = os.path.join(webapp_root, 'templates')


app =  Flask(__name__, static_folder=static_dir, template_folder=template_dir)

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == 'POST':
        try:
            if request.form:
                dict_req = dict(request.form)
                print(dict_req)
                response = form_response(dict_req)
                return render_template("index.html", response= response)
            
            elif request.json:
                response =  api_response(request.json)
                return jsonify(response)
        except Exception as e:
            print(e)
            error =  {"error" : e}
            render_template("404.html", error=  error)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

