# NOTE:
# This file is based on your original app.py and modified to render
# the merged "index.html" instead of "result.html".

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

def make_plots(y_vals, predictions, y_name):
    plt.figure(figsize=(6,4))
    plt.scatter(y_vals, predictions)
    mn=min(float(np.min(y_vals)), float(np.min(predictions)))
    mx=max(float(np.max(y_vals)), float(np.max(predictions)))
    plt.plot([mn,mx],[mn,mx])
    plt.xlabel(f"Actual {y_name}")
    plt.ylabel(f"Predicted {y_name}")
    plt.grid(True)

    b1=io.BytesIO()
    plt.savefig(b1, format="png", bbox_inches="tight")
    plt.close()
    b1.seek(0)
    p1=base64.b64encode(b1.read()).decode()

    mse=mean_squared_error(y_vals,predictions)
    mae=mean_absolute_error(y_vals,predictions)
    rmse=np.sqrt(mse)
    r2=r2_score(y_vals,predictions)

    plt.figure(figsize=(6,4))
    names=["R2","RMSE","MAE","MSE"]
    vals=[r2,rmse,mae,mse]
    bars=plt.barh(names,vals)
    for bar in bars:
        w=bar.get_width()
        plt.text(w,bar.get_y()+bar.get_height()/2,f" {w:.4f}",va="center")
    plt.grid(True)

    b2=io.BytesIO()
    plt.savefig(b2, format="png", bbox_inches="tight")
    plt.close()
    b2.seek(0)
    p2=base64.b64encode(b2.read()).decode()

    return p1,p2,mse,mae,rmse,r2


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    y_name=request.form["y_name"].strip()
    y_input=request.form["y_values"]

    y_vals=np.array([float(i) for i in y_input.split(",") if i.strip()])

    x_names=[]
    x_values_raw=[]
    x_data={}

    idx=0
    while f"x_name_{idx}" in request.form:
        n=request.form[f"x_name_{idx}"].strip()
        v=request.form[f"x_values_{idx}"].strip()
        if v:
            arr=np.array([float(i) for i in v.split(",") if i.strip()])
            if len(arr)!=len(y_vals):
                return "All columns must have same number of values.",400
            x_names.append(n)
            x_values_raw.append(v)
            x_data[f"x{idx+1}"]=arr
        idx+=1

    model=LinearRegression()
    X=pd.DataFrame(x_data)
    model.fit(X,y_vals)
    pred=model.predict(X)

    intercept=model.intercept_
    coeff=model.coef_

    equation=f"y = {intercept:.4f}"
    slopes={}
    for i,c in enumerate(coeff):
        equation+=f" + ({c:.4f} × {x_names[i]})"
        slopes[x_names[i]]=f"{c:.4f}"

    plot1,plot2,mse,mae,rmse,r2=make_plots(y_vals,pred,y_name)

    return render_template(
        "index.html",
        equation=equation,
        intercept=f"{intercept:.4f}",
        slopes=slopes,
        y_name=y_name,
        y_input=y_input,
        x_names=x_names,
        x_raw_str=";".join(x_values_raw),
        plot1=plot1,
        plot2=plot2,
        prediction=None,
        mse=f"{mse:.4f}",
        mae=f"{mae:.4f}",
        rmse=f"{rmse:.4f}",
        r2=f"{r2:.4f}"
    )


@app.route("/predict", methods=["POST"])
def predict():
    y_name=request.form["y_name"]
    y_input=request.form["y_input"]
    x_raw=request.form["x_raw_str"]

    y_vals=np.array([float(i) for i in y_input.split(",") if i.strip()])

    x_data={}
    x_names=[]
    feature_strings=x_raw.split(";")
    for i,s in enumerate(feature_strings):
        x_data[f"x{i+1}"]=[float(j) for j in s.split(",") if j.strip()]
        x_names.append(f"x{i+1}")

    model=LinearRegression()
    X=pd.DataFrame(x_data)
    model.fit(X,y_vals)
    pred=model.predict(X)

    sample={}
    for i in range(len(feature_strings)):
        sample[f"x{i+1}"]=float(request.form[f"pred_x_{i}"])

    result=model.predict(pd.DataFrame([sample]))[0]

    intercept=model.intercept_
    coeff=model.coef_
    slopes={f"x{i+1}":f"{coeff[i]:.4f}" for i in range(len(coeff))}
    equation="y = {:.4f}".format(intercept)
    for i,c in enumerate(coeff):
        equation+=f" + ({c:.4f} × x{i+1})"

    plot1,plot2,mse,mae,rmse,r2=make_plots(y_vals,pred,y_name)

    return render_template(
        "index.html",
        equation=equation,
        intercept=f"{intercept:.4f}",
        slopes=slopes,
        y_name=y_name,
        y_input=y_input,
        x_names=x_names,
        x_raw_str=x_raw,
        plot1=plot1,
        plot2=plot2,
        prediction=f"{result:.4f}",
        mse=f"{mse:.4f}",
        mae=f"{mae:.4f}",
        rmse=f"{rmse:.4f}",
        r2=f"{r2:.4f}"
    )


if __name__=="__main__":
    app.run(debug=True)
