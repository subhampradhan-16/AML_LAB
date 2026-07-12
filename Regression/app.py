from flask import Flask,render_template,request
import matplotlib.pyplot as plt
import math
import numpy as np


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def linearregression(x,y):
    x = np.array(list(map(float, x.split(","))))
    y = np.array(list(map(float, y.split(","))))
    
    if len(x) != len(y):
        return "Error: Number of X and Y values must be equal."

    if len(x) < 2:
        return "Error: At least 2 data points are required."

    n=len(x)
    
    x_sum=sum(x)
    y_sum=sum(y)
    xy_sum=sum(x*y)
    x2_sum=sum(x**2)
    denominator = (n * x2_sum) - (x_sum ** 2)

    if denominator == 0:
        return "Error: Cannot calculate regression because all X values are identical."

    m = ((n * xy_sum) - (x_sum * y_sum)) / denominator
    c = (y_sum - (m * x_sum)) / n
    
    equation=f"y = {m:.2f}x + {c:.2f}"
    
    y_pred = []

    for value in x:
        y_pred.append(m * value + c)
        
    mse = 0

    for i in range(n):
        mse += (y[i] - y_pred[i]) ** 2

    mse /= n
    
    rmse = math.sqrt(mse)
    mae = 0

    for i in range(n):
        mae += abs(y[i] - y_pred[i])

    mae /= n
    ss_res = 0
    ss_tot = 0

    for i in range(n):
        ss_res += (y[i] - y_pred[i]) ** 2
        ss_tot += (y[i] - (y_sum/n)) ** 2

    r2 = 1 - (ss_res / ss_tot)
    
    return x, y,equation, y_pred, mse, rmse, mae, r2


@app.route("/calculate", methods=["POST"])
def calculate():
    x = request.form["x_values"]
    y = request.form["y_values"]
    x,y,equation, y_pred, mse, rmse, mae, r2 = linearregression(x, y)

    return render_template(
    "index.html",
    x_values=request.form["x_values"],
    y_values=request.form["y_values"],
    equation=equation
)
@app.route("/visualize", methods=["POST"])
def visualize():
    x = request.form["x_values"]
    y = request.form["y_values"]
    
    x,y,equation, y_pred, mse, rmse, mae, r2 = linearregression(x, y)
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, color="blue", label="Actual Data")
    plt.plot(x, y_pred, color="red", label="Best Fit Line")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.title("Linear Regression Best Fit Line")
    plt.grid(True)
    text = (
    f"MSE  = {mse:.4f}\n"
    f"RMSE = {rmse:.4f}\n"
    f"MAE  = {mae:.4f}\n"
    f"R²   = {r2:.4f}"
)

    plt.text(
        0.02,
        0.98,
        text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.8)
    )
    plt.legend()
    plt.savefig("static/graph.png")
    plt.close()
    return render_template(
    "index.html",
    x_values=request.form["x_values"],
    y_values=request.form["y_values"],
    equation=equation,
    show_graph=True
)
if __name__=="__main__":
    app.run(debug=True)