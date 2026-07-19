import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_name = input("Output Name : ")
y = list(map(float, input(f"{y_name} Values : ").split(",")))

n = int(input("No. of Input Features : "))

data = {}

for i in range(n):
    name = input(f"Input {i+1} Name : ")
    val = list(map(float, input(f"{name} Values : ").split(",")))
    data[name] = val

data[y_name] = y

df = pd.DataFrame(data)

X = df.drop(columns=[y_name])
y = df[y_name]

model = LinearRegression()
model.fit(X, y)

print("\nIntercept :", round(model.intercept_,4))
print("\nCoefficients")

for i, j in zip(X.columns, model.coef_):
    print(i, ":", round(j,4))

while True:

    ch = input("\nTest (yes/no) : ").lower()

    if ch == "yes":

        test = []

        for i in X.columns:
            test.append(float(input(f"{i} : ")))

        ans = model.predict([test])

        print(f"Predicted {y_name} :", round(ans[0],4))

    elif ch == "no":
        break

    else:
        print("Wrong Value")

pred = model.predict(X)

mae = mean_absolute_error(y, pred)
mse = mean_squared_error(y, pred)
rmse = mse**0.5
r2 = r2_score(y, pred)

print("\nMAE :", round(mae,4))
print("MSE :", round(mse,4))
print("RMSE:", round(rmse,4))
print("R2  :", round(r2,4))

plt.figure(figsize=(8,5))
plt.plot(y.values, marker="o", label="Actual")
plt.plot(pred, marker="s", label="Predicted")
plt.title("Actual vs Predicted")
plt.xlabel("Sample")
plt.ylabel(y_name)
plt.legend()
plt.grid()
plt.show()

metrics = ["MAE","MSE","RMSE","R2"]
values = [mae,mse,rmse,r2]

plt.figure(figsize=(8,5))
bars = plt.bar(metrics, values)

plt.title("Error Metrics")
plt.xlabel("Metrics")
plt.ylabel("Value")

for i in range(len(metrics)):
    plt.text(metrics[i], values[i], f"{values[i]:.2f}")

plt.grid(axis="y")
plt.show()