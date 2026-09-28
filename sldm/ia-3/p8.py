import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('synthetic_salary_data.csv')

X = df[['Qualification Level', 'Experience (Years)']] 
y = df['Salary'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

new_data = np.array([[6, 3]])
predicted_salary = model.predict(new_data)
print("Predicted Salary for new data:", predicted_salary[0])

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Experience (Years)', y='Salary', hue='Qualification Level', palette='viridis')
plt.title("Salary vs. Experience Colored by Qualification Level")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary ($)")
plt.legend(title="Qualification Level")
plt.tight_layout()
plt.savefig('p8_plot.png')
print("Graph saved as p8_plot.png")
