import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("House_Prediction_Clean.csv")
#Define Features and Target
X = df.drop('HousePrice', axis=1)  # independent variables
y = df['HousePrice']               # target variable

#Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

#R-squared
r2 = model.score(X_test, y_test)
print("R-squared:", r2)

#Make predictions
y_pred = model.predict(X_test)

#Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

#Interpret Coefficients
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coeff_df.sort_values(by='Coefficient', ascending=False))

#Select Features (We only use important features)
features = df[['Rooms', 'CrimeRate', 'HousePrice']]

#Standardize the Data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

#Find Optimal Number of Clusters
inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia)
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

#Apply K-Means
kmeans = KMeans(n_clusters=3)
df['Cluster'] = kmeans.fit_predict(scaled_features)

#Visualize Clusters
plt.scatter(df['Rooms'], df['HousePrice'], c=df['Cluster'])

plt.xlabel('Rooms')
plt.ylabel('House Price')
plt.title('House Clusters')
plt.show()

