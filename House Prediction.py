import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("House_Prediction.csv")

df.columns = [
    'CrimeRate', 'ResidentialLand', 'IndustrialArea', 'NearRiver',
    'Pollution', 'Rooms', 'OldHouses', 'DistanceToCity',
    'HighwayAccess', 'PropertyTax', 'StudentTeacherRatio',
    'PopulationScore', 'LowIncomePercent', 'HousePrice'
]
df.isnull().sum()
df.fillna(df.mean(), inplace=True)
df.duplicated().sum()
df.drop_duplicates(inplace=True)

# Quick check
print(df.head())
print(df.info())
print(df.describe())

import matplotlib.pyplot as plt

df['HousePrice'].hist()
plt.title('Distribution of House Prices')
plt.show()

plt.scatter(df['Rooms'], df['HousePrice'])
plt.xlabel('Rooms')
plt.ylabel('House Price')
plt.title('Rooms vs House Price')
plt.savefig('rooms_vs_price.png')  # saves image
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()