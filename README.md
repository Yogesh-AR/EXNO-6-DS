# EXNO-6-DS-DATA VISUALIZATION USING SEABORN LIBRARY

# Aim:
  To Perform Data Visualization using seaborn python library for the given datas.

# EXPLANATION:
Data visualization is the graphical representation of information and data. By using visual elements like charts, graphs, and maps, data visualization tools provide an accessible way to see and understand trends, outliers, and patterns in data.

# Algorithm:
STEP 1:Include the necessary Library.

STEP 2:Read the given Data.

STEP 3:Apply data visualization techniques to identify the patterns of the data.

STEP 4:Apply the various data visualization tools wherever necessary.

STEP 5:Include Necessary parameters in each functions.

# Coding and Output:
```
import seaborn as sns 
import matplotlib.pyplot as plt
x=[1,2,3,4,5]
y=[3,6,2,7,1]
sns.lineplot(x=x,y=y)
```
<img width="720" height="567" alt="image" src="https://github.com/user-attachments/assets/8e6bea02-c1a0-496d-a1a4-46e5a5766b14" />

```
df=sns.load_dataset("tips")
df
```
<img width="483" height="462" alt="image" src="https://github.com/user-attachments/assets/c718611e-472c-44f0-bf6d-e998ecb1bc14" />

```
sns.lineplot(x="total_bill",y="tip",data=df,hue="sex",linestyle="solid",legend="auto")
```
<img width="794" height="588" alt="image" src="https://github.com/user-attachments/assets/7ed23c01-c22e-4f89-a84e-a10481bf9cce" />

```
x=[1,2,3,4,5]
y1=[3,5,2,6,1]
y2=[1,6,4,1,8]
y3=[5,2,7,1,4]
sns.lineplot(x=x,y=y1)
sns.lineplot(x=x,y=y2)
sns.lineplot(x=x,y=y3)
plt.title('Multi-Line Plot')
plt.xlabel('X Label')
plt.ylabel('Y Label')
```
<img width="762" height="620" alt="image" src="https://github.com/user-attachments/assets/26fdce42-2407-4c11-b309-1348fbea3a59" />

```
import seaborn as sns 
import matplotlib.pyplot as plt  
tips = sns.load_dataset('tips') 
avg_total_bill = tips.groupby('day') ['total_bill'].mean() 
avg_tip = tips.groupby('day') ['tip'].mean()
plt.figure(figsize=(8, 6))
p1 = plt.bar(avg_total_bill.index, avg_total_bill, label='Total Bill') 
p2 = plt.bar(avg_tip.index, avg_tip, bottom=avg_total_bill, label='Tip')
plt.xlabel('Day of the Week')
plt.ylabel('Amount')
plt.title ('Average Total Bill and Tip by Day')
plt.legend()
```
<img width="946" height="737" alt="image" src="https://github.com/user-attachments/assets/fe2057ae-ef1e-48b5-9a43-47f2e35083ae" />

```
years=range(2000,2012)
apples=[0.895,0.91,0.919,0.926,0.929,0.931,0.934,0.936,0.937,0.9375,0.9372,0.939]
oranges=[0.962,0.941,0.930,0.923,0.918,0.908,0.907,0.904,0.901,0.898,0.9,0.896]
plt.bar(years,apples)
plt.bar(years,oranges,bottom=apples)
```
<img width="738" height="571" alt="image" src="https://github.com/user-attachments/assets/f8e079f5-3fe2-49fe-91a1-8a9a3025e77d" />

```
import seaborn as sns
dt=sns.load_dataset('tips')
sns.barplot(x='day',y='total_bill',hue='sex',data=dt,palette='Set1')
plt.xlabel('Day of the Week')
plt.ylabel('Total Bill')
plt.title('Total Bill by day and Gender')
```
<img width="775" height="608" alt="image" src="https://github.com/user-attachments/assets/1ed1ed52-241e-4986-8e7a-91dba5344f6c" />

```
import pandas as pd
tit=pd.read_csv("titanic_dataset (1).csv")
tit
```
<img width="1243" height="452" alt="image" src="https://github.com/user-attachments/assets/87894950-0787-4539-8dd0-c3c0ac084fa7" />

```
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare',data=tit,palette='rainbow')
plt.title("Fare of Passenger by Embarked Town")
```
<img width="916" height="635" alt="image" src="https://github.com/user-attachments/assets/6135daba-5e08-4f27-b591-000f9fe7fb04" />

```
plt.figure(figsize=(8,5))
sns.barplot(x='Embarked',y='Fare', data=tit, palette='rainbow', hue='Pclass')
plt.title("Fare of Passenger by Embarked Town, Divided by Class")
```
<img width="974" height="635" alt="image" src="https://github.com/user-attachments/assets/c4f77c66-d41a-409d-8369-f1fd28b49c49" />

```
import seaborn as sns
tips = sns.load_dataset('tips')
sns.scatterplot(x='total_bill', y='tip', hue='sex', data=tips)
plt.xlabel ('Total Bill')
plt.ylabel('Tip Amount')
plt.title('Scatter Plot of Total Bill vs. Tip Amount')
```
<img width="781" height="623" alt="image" src="https://github.com/user-attachments/assets/aa4dc05c-5f97-45eb-a198-528be19e95e9" />

```
import seaborn as sns 
import numpy as np 
import pandas as pd
np.random.seed (1)
num_var = np.random.randn(1000)
num_var = pd.Series (num_var,name="Numarical Variable")
num_var
```
<img width="582" height="278" alt="image" src="https://github.com/user-attachments/assets/d20af2b3-db20-4252-b0ee-fa4b7907ab23" />

```
sns.histplot(data=num_var,kde=True)
```
<img width="782" height="594" alt="image" src="https://github.com/user-attachments/assets/95f7c0b8-d80b-47a7-b100-30e27015063c" />

```
df=pd.read_csv("titanic_dataset (1).csv")
df
```
<img width="1243" height="458" alt="image" src="https://github.com/user-attachments/assets/3689bd21-5d54-46ab-9e53-a146abca8536" />

```
sns.histplot(data=df,x="Pclass",hue="Survived",kde=True)
```
<img width="799" height="593" alt="image" src="https://github.com/user-attachments/assets/677e6457-bfbb-4361-9c84-67e39803c402" />

```
import seaborn as sns
import pandas as pd
tips=sns.load_dataset('tips')
sns.boxplot(x=tips['day'],y=tips['total_bill'],hue=tips['sex'])
```
<img width="803" height="580" alt="image" src="https://github.com/user-attachments/assets/38f7095a-3266-4c86-b0e6-8a31146d997a" />

```
sns.boxplot(x="day", y="total_bill", hue="smoker", data=tips, linewidth=2, width=0.6, 
boxprops={"facecolor": "lightblue", "edgecolor": "darkblue"}, 
whiskerprops={"color": "black", "linestyle": "--", "linewidth": 1.5 }, 
capprops={"color": "black", "linestyle": "--", "linewidth":1.5})
```
<img width="798" height="597" alt="image" src="https://github.com/user-attachments/assets/e9415dad-2f9a-4421-b157-1adf0d320867" />

```
sns.violinplot(x="day", y="total_bill", hue="smoker", data=tips, linewidth=2, width=0.6, palette="Set3", inner="quartile")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill")
plt.title("Violin Plot of Total Bill by Day and Smoker Status")
```
<img width="782" height="606" alt="image" src="https://github.com/user-attachments/assets/f95dac52-05f8-41ed-a9ba-4056c0518cfe" />

```
import seaborn as sns
sns.set(style='whitegrid')
tip=sns.load_dataset('tips')
sns.violinplot(x='day',y='tip',data=tip)
```
<img width="778" height="596" alt="image" src="https://github.com/user-attachments/assets/324148a0-ba86-4236-9562-f316d6ee220f" />

```
sns.kdeplot(data=tips,x='total_bill',hue='time',multiple='fill',linewidth=3,palette='Set2',alpha=0.8)
```
<img width="788" height="599" alt="image" src="https://github.com/user-attachments/assets/04cc0779-522a-4b1b-9ce7-23dea8197517" />

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
mart=pd.read_csv("supermarket.csv")
mart
```
<img width="1233" height="640" alt="image" src="https://github.com/user-attachments/assets/a89a34d1-cc6d-4bed-a71d-df347ac8bc70" />

```
sns.kdeplot(data=mart,x='Total',hue='Payment',multiple='stack')
```
<img width="821" height="597" alt="image" src="https://github.com/user-attachments/assets/7b58d69b-1688-4ff8-9082-6654fc494035" />

```
sns.kdeplot(data=mart,x='Unit price',y='gross income')
```
<img width="781" height="597" alt="image" src="https://github.com/user-attachments/assets/0dcd2d4d-f014-428e-8f1e-287e46f570d4" />

```
data=np.random.randint(low=1,high=100,size=(10,10))
print("The data to be plotted:\n")
print(data)
```
<img width="375" height="278" alt="image" src="https://github.com/user-attachments/assets/39215a83-e9c1-496b-89a9-7592854b51de" />

```
hm=sns.heatmap(data=data,annot=True)
```
<img width="683" height="541" alt="image" src="https://github.com/user-attachments/assets/ed4eb205-9da0-4f9f-9d98-907d8af33620" />

```
import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt
tips = sns.load_dataset("tips")
numeric_cols = tips.select_dtypes(include=np.number).columns 
corr = tips [numeric_cols].corr()
sns.heatmap(corr,annot=True,cmap="plasma",linewidths=0.5)
```
<img width="686" height="577" alt="image" src="https://github.com/user-attachments/assets/8afb2c7f-a3e4-4bd7-b31d-275cd9c90ad0" />


# Result:
Thus we have successfully performed Data Visualization using seaborn library for the given data.
