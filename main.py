import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

#reading data
DATA = pd.read_csv("weblog_1.csv")
DATA = pd.DataFrame(DATA)

#get most frequent site
maxFrequentSite = DATA['URL'].mode()[0]

counter = 0
for index, row in DATA.iterrows():
    if(maxFrequentSite == row["URL"]):
        counter += 1
print("the most frequent site is:%s, frequency:%d " % (maxFrequentSite, counter))

#counts of months
months = []
counts = []
counter = -1
availableMonths = [
    "Nov",
    "Feb",
    "Mar",
    "Dec",
    "Jan",
]
for index, row in DATA.iterrows():
    temp = str(row["Time"]).split('/')
    if (maxFrequentSite in row["URL"]) and (temp[1] not in months) and temp[1] in availableMonths:
        months.append(temp[1])
        counter += 1
        counts.append(1)
    elif( (maxFrequentSite in row["URL"]) and (temp[1] in months) and temp[1] in availableMonths):
        counts[months.index(temp[1])] += 1
    elif(maxFrequentSite in row["URL"] and len(temp) > 1 and temp[1] in availableMonths):
        counts[counter] += 1

plt.bar(months, counts)
plt.xlabel("months")
plt.ylabel("counts")
plt.show()
plt.scatter(months, counts)
plt.xlabel("months")
plt.ylabel("counts")
plt.show()

#counts of hour and getting rush hour
hours = []
counts = []
counter = -1
for index, row in DATA.iterrows():
    temp = str(row["Time"]).split('/')
    if (len(temp) > 1) and (len(temp[2].split(':')) > 1) and (temp[2].split(':')[1] not in hours) and (temp[1] in availableMonths):
        hours.append(temp[2].split(':')[1])
        counter += 1
        counts.append(1)
    elif(len(temp) > 1 and len(temp[2].split(':')) > 1 and temp[2].split(':')[1] in hours and temp[1] in availableMonths):
        counts[hours.index(temp[2].split(':')[1])] += 1
    elif(len(temp) > 1 and temp[1] in availableMonths):
        counts[counter] += 1

for i in range(len(hours)):
    for j in range(len(hours)):
        if(int(hours[i]) < int(hours[j])):
            hours[i],hours[j] = hours[j], hours[i]
            counts[i],counts[j] = counts[j], counts[i]

rushHour = hours[counts.index(max(counts))]
print("the rush hour is: ", rushHour)
plt.bar(hours, counts)
plt.xlabel("hours")
plt.ylabel("counts")
plt.show()
plt.scatter(hours,counts)
plt.xlabel("hours")
plt.ylabel("counts")
plt.show()

for i in range(len(hours)):
    hours[i] = int(hours[i])

#time series algorithm
X = pd.DataFrame({'hours': hours, 'counts': counts})
X = X.hours.values

size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]

predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(0,1,3))
    modelFit = model.fit(disp=1)
    output = modelFit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f, difference=%f' % (yhat, obs, abs(yhat-obs)))

error = mean_squared_error(test, predictions)
print("MSE is: ", error)
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()