import pandas as pd
from datetime import datetime
import csv

df = pd.read_csv("../../datasets/text_data/Electricity/cleaned_data_2.csv")

# list = []
# id = 0
# month = 0
# for i in range(df.shape[0]):
#     id1 = df["Customer ID"][i]
#     x = datetime.strptime(str(df["Consumption Month"][i]), '%m-%Y')
#     newmonth = x.month
#     if id1 == id:
#         if month == 12:
#             if newmonth != 1:
#                 print("wrong id: ", id1)
#
#         else:
#             if newmonth != month + 1:
#                 print("wrong id: ", id1)
#
#     id = id1
#     month = newmonth

l = []
id = 0
file = open("../../datasets/text_data/Electricity/formatted_data.csv", "a")
# w = csv.writer(file, delimiter = ',')
#
for i in range(df.shape[0]):
    id1 = df["Customer ID"][i]

    if id1 != id:
        id = id1
        # file.write(id1)
        # w([x for x in l])
        file.writelines("\n")
        for item in l:
            file.write("%s," % item)
        l = []

    l.append(df["Sum"][i])

