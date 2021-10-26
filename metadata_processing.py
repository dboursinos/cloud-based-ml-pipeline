import pandas as pd
import pymysql
import tensorflow as tf
import mlflow

mysql_connection = pymysql.connect(host=mysql-server, user='root', password='admin', db='mlflow', charset='utf8')
                    
metr = "SELECT * FROM metrics"
df = pd.read_sql(metr, mysql_connection)

# keep only the metric you optimize - in my case the Test Silhouette
df=df[df['key']=='Test Silhouette']
print(df)
print("Test sillhouettes for all the trained models:")
print(df)
best_silhouette=df.iloc[df['value'].argmax()]
print("UUID\n{}\nhas the maximum test silhouette of\n{}\n".format(best_silhouette['run_uuid'],best_silhouette['value']))
print("The raining parameters used are:")

par = "SELECT * FROM params"
df = pd.read_sql(par, mysql_connection)
print(df[df['run_uuid']==best_silhouette['run_uuid']])

runs = "SELECT * FROM runs"
df = pd.read_sql(runs, mysql_connection)

model_uri=df[df['run_uuid']==best_silhouette['run_uuid']]['artifact_uri'].values[0]+'/keras-model'
model=mlflow.keras.load_model(model_uri)
model.summary()