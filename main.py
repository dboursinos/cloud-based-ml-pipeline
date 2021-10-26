import tensorflow as tf
from tensorflow.keras.layers import Input,Lambda, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import generic_utils
import numpy as np
import time
from minio import Minio
import os
from sklearn.metrics import silhouette_score
import helper
import SafeML
import pickle
import mlflow
import tf2onnx
from sklearn.manifold import TSNE

input_side_size=30
tf.config.run_functions_eagerly(False)
minioClient = Minio(os.environ.get('MLFLOW_S3_ENDPOINT_URL').split('//')[1],
                  access_key=os.environ.get('AWS_ACCESS_KEY_ID'),
                  secret_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                  secure=False)

mlflow.set_tracking_uri('http://mlflow-server:5000')
data=pickle.loads(minioClient.get_object(bucket_name='dataset',object_name="GeneratedData/preprocessed_data.pickle").read())

def train(data,params):
    cnn_model=SafeML.create_base_model(**params)
    base_model=Model(inputs=cnn_model.input,outputs=cnn_model.get_layer('embedding').output)
    base_model.summary()

    input_a = Input(shape=(input_side_size, input_side_size, 3),name='input1')
    input_b = Input(shape=(input_side_size, input_side_size, 3),name='input2')
    processed_a = base_model(input_a)
    processed_b = base_model(input_b)
    l2_distance_layer = Lambda(
                lambda tensors: K.sqrt(K.sum(K.square(tensors[0] - tensors[1]), axis=1, keepdims=True)))
    l2_distance = l2_distance_layer([processed_a, processed_b])
    siamese_network=Model([input_a, input_b], l2_distance)

    siamese_network.compile(loss=helper.euclidean_loss, optimizer=Adam())
    siamese_network.summary()

    batch_size=32
    num_epochs=600
    epoch_length=200
    callbacks=[]

    best_loss = np.Inf
    train_step = 0
    losses = np.zeros(epoch_length)
    idxs_per_class=helper.class_separation(data['y_train'])
    for epoch_num in range(num_epochs):
        print('Epoch {}/{}'.format(epoch_num+1,num_epochs))
        progbar = generic_utils.Progbar(epoch_length)   # keras progress bar
        iter_num = 0
        start_time = time.time()
        for batch_num in range(epoch_length):
            inputs1,inputs2,targets=helper.get_batch(batch_size,data['x_train'],data['y_train'],idxs_per_class)
            inputs1=np.array(inputs1)
            inputs2=np.array(inputs2)
            loss = siamese_network.train_on_batch([inputs1, inputs2], targets)
            losses[iter_num] = loss
            iter_num+=1
            train_step += 1
            progbar.update(iter_num, [('loss', np.mean(losses[:iter_num]))])

            if iter_num == epoch_length:
                epoch_loss = np.mean(losses)
                mlflow.log_metric("train loss", epoch_loss)
                mlflow.log_metric('episode time', time.time() - start_time)

    onnx_model,_=tf2onnx.convert.from_keras(base_model, opset=13, output_path="test.onnx")
    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    mlflow.log_artifact('model.tflite','tflite-model')
    os.remove('model.tflite')

    mlflow.log_param("embeddings size",p['classifier_embeddings'])
    mlflow.log_param("batch size", batch_size)
    mlflow.log_param("epochs", epoch_num)
    
    mlflow.keras.log_model(base_model, artifact_path="keras-model")
    mlflow.onnx.log_model(onnx_model, artifact_path="onnx-model")

    train_embeds=base_model.predict(data['x_train'])
    mlflow.log_metric("Training Silhouette",silhouette_score(train_embeds,data['y_train']))
    tsne = TSNE()
    tsne_embeds = tsne.fit_transform(train_embeds[:5000])
    fig=helper.scatter(tsne_embeds, data['y_train'][:5000])
    mlflow.log_figure(fig, "train_scatter.png")

    calibration_embeds=base_model.predict(data['x_validation'])
    mlflow.log_metric(r"Validation Silhouette",silhouette_score(calibration_embeds,data['y_validation']))
    tsne_embeds = tsne.fit_transform(calibration_embeds)
    fig=helper.scatter(tsne_embeds, data['y_validation'])
    mlflow.log_figure(fig, "validation_scatter.png")

    test_embeds=base_model.predict(data['x_test'])
    mlflow.log_metric("Test Silhouette",silhouette_score(test_embeds,data['y_test']))
    tsne_embeds = tsne.fit_transform(test_embeds)
    fig=helper.scatter(tsne_embeds, data['y_test'])
    mlflow.log_figure(fig, "test_scatter.png")
    return train_embeds,calibration_embeds,test_embeds

if __name__ == "__main__":
    num_classes=len(set(data['y_train']))

    params=[
    {'classifier_embeddings':8,
    'num_classes':num_classes},
    {'classifier_embeddings':16,
    'num_classes':num_classes},
    {'classifier_embeddings':32,
    'num_classes':num_classes},
    {'classifier_embeddings':64,
    'num_classes':num_classes},
    {'classifier_embeddings':128,
    'num_classes':num_classes},
    {'classifier_embeddings':256,
    'num_classes':num_classes},
    {'classifier_embeddings':512,
    'num_classes':num_classes}
    ]

    mlflow.create_experiment('exp1', artifact_location='s3://mlflow')
    mlflow.set_experiment('exp1')
    for p in params:
        tf.keras.backend.clear_session()
        with mlflow.start_run() as run:
            train_embeds,calibration_embeds,test_embeds=train(data,p)
            calibration_nc,centroids=helper.nearest_centroid_NCM(data,train_embeds,calibration_embeds,p['num_classes'],p['classifier_embeddings'])
            p_values=helper.compute_pvalues(data,calibration_nc,centroids,test_embeds)
            helper.plot_efficiency_calibration(p_values,data['y_test'],0.001,0.001,0.2)