from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np


def focal_loss(gamma=2., alpha=0.25):
    
    def focal_loss_fixed(y_true, y_pred):
        epsilon = 1e-7  
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(tf.reduce_sum(loss, axis=1))
    return focal_loss_fixed


def create_custom_resnet():
    base_model = ResNet50(
        weights='imagenet', 
        include_top=False, 
        input_shape=(224, 224, 3)
    )
    
    
    for layer in base_model.layers[-20:]:
        layer.trainable = True
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)  
    predictions = Dense(4, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=focal_loss(gamma=2, alpha=0.25),  
        metrics=['accuracy']
    )
    return model