

import sys
import os
import time
import einops
import pickle
import random
import numpy

import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow import keras
from keras.models import Model
from keras import optimizers
from tensorflow.keras import layers
from official.nlp import optimization
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report


class VQA_DataLoader():

    BATCH_SIZE = 32
    IMAGE_SIZE = (224, 224)
    IMAGE_SHAPE = (224, 224, 3)
    SENTENCE_EMB_SHAPE = (384)
    IMAGES_PATH = r"C:\Users\Staff\Downloads\AssessmentI-tem1\vaq2.0.dataset-cmp9137-item1\val2014-resised"
    IMAGES_PATH = r"E:\UOL\Machine learning\Assignment\vaq2.0.dataset-cmp9137-item1\val2014-resised"
    train_data_file = IMAGES_PATH+"\\..\\vaq2.0.TrainImages.txt"
    dev_data_file = IMAGES_PATH+"\\..\\vaq2.0.DevImages.txt"
    test_data_file = IMAGES_PATH+"\\..\\vaq2.0.TestImages.txt"
    sentence_embeddings_file = "vaq2.cmp9137.sentence_transformers.txt"
    sentence_embeddings = {}
    train_ds = None
    val_ds = None
    test_ds = None
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self):
        self.sentence_embeddings = self.load_sentence_embeddings()
        self.train_ds, _g= self.load_classifier_data(self.train_data_file)
        self.val_ds, _a= self.load_classifier_data(self.dev_data_file)
        self.test_ds, self.test_labels = self.load_classifier_data(self.test_data_file)
        self.vgg_model = self.load_vgg_model()

    def load_sentence_embeddings(self):
        sentence_embeddings = {}
        print("READING sentence embeddings...")
        with open(self.sentence_embeddings_file, 'rb') as f:
            data = pickle.load(f)
            for sentence, dense_vector in data.items():
                sentence_embeddings[sentence] = dense_vector
        print("Done reading sentence_embeddings!")
        return sentence_embeddings
    

    #LOAD VGG MODEL
    def load_vgg_model(self):
        vgg_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=self.IMAGE_SHAPE)
        for layer in vgg_model.layers:
            layer.trainable = False
        return vgg_model

    def process_input(self, img_path, dense_vector, text, label):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.IMAGE_SIZE)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.cast(img, tf.float32) / 255
        features = {}
        features["image_input"] = img
        features["text_embedding"] = dense_vector
        features["question_answer"] = text
        features["file_name"] = img_path
        features["text_input"] = text  # add text_input key
        return features, label

    
    def load_classifier_data(self, data_files):
        print("LOADING data from "+str(data_files))
        print("=========================================")
        image_data = []
        text_data = []
        embeddings_data = []
        label_data = []
        label_data_str = []

        # get image, text, label of image_files
        with open(data_files) as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip("\n")
                img_name, question_answer = line.split("\t")
                img_name = img_name.split("#")[0]
                img_name = os.path.join(self.IMAGES_PATH, img_name.strip())

                # get binary labels from yes/no answers
                label = [1, 0] if question_answer.endswith("? yes") else [0, 1]

                # get sentence embeddings WITHOUT answers
                text = question_answer.replace("? yes", "?")
                text = text.replace("? no", "?")
                text_sentence_embedding = self.sentence_embeddings[text]
                text_sentence_embedding = tf.constant(text_sentence_embedding)

                image_data.append(img_name)
                embeddings_data.append(text_sentence_embedding)
                text_data.append(question_answer)
                label_data.append(label)
                str_label = str(label).replace(',','')
                label_data_str.append(str(str_label))

        print("|image_data|="+str(len(image_data)))
        print("|text_data|="+str(len(text_data)))
        print("|label_data|="+str(len(label_data)))

        # prepare a tensorflow dataset using the lists generated above
        dataset = tf.data.Dataset.from_tensor_slices((image_data, embeddings_data, text_data, label_data))
        dataset = dataset.shuffle(self.BATCH_SIZE * 8)
        dataset = dataset.map(self.process_input, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.batch(self.BATCH_SIZE).prefetch(self.AUTOTUNE)
        self.print_data_samples(dataset)
        return dataset, label_data_str

    def print_data_samples(self, dataset):
        print("PRINTING data samples...")
        print("-----------------------------------------")
        for features_batch, label_batch in dataset.take(1):
            for i in range(1):
                print(f'Image pixels: {features_batch["image_input"]}')
                print(f'Sentence embeddings: {features_batch["text_embedding"]}')
                print(f'Question-answer: {features_batch["question_answer"].numpy()}')
                label = label_batch.numpy()[i]
                print(f'Label : {label}')
        print("-----------------------------------------")

class VQA_Classifier(VQA_DataLoader):
    epochs = 1
    learning_rate = 3e-4
    model_arch_file = 'model_arch_vqa_baseline.png'
    class_names = {'yes', 'no'}
    num_classes = len(class_names)
    classifier_model = None
    history = None
    classifier_model_name = 'VQA_Classifier'
    AUTOTUNE = tf.data.AUTOTUNE
    total_predictions = []
	
    def __init__(self):
        super().__init__()
        self.build_vgg_model()
        self.train_classifier_model()
        self.test_classifier_model()
        self.plot_learning_curves()

    def project_embeddings(self, embeddings, num_projection_layers, projection_dims, dropout_rate):
        projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
        for _ in range(num_projection_layers):
            x = tf.nn.gelu(projected_embeddings)
            x = layers.Dense(projection_dims)(x)
            x = layers.Dropout(dropout_rate)(x)
            x = layers.Add()([projected_embeddings, x])
            projected_embeddings = layers.LayerNormalization()(x)
        return projected_embeddings

    def create_text_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        text_input = keras.Input(shape=self.SENTENCE_EMB_SHAPE, name='text_embedding')
        print("shape text", K.int_shape(text_input))
        outputs = self.project_embeddings(text_input, num_projection_layers, projection_dims, dropout_rate)
        print("output text", K.int_shape(outputs))
        return text_input, outputs
    
    #BUILD VGG MODEL
    def build_vgg_model(self):
        # image input branch
        image_input = keras.Input(shape=(224, 224, 3), name="image_input")
        vgg = keras.applications.VGG16(
            weights="imagenet", include_top=False, input_tensor=image_input
        )
        for layer in vgg.layers:
            layer.trainable = False
        flatten = layers.Flatten()(vgg.output)

        print(f'BUILDING model')
        text_input, text_net = self.create_text_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        net = tf.keras.layers.Concatenate(axis=1)([flatten, text_net])
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.classifier_model_name)(net)
        print("output", K.int_shape(net))
        #exit()
        self.classifier_model = tf.keras.Model(inputs=[image_input, text_input], outputs=net)

        self.classifier_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=[keras.metrics.BinaryAccuracy()],
        )
        self.classifier_model.summary()

    def build_classifier_model(self):
        print(f'BUILDING model')
        img_input, vision_net = self.create_vision_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        print("image", K.int_shape(vision_net))
        text_input, text_net = self.create_text_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        net = tf.keras.layers.Concatenate(axis=1)([vision_net, text_net])
        net = tf.keras.layers.Dropout(0.1)(net)
        net = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.classifier_model_name)(net)
        print("output", K.int_shape(net))
        #exit()
        self.classifier_model = tf.keras.Model(inputs=[img_input, text_input], outputs=net)
        #self.classifier_model.summary()

    def train_classifier_model(self):
        print(f'TRAINING model')
        steps_per_epoch = tf.data.experimental.cardinality(self.train_ds).numpy()
        num_train_steps = steps_per_epoch * self.epochs
        num_warmup_steps = int(0.2*num_train_steps)

        loss = tf.keras.losses.KLDivergence()
        metrics = tf.keras.metrics.BinaryAccuracy()
        optimizer = optimization.create_optimizer(init_lr=self.learning_rate,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

        self.classifier_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # the next two lines if you want the neural architecture to be generated in a file, which 
		# requires installing graphviz (from https://www.graphviz.org/download/) and pip install pydot       
        #tf.keras.utils.plot_model(self.classifier_model, to_file=self.model_arch_file, show_shapes=False, show_layer_names =True)
        #print("File %s created!" % (self.model_arch_file))

        # uncomment the next line if you wish to make use of early stopping during training
        #callbacks = [tf.keras.callbacks.EarlyStopping(patience=11, restore_best_weights=True)]

        self.history = self.classifier_model.fit(x=self.train_ds, validation_data=self.val_ds, epochs=self.epochs)#, callbacks=callbacks)
        print("model trained!")

    def test_classifier_model(self):
        print("TESTING classifier model (showing a sample of question-answer-predictions)...")
        num_classifications = 0
        num_correct_predictions = 0

        # read test data for VQA classification
        for features, groundtruth in self.test_ds:
            groundtruth = groundtruth.numpy()
            predictions = self.classifier_model(features)
            predictions = predictions.numpy()
            question_answers = features["question_answer"].numpy()
            file_names = features["file_name"].numpy()

            # read test data per batch
            for batch_index in range(0, len(groundtruth)):
                predicted_values = predictions[batch_index]
                probability_yes = predicted_values[0]
                probability_no = predicted_values[1]
                predicted_class = "[1 0]" if probability_yes > probability_no else "[0 1]"
                self.total_predictions.append(predicted_class)
                if str(groundtruth[batch_index]) == predicted_class: 
                    num_correct_predictions += 1
                num_classifications += 1

                # print a sample of predictions, e.g. 10% of all possible
                if random.random() < 0.1:
                    question_answer = question_answers[batch_index]
                    file_name = file_names[batch_index].decode("utf-8")
                    file_name_index = file_name.index("COCO") 
                    file_name = file_name[file_name_index:]
                    print("QA=%s PREDICTIONS: yes=%s, no=%s \t -> \t %s" % \
                             (question_answer, probability_yes, probability_no, file_name))
                
        # reveal test performance using our own calculations above
        accuracy = num_correct_predictions/num_classifications
        print("TEST accuracy=%4f" % (accuracy))

        # reveal test performance using Tensorflow calculations
        loss, accuracy = self.classifier_model.evaluate(self.test_ds)
        print(f'Tensorflow test method: Loss: {loss}; ACCURACY: {accuracy}')

        report = classification_report(self.test_labels,self.total_predictions)
        #df = pd.DataFrame(report).transpose()
        #df.to_csv(self.report_name, index= True)
        print(report)
        print()
        print("The confusion matrix: ")
        print(confusion_matrix(self.test_labels,self.total_predictions))


    def plot_learning_curves(self):
        history_dict = self.history.history
        print("history_dict="+str(history_dict))
        acc = history_dict['binary_accuracy']
        val_acc = history_dict['val_binary_accuracy']
        train_loss = history_dict['loss']
        test_loss =  history_dict['val_loss']

        epochs = range(1, len(acc) + 1)
        fig = plt.figure(figsize=(10, 6))
        fig.tight_layout()

        plt.plot(epochs, acc, 'r', label='Training ACC', linestyle='solid')
        plt.plot(epochs, val_acc, 'b', label='Validation ACC', linestyle='dashdot')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(loc='lower right')
        plt.legend(fontsize=14)
        plt.show()

        fig1 = plt.figure(figsize=(10, 6))
        fig1.tight_layout()
        plt.plot(epochs, train_loss, 'g', label='Training Loss', linestyle='solid')
        plt.plot(epochs, test_loss, 'y', label='Validation Loss', linestyle='dashdot')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend(loc='lower right')
        plt.legend(fontsize=14)
        plt.show()

d = VQA_Classifier()

