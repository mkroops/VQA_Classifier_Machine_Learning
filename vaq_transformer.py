
import sys
import os
import time
import einops
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from official.nlp import optimization
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import classification_report

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import os
import sys
import pathlib
import matplotlib.pyplot as plt

def read_image_data(data_path, verbose=False):
    count=1
    X = []
    Y = []
    data_dir = pathlib.Path(data_path)
    print("Reading folder="+str(data_dir))
    dataset = tf.keras.utils.image_dataset_from_directory(data_dir)
    for image_batch, labels_batch in dataset:
        labels = labels_batch.numpy()
        if (verbose): print("Batch labels -> "+str(labels))
        for i in range(0, len(image_batch)):
            image = image_batch[i]
            label = labels[i]
            X.append(image)
            Y.append([label])
            if (verbose): print("["+str(count)+"] image="+str(image.shape)+" label="+str(label))
            count += 1
      
    X = np.array(X)
    Y = np.array(Y)
    class_names = dataset.class_names
    
    print("X="+str(X.shape))
    print("Y="+str(Y.shape))
    print("class_names="+str(class_names))
    
    return X, Y, class_names

IMAGES_PATH = r"E:\UOL\Machine learning\Assignment\vaq2.0.dataset-cmp9137-item1\val2014-resised"
#train_data_dir = IMAGES_PATH+"\\..\\vaq2.0.TrainImages.txt"
#test_data_dir = IMAGES_PATH+"\\..\\vaq2.0.TestImages.txt"
#x_train, y_train, class_names = read_image_data(train_data_dir)
#x_test, y_test, class_names = read_image_data(test_data_dir)


num_classes = 2
input_shape = (256,256,3)

#print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
#print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 32
num_epochs = 40
image_size = 72  # We'll resize input images to this size
patch_size = 6  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
#mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
#mlp_head_units = [1024, 768]  # Size of the dense layers of the final classifier
mlp_head_units = [512, 128]  # Size of the dense layers of the final classifier

data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
#data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
    
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
        
import matplotlib.pyplot as plt

'''plt.figure(figsize=(4, 4))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))
plt.axis("off")

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(4, 4))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")
'''

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
        
def create_vit_classifier(inputs):
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    #patches = Patches(patch_size)(inputs)
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    #model = keras.Model(inputs=inputs, outputs=logits)
    #model.summary()
    return inputs, logits

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


    def __init__(self):
        self.sentence_embeddings = self.load_sentence_embeddings()
        self.train_ds, _g= self.load_classifier_data(self.train_data_file)
        self.val_ds, _a= self.load_classifier_data(self.dev_data_file)
        self.test_ds, self.test_labels = self.load_classifier_data(self.test_data_file)

    # Sentence embeddings are dense vectors representing text data, one vector per sentence. 
    # Sentences with similar vectors would mean sentences with equivalent meanning.  
	# They are useful here to provide text-based features of questions in the data.
    # Note: sentence embeddings don't include answers, they are solely based on the question.
    def load_sentence_embeddings(self):
        sentence_embeddings = {}
        print("READING sentence embeddings...")
        with open(self.sentence_embeddings_file, 'rb') as f:
            data = pickle.load(f)
            for sentence, dense_vector in data.items():
                sentence_embeddings[sentence] = dense_vector
        print("Done reading sentence_embeddings!")
        return sentence_embeddings

    # In contrast to text-data based on pre-trained features, image data does not use
    # any form of pre-training in this program. Instead, it makes use of raw pixels.
    # Notes: 
	#       (1) The question and answer are not used for training, only for testing.
    #       (2) Input features to the classifier are only pixels and sentence embeddings.
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
        return features, label

    # This method load the multimodal data, which comes from the following sources:
    # (1) image files in IMAGES_PATH, and (2) files with pattern vaq2.0.*Images.txt
    # The data is stored in a tensorflow data structure to make it easy to use by
    # the tensorflow model during training, validation and test.
    # This method was carefully prepared to load the data rapidly, i.e., by loading
    # already creating sentence embeddings rather than creating them at runtime.
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
        #str_label = str(label).replace(',','')  
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
    learning_rate = 3e-5
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
        self.build_classifier_model()
        self.train_classifier_model()
        self.test_classifier_model()
        self.plot_learning_curves()

    def create_vision_encoder(self, num_projection_layers, projection_dims, dropout_rate):
        img_input = layers.Input(shape=self.IMAGE_SHAPE, name="image_input")
        cnn_layer = layers.Conv2D(16, 3, padding='same', activation='relu')(img_input)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(32, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Conv2D(64, 3, padding='same', activation='relu')(cnn_layer)
        cnn_layer = layers.MaxPooling2D()(cnn_layer)
        cnn_layer = layers.Dropout(dropout_rate)(cnn_layer)
        cnn_layer = layers.Flatten()(cnn_layer)
        outputs = self.project_embeddings(cnn_layer, num_projection_layers, projection_dims, dropout_rate)
        return img_input, outputs

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
        outputs = self.project_embeddings(text_input, num_projection_layers, projection_dims, dropout_rate)
        return text_input, outputs

    def build_classifier_model(self):
        print(f'BUILDING model')
        img_input = keras.Input(shape=(224, 224, 3), name="image_input")

        #calling transformer function for image classification
        img_input, vision_net = create_vit_classifier(img_input)
        text_input, text_net = self.create_text_encoder(num_projection_layers=1, projection_dims=128, dropout_rate=0.1)
        net = tf.keras.layers.Concatenate(axis=1)([vision_net, text_net])
        net = tf.keras.layers.Dropout(0.1)(net)
        #softmax functionm
        net = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.classifier_model_name)(net)
        #relu activation function
        # net = tf.keras.layers.Dense(self.num_classes, activation='softmax', name=self.classifier_model_name)(net)
        self.classifier_model = tf.keras.Model(inputs=[img_input, text_input], outputs=net)
        self.classifier_model.summary()
	
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

        report = classification_report(self.test_labels,self.total_predictions,output_dict=True)
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

d = VQA_Classifier()










