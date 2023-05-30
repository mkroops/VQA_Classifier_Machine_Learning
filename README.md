# VQA_Classifier_Machine_Learning

VQA combines advanced methods of computer vision and
natural language processing to develop a system that can
answer to a question about an image which has attracted a lot
of interest and enthusiasm in recent years. With the help of
the existing research different deep learning techniques such
as transformers and convolutional neural networks has been
used to solve this task

Using a pretrained sentence transformer model, Create
Text Encoder() method in the VQA baseline generated text
features by computing sentence embeddings. The outcomes
of the sentence transformer model are stored in the sentence
embedding dictionary. This can be used as as input feature
in tf.keras.layers.Concatenate() function model to predict an-
swers to questions about visual content. Hyperparameters like
dropout rate and projection layers were tested using Create
text encoder() function. Projection layers helped in reducing
the dimensionality of the embeddings. Higher projection layers
solve complex representations but run the danger of overfitting.
An increased dropout rate helped to regularize the model and
avoided overfitting.

A. VGG 16
VGG-16 is a deep convolutional neural network consists
of 13 convolutional layers and 3 fully connected layers.
Convolutional layers are used to extract features and fully
connected layers are used for classification. Because it is
a pretrained model, it has previously been trained on a sizable
set of image data and may be used as a baseline to train on
further data. This architecture has input size of 224*224 pixels
and also uses smaller filter size of 3 x 3 pixels.
B. DenseNet
DenseNet is a deep convolutional neural network used
for image recognition tasks. It is unique because it uses
dense connections between the layers, which are called Dense
Blocks. In a Dense Block, each layer is directly connected
to all other layers that have matching feature-map sizes [5].
This means that every layer gets additional inputs from all the
layers that came before it, and also passes on its own feature-
maps to all the layers that come after it. This helps to preserve
the feed-forward nature of the network, which is important for
efficient computation and training .
C. Transformer
Transformer is a neural network architecture that is widely
used in natural language processing which uses an attention
mechanism to weigh the importance of different parts of the
input sequence and combine them to create the output se-
quence. It has an advantage over CNNs for tasks involving
images since they are not constrained by their convolutional
filters and can record more intricate correlations between the
picture components. Transformers are also more adaptable and
have the ability to accept inputs of different sizes and aspect
ratios making them ideal for image collections with a variety
of image dimensions.
