from utils import *
from model_utils import *

# Loads dataset
X, Y = load_mnist()

# Plotting tools
# 1. sshuffle_plot(X[img_num], height= kernal_sz, width= kernal_sz) 
#    makes a shuffled instance of the image with kernal_sz as patch dims
#  
# 2. generate_patch_cm(X[img_num], kernal_sz, kernal_sz), makes the 
#    confidence maps for image with kernal_sz as patch dims

# Creating model where num_channels is the numebr of patches
m = build_unet((28, 28, 1), num_channels=4)

# Generates shuffled training set
X_s, label_s = generate_shuffled(X, patch_dim=2,return_label=True)

# Training set up
X_train, X_test, X_dev = X_s[:65000], X_s[65000:69500], X_s[69500:]
Y_train, Y_test, Y_dev = label_s[:65000], label_s[65000:69500], label_s[69500:]

X_train = tf.convert_to_tensor(X_train)
X_test = tf.convert_to_tensor(X_test)
Y_train = tf.convert_to_tensor(Y_train)
Y_test = tf.convert_to_tensor(Y_test)

Y_train = np.reshape(Y_train, (np.shape(Y_train)[0], 28, 28, 4))
Y_test = np.reshape(Y_test, (np.shape(Y_test)[0], 28, 28, 4))

m.compile(optimizer="Adam", loss="mse", metrics=["mae"])
m.fit(x=X_train,
      y=Y_train,
      epochs=5,
      validation_data=(X_test, Y_test))

# Visual Checker for reassembling predictions
X_t = tf.convert_to_tensor(X_dev)
cm = m.predict(X_t)
cm = np.reshape(cm, (np.shape(X_dev)[0], 4, 28, 28))

img_num = 2

r = reassembler(img=X_dev[img_num], cms=cm[img_num], height=2, width=2)
plot_image(X[69500+img_num], title="Original")
plot_image(X_dev[img_num], title="Shuffled")
plot_image(r, title="reassembled")