import numpy as np
import tensorflow as tf
import matplotlib.cm as cm
import cv2
import os
from PIL import Image

def get_img_array(pil_img, size):
    pil_img = pil_img.resize(size)  # Resize the already loaded PIL image
    array = tf.keras.preprocessing.image.img_to_array(pil_img)
    array = np.expand_dims(array, axis=0)
    return array / 255.0

    

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(pil_img, heatmap, alpha=0.4):
    import matplotlib.cm as cm

    # Resize heatmap to match image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(pil_img.size)
    heatmap = np.array(heatmap)

    # Convert to RGB if not already
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    # Apply colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = np.uint8(jet_heatmap * 255)

    jet_heatmap = Image.fromarray(jet_heatmap).convert("RGB")

    # Superimpose heatmap on image
    superimposed_img = Image.blend(pil_img, jet_heatmap, alpha)
    return superimposed_img