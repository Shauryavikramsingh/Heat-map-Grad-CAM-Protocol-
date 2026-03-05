Protocol 04: Grad-CAM Explainability (XAI)
In medical diagnostics, a "Black Box" prediction is a liability. MedRadio solves this by implementing Gradient-weighted Class Activation Mapping (Grad-CAM).

Why this is Rare:
Most AI projects only provide a probability score (e.g., Pneumonia: 92%). Our protocol forces the neural network to "justify" its decision by highlighting the specific spatial features that triggered the classification.

The Technical Mechanism:
Gradient Tracking: Uses tf.GradientTape to calculate the importance of the last convolutional layer.

Feature Mapping: It identifies which pixels in the X-ray had the highest positive influence on the "Pneumonia" prediction.

Thermal Superimposition: The result is a JET-Map (heatmap) overlaid on the original X-ray, allowing doctors to visually verify the infected zones.

3. The "Patent-Ready" Code Block
This is the clean Markdown version of your rare code for the README:

Python
# --- THE GRAD-CAM HEATMAP PROTOCOL ---
# High-transparency logic for Medical-Legal Evidence

def generate_heatmap(model, x_ray_image, layer_name):
    # 1. Create a sub-model to extract gradients
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    # 2. Track the "Diagnostic Intent"
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x_ray_image)
        loss = predictions[:, np.argmax(predictions[0])]

    # 3. Calculate Neural Importance (Gradients)
    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # 4. Generate the Explainable Heatmap
    heatmap = conv_outputs[0] @ weights[..., tf.newaxis]
    return np.maximum(heatmap, 0) / np.max(heatmap)
