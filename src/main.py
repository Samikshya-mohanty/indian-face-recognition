"""Run the Indian Face Recognition demo locally with Gradio.

Before running, place the fine-tuned model weights and the MTCNN test
split on disk (see README "Running Locally" section):

    models/facenet_finetuned_best.keras
    data/mtcnn_splits/test/<actor>/*.jpg
"""

import os
from pathlib import Path

import gradio as gr
import numpy as np
import tensorflow as tf
from keras_facenet import FaceNet
from mtcnn import MTCNN
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers, models
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "facenet_finetuned_best.keras"
TEST_DIR = ROOT_DIR / "data" / "mtcnn_splits" / "test"

IMG_SIZE = (160, 160)
NUM_CLASSES = 100
MAX_REFS_PER_ACTOR = 10


def build_finetuned_model():
    facenet_base = FaceNet().model
    facenet_base.trainable = True
    for layer in facenet_base.layers[:-30]:
        layer.trainable = False

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = facenet_base(inputs, training=False)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.load_weights(str(MODEL_PATH))
    return model


def build_reference_database(embedding_model, class_names):
    reference_embeddings = {}
    for actor in tqdm(class_names, desc="Building reference DB"):
        actor_dir = TEST_DIR / actor
        img_files = [
            f for f in os.listdir(actor_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ][:MAX_REFS_PER_ACTOR]

        embeds = []
        for img_file in img_files:
            img = Image.open(actor_dir / img_file).resize(IMG_SIZE)
            arr = np.expand_dims(np.array(img) / 255.0, axis=0)
            embeds.append(embedding_model.predict(arr, verbose=0)[0])

        if embeds:
            reference_embeddings[actor] = np.mean(embeds, axis=0)

    return reference_embeddings


def make_predict_fn(detector, embedding_model, reference_embeddings):
    def predict_actor(image):
        detections = detector.detect_faces(image)
        if not detections:
            return "❌ No face detected. Try a clearer or more frontal photo.", {}

        best = max(detections, key=lambda d: d["confidence"])
        x, y, w, h = best["box"]

        margin = int(0.2 * max(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(image.shape[1], x + w + margin)
        y2 = min(image.shape[0], y + h + margin)

        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return "❌ Face crop failed. Please try another photo.", {}

        img = Image.fromarray(face_crop).resize(IMG_SIZE)
        arr = np.expand_dims(np.array(img) / 255.0, axis=0)
        query_embed = embedding_model.predict(arr, verbose=0)[0]

        scores = {
            actor: float(cosine_similarity([query_embed], [ref])[0][0])
            for actor, ref in reference_embeddings.items()
        }
        top5 = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5]
        result = dict(top5)
        top_name, top_score = top5[0]

        if top_score >= 0.6:
            verdict = f"✅ Identified: {top_name} ({top_score:.2f} similarity)"
        elif top_score >= 0.4:
            verdict = f"🤔 Possibly: {top_name} ({top_score:.2f} similarity)"
        else:
            verdict = f"❓ Unknown or low confidence ({top_score:.2f} similarity)"

        return verdict, result

    return predict_actor


def main():
    print("Loading fine-tuned FaceNet model...")
    model = build_finetuned_model()
    embedding_model = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)

    class_names = sorted(os.listdir(TEST_DIR))
    reference_embeddings = build_reference_database(embedding_model, class_names)

    detector = MTCNN()
    predict_actor = make_predict_fn(detector, embedding_model, reference_embeddings)

    demo = gr.Interface(
        fn=predict_actor,
        inputs=gr.Image(type="numpy", label="Upload a face photo"),
        outputs=[
            gr.Textbox(label="Prediction"),
            gr.Label(num_top_classes=5, label="Top 5 matches (cosine similarity)"),
        ],
        title="Indian Face Recognition System",
        description=(
            "Upload a photo of an Indian actor. MTCNN detects the face → "
            "FaceNet extracts embedding → Cosine similarity finds the best match."
        ),
        theme=gr.themes.Soft(),
    )
    demo.launch()


if __name__ == "__main__":
    main()
