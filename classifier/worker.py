import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import paho.mqtt.client as mqtt
from joblib import load

import shared_state
from cocoa_classifier import predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MQTT_BROKER = os.getenv("MQTT_BROKER", "")
if not MQTT_BROKER:
    raise RuntimeError("MQTT_BROKER environment variable not set")


MQTT_PORT = 1883
MQTT_TOPIC = "image/transfer"
MQTT_RESULT_TOPIC = "cocoa/results"


def load_model():
    try:
        model_dir = Path("models/svm_v1")
        model = load("models/svm_v1/model.pkl")
        with open(model_dir / "classes.json") as f:
            classes = json.load(f)

        shared_state.model = model
        shared_state.classes = classes
        logger.info("Model loaded successfully")
        return model, classes
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info(f"Connected to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC)
        logger.info(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        logger.error(f"Failed to connect to MQTT broker. Return code: {rc}")


def on_message(client, userdata, msg):
    try:
        logger.info(f"Received message on topic: {msg.topic}")

        image_data = msg.payload

        logger.info("Running prediction...")
        overlay, results = predictor.predict(
            file=image_data,
            model=shared_state.model,
            classes=shared_state.classes,
            single_bean=False,
        )

        logger.info(f"Prediction completed. Found {len(results)} objects")

        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        overlay_filename = f"overlay_{timestamp}.jpg"
        overlay_path = runs_dir / overlay_filename

        cv2.imwrite(str(overlay_path), overlay)
        logger.info(f"Overlay saved to {overlay_path}")

        result_payload = {
            "topic": msg.topic,
            "results": results,
            "count": len(results),
            "overlay_path": str(overlay_path),
        }

        client.publish(MQTT_RESULT_TOPIC, json.dumps(result_payload), qos=1)
        logger.info(f"Results published to {MQTT_RESULT_TOPIC}")

        for result in results:
            logger.info(
                f"  Bean {result['idx']}: {result['pred_class']} "
                f"(confidence: {result['confidence']:.2f})"
            )

    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)


def on_disconnect(client, userdata, rc):
    if rc != 0:
        logger.warning(f"Unexpected disconnection. Return code: {rc}")
    else:
        logger.info("Disconnected from MQTT broker")


def start_mqtt_worker():
    logger.info("Starting MQTT worker...")

    load_model()

    client = mqtt.Client(client_id="cocoa_classifier_worker")

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        logger.info(f"Connecting to MQTT broker at {MQTT_BROKER}:{MQTT_PORT}")
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

        logger.info("Starting MQTT loop...")
        client.loop_forever()

    except KeyboardInterrupt:
        logger.info("Shutting down MQTT worker...")
        client.disconnect()
    except Exception as e:
        logger.error(f"Error in MQTT worker: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    start_mqtt_worker()
