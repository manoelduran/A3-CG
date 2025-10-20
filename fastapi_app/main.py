import os
from contextlib import asynccontextmanager

import paho.mqtt.client as mqtt
from fastapi import FastAPI
from paho.mqtt.enums import CallbackAPIVersion

BROKER_HOST = os.getenv("MQTT_BROKER_HOST", "mosquitto")
BROKER_PORT = 1883
MQTT_TOPIC = "image/transfer"
OUTPUT_PATH = "/app/imagem_recebida.jpg"


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("API conectada ao Broker MQTT com sucesso!")
        client.subscribe(MQTT_TOPIC)
        print(f"Inscrito no tópico: '{MQTT_TOPIC}'")
    else:
        print(f"Falha ao conectar ao broker, código: {rc}")


def on_message(client, userdata, msg):
    print(f"Mensagem recebida no tópico '{msg.topic}'!")
    try:
        with open(OUTPUT_PATH, "wb") as f:
            f.write(msg.payload)
        print(f"Imagem salva com sucesso em: '{OUTPUT_PATH}' dentro do container.")
    except Exception as e:
        print(f"Ocorreu um erro ao salvar a imagem: {e}")


mqtt_client = mqtt.Client(CallbackAPIVersion.VERSION1, "FastAPI_Receiver_Client")
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message

app = FastAPI(title="MQTT Image Receiver API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Iniciando a conexão com o MQTT...")
    mqtt_client.connect(BROKER_HOST, BROKER_PORT, 60)
    mqtt_client.loop_start()

    yield

    print("Desconectando do MQTT...")
    mqtt_client.loop_stop()
    mqtt_client.disconnect()


@app.get("/")
def read_root():
    return {"status": "API está no ar e ouvindo o tópico MQTT."}
