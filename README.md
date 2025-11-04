# Projeto A3 - Computação Gráfica

## Sistema de Classificação de Grãos de Cacau

---

## 1. Visão Geral do Projeto

Este projeto implementa um sistema completo de classificação automática de grãos de cacau utilizando técnicas de visão computacional e aprendizado de máquina. O sistema é capaz de identificar e classificar grãos individuais em imagens, distinguindo entre grãos de boa qualidade (`good`) e grãos defeituosos (`bad`).

O projeto é composto por dois componentes principais:

- **Classifier**: Módulo de processamento de imagens e classificação usando SVM
- **Mosquitto**: Broker MQTT para comunicação assíncrona

A arquitetura permite processamento assíncrono de imagens através de mensageria, facilitando a integração com sistemas de produção ou linhas de inspeção automatizadas.

---

## 2. Arquitetura do Sistema

### 2.1 Componentes Principais

```
┌─────────────┐
│   Cliente   │─── Publica imagem ───┐
└─────────────┘                      │
                                     ▼
                            ┌─────────────────┐
                            │  Mosquitto MQTT  │
                            │     Broker       │
                            └─────────────────┘
                                     │
                                     ▼
                            ┌──────────────────────┐
                            │  Classifier Worker    │
                            └──────────────────────┘
                                     │
                                     ▼
                            Publica resultados em
                            tópico 'cocoa/results'
```

### 2.2 Tecnologias Utilizadas

- **Python 3.12**: Linguagem principal
- **OpenCV**: Processamento de imagens e visão computacional
- **scikit-learn**: Machine learning (SVM)
- **MQTT (Mosquitto)**: Protocolo de mensageria
- **Docker & Docker Compose**: Containerização e orquestração
- **NumPy**: Computação numérica
- **Pydantic**: Validação de dados

---

## 3. Detalhamento do Classifier

O módulo `classifier` é o núcleo do sistema, responsável por todo o processamento de imagens e classificação. É composto por vários módulos especializados que trabalham em conjunto.

### 3.1 Estrutura de Módulos

#### 3.1.1 `bean_segmenter.py` - Segmentação de Grãos

Este módulo implementa algoritmos avançados de segmentação para identificar e isolar grãos individuais em imagens.

**Funcionalidades principais:**

1. **Segmentação Multi-Grão (`segment_beans`)**:

   - Utiliza o algoritmo **Watershed** para separar grãos que estão em contato
   - Pré-processamento com normalização de contraste (CLAHE no espaço LAB)
   - Binarização adaptativa usando Threshold de Otsu
   - Operações morfológicas (opening, dilatação)
   - Transformada de distância para identificar centros dos grãos
   - Separação de regiões "sure foreground", "sure background" e "unknown"
   - Aplicação do Watershed para separação final

2. **Segmentação de Grão Único (`segment_single_bean`)**:
   - Algoritmo simplificado para imagens de treinamento
   - Threshold simples com operações morfológicas
   - Filtragem por área para remover ruídos

**Parâmetros configuráveis (`SegmentParams`):**

- `min_area`: Área mínima para considerar um contorno válido
- `max_area`: Área máxima permitida
- `open_ksize`: Tamanho do kernel para operação de opening morfológico
- `sure_bg_dilate`: Tamanho da dilatação para background certo
- `distance_thresh`: Threshold para transformada de distância

**Algoritmo Watershed em detalhes:**
O Watershed é particularmente eficaz para separar objetos em contato. O processo funciona assim:

1. Identifica regiões que são definitivamente foreground (centros dos grãos)
2. Identifica regiões que são definitivamente background
3. Marca a região desconhecida entre elas
4. Aplica o algoritmo Watershed que "inunda" a partir dos marcadores para criar divisões precisas

#### 3.1.2 `feature_contourer.py` - Extração de Características

Extrai um conjunto robusto de features de cada grão segmentado para alimentar o classificador.

**Features extraídas (12 dimensões):**

**Features geométricas:**

1. **Área**: Número de pixels do contorno
2. **Perímetro**: Comprimento do contorno
3. **Aspect Ratio**: Razão largura/altura do retângulo delimitador
4. **Circularidade**: `4π × área / perímetro²` - mede quão circular é o grão
5. **Solidez (Solidity)**: Razão entre área do contorno e área do convex hull
6. **Excentricidade**: Medida de alongamento (baseada em elipse ajustada)

**Features de cor (espaço LAB):** 7. **L (Luminosidade) - média e desvio padrão** 8. **A (verde-vermelho) - média e desvio padrão** 9. **B (azul-amarelo) - média e desvio padrão**

O espaço de cores LAB é escolhido porque é perceptualmente uniforme e separa bem a luminosidade da informação de cor, sendo ideal para análise de qualidade de produtos agrícolas.

#### 3.1.3 `data_loader.py` - Carregamento de Dados de Treinamento

Responsável por carregar e processar as imagens de treinamento.

**Processo:**

1. Percorre diretórios de classes (`good/`, `bad/`)
2. Para cada imagem:
   - Decodifica a imagem
   - Segmenta o grão (assumindo imagem de grão único)
   - Extrai features do maior contorno encontrado
   - Associa label da classe
3. Retorna matriz de features e vetor de labels

**Fallback:** Se a segmentação padrão falhar, utiliza threshold simples para encontrar contornos.

#### 3.1.4 `trainer.py` - Treinamento do Modelo

Implementa o pipeline de treinamento do classificador SVM.

**Pipeline de Machine Learning:**

1. **StandardScaler**: Normalização das features (média=0, desvio=1)
2. **SVM (Support Vector Machine)**:
   - Kernel: RBF (Radial Basis Function)
   - `probability=True`: Permite obter probabilidades, não apenas classes
   - `C=10.0`: Parâmetro de regularização
   - `gamma='scale'`: Escala automática do parâmetro gamma

**Validação Cruzada:**

- Utiliza `StratifiedKFold` para garantir proporção de classes em cada fold
- Número de folds adaptativo baseado no tamanho da menor classe
- Exibe métricas de acurácia média e desvio padrão

**Saída:**

- `model.pkl`: Modelo treinado serializado (joblib)
- `classes.json`: Mapeamento de índices para nomes de classes

#### 3.1.5 `predictor.py` - Predição

Orquestra o processo completo de predição em imagens.

**Fluxo de predição:**

1. Decodifica imagem de entrada
2. Segmenta todos os grãos na imagem
3. Para cada grão:
   - Extrai features
   - Executa predição com probabilidades
   - Obtém classe predita e confiança
   - Desenha retângulo delimitador e label na imagem overlay
4. Retorna imagem anotada e lista de resultados

**Visualização:**

- Retângulos verdes ao redor de cada grão detectado
- Labels com classe e confiança sobre cada grão
- Resultados exportados em CSV com coordenadas e predições

#### 3.1.6 `helpers.py` - Funções Auxiliares

Conjunto de funções utilitárias para conversão de espaços de cor e pré-processamento:

- Conversão BGR ↔ LAB ↔ Grayscale
- Aplicação de blur gaussiano
- Normalização de contraste com CLAHE

#### 3.1.7 `cli.py` - Interface de Linha de Comando

Interface principal para interação com o sistema.

**Comandos disponíveis:**

1. **`train`**:

   ```bash
   python cli.py train --data-dir data/train --out-dir model/svm_v1
   ```

   - Treina modelo usando imagens do diretório especificado
   - Salva modelo e classes no diretório de saída

2. **`predict`**:
   ```bash
   python cli.py predict --image path/to/image.jpg --model-dir model/svm_v1 --out-dir runs/
   ```
   - Carrega modelo pré-treinado
   - Processa imagem de entrada
   - Gera imagem anotada e CSV com resultados

#### 3.1.8 `worker.py` - Worker MQTT

Implementa processamento assíncrono via MQTT.

**Funcionalidades:**

- Conecta ao broker MQTT configurado
- Subscreve ao tópico `image/transfer`
- Carrega modelo uma vez na inicialização (via `shared_state`)
- Processa cada imagem recebida:
  - Executa predição
  - Salva overlay com timestamp
  - Publica resultados no tópico `cocoa/results`
- Logging detalhado de todas as operações

**Estado compartilhado (`shared_state.py`):**

- Armazena modelo e classes em memória para evitar recarregamento a cada mensagem

### 3.2 Configuração e Schemas

#### 3.2.1 `schemas/classifier_config.py`

Define estruturas de configuração usando Pydantic:

- `TrainConfig`: Parâmetros para treinamento
- `PredictConfig`: Parâmetros para predição
- `ClassifierConfig`: Configuração principal com parser de argumentos

Validação automática de tipos e paths garantem robustez na interface CLI.

### 3.3 Dataset de Treinamento

O sistema utiliza um dataset estruturado:

```
data/train/
├── good/    (458 imagens)
└── bad/     (458 imagens)
```

Cada classe contém 458 imagens de grãos individuais, totalizando 916 amostras de treinamento. As imagens são processadas automaticamente para extrair features e treinar o modelo.

---

## 4. Pipeline Completo de Processamento

### 4.1 Fluxo de Treinamento

```
Imagens de Treinamento
    ↓
[data_loader.py]
    ├─ Segmentação de grão único
    ├─ Extração de features (12 dims)
    └─ Associação de labels
    ↓
Matriz de Features + Labels
    ↓
[trainer.py]
    ├─ Normalização (StandardScaler)
    ├─ Validação cruzada
    ├─ Treinamento SVM
    └─ Serialização do modelo
    ↓
Modelo Treinado (model.pkl + classes.json)
```

### 4.2 Fluxo de Predição

```
Imagem de Entrada
    ↓
[predictor.py]
    ├─ Decodificação
    ↓
[bean_segmenter.py]
    ├─ Normalização de contraste (CLAHE)
    ├─ Binarização (Otsu)
    ├─ Operações morfológicas
    ├─ Transformada de distância
    ├─ Algoritmo Watershed
    └─ Extração de contornos válidos
    ↓
Lista de Contornos (grãos individuais)
    ↓
[feature_contourer.py]
    ├─ Para cada contorno:
    │   ├─ Features geométricas (6)
    │   └─ Features de cor LAB (6)
    ↓
Matriz de Features
    ↓
Modelo SVM
    ├─ Predição de classe
    └─ Probabilidades
    ↓
Resultados:
    ├─ Imagem anotada (overlay)
    └─ CSV com predições detalhadas
```

---

## 5. Tecnologias e Algoritmos de Visão Computacional

### 5.1 Algoritmos Implementados

1. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**

   - Normalização adaptativa de contraste
   - Aplicada no canal L do espaço LAB
   - Melhora a robustez a variações de iluminação

2. **Threshold de Otsu**

   - Binarização adaptativa automática
   - Seleciona threshold ótimo baseado em histograma
   - Inversão automática se necessário

3. **Operações Morfológicas**

   - Opening: Remove ruídos pequenos
   - Dilatação: Expande regiões para criar marcadores

4. **Transformada de Distância**

   - Calcula distância de cada pixel ao background mais próximo
   - Identifica centros dos objetos (picos de distância)

5. **Watershed**

   - Algoritmo de segmentação baseado em marcadores
   - Separa objetos em contato efetivamente
   - Fundamental para processar múltiplos grãos

6. **SVM (Support Vector Machine)**
   - Classificador de aprendizado supervisionado
   - Kernel RBF para fronteiras não-lineares
   - Probabilidades calibradas para confiança

### 5.2 Espaços de Cor Utilizados

- **BGR**: Formato padrão do OpenCV (equivalente RGB invertido)
- **LAB**: Espaço perceptualmente uniforme, ideal para análise de cor
- **Grayscale**: Para processamento morfológico e binarização

---

## 6. Integração com MQTT

### 6.1 Protocolo de Comunicação

O sistema utiliza MQTT para comunicação assíncrona entre componentes:

**Tópicos:**

- `image/transfer`: Recebe imagens para processamento
- `cocoa/results`: Publica resultados das predições

**Formato de Mensagem:**

- Payload: Imagem codificada em bytes (JPEG/PNG)
- QoS: 1 (pelo menos uma vez)

**Resultados publicados:**

```json
{
  "topic": "image/transfer",
  "results": [
    {
      "idx": 0,
      "x": 100,
      "y": 200,
      "w": 50,
      "h": 50,
      "pred_class": "good",
      "confidence": 0.95
    }
  ],
  "count": 1,
  "overlay_path": "runs/overlay_20251102_180156.jpg"
}
```

---

## 7. Dockerização e Deploy

### 7.1 Docker Compose

O projeto utiliza Docker Compose para orquestração:

**Serviços:**

1. **mosquitto**: Broker MQTT (porta 1883)
2. **classifier**: Worker de classificação (porta 8100)

**Dependências:**

- Classifier depende do Mosquitto estar saudável
- Healthcheck garante inicialização ordenada

### 7.2 Dockerfile do Classifier

```dockerfile
FROM python:3.12-slim
# Instala dependências do sistema (OpenCV)
# Instala dependências Python
# Expõe porta 8100
```

---

## 8. Uso Prático

### 8.1 Treinamento

```bash
cd classifier
uv run cli.py train --data-dir data/train --out-dir model/svm_v1
```

### 8.2 Predição Local

```bash
cd classifier
uv run cli.py predict \
  --image data/predict/cocoa-test.jpeg \
  --model-dir model/svm_v1 \
  --out-dir runs/test-run
```

### 8.3 Worker MQTT

```bash
cd classifier
MQTT_BROKER=mosquitto uv run worker.py
```

### 8.4 Makefile

Atalhos convenientes:

```bash
make train                    # Treina modelo
make predict image=path.jpg   # Predição
make classifier-worker        # Inicia worker MQTT
```

---

## 9. Estrutura de Arquivos

```
classifier/
├── cli.py                    # Interface CLI principal
├── worker.py                 # Worker MQTT
├── shared_state.py          # Estado compartilhado
├── cocoa_classifier/
│   ├── __init__.py
│   ├── bean_segmenter.py    # Segmentação Watershed
│   ├── feature_contourer.py # Extração de features
│   ├── data_loader.py       # Carregamento de dados
│   ├── trainer.py           # Treinamento SVM
│   ├── predictor.py         # Predição
│   ├── helpers.py           # Funções auxiliares
│   └── segment_params.py    # Parâmetros de segmentação
├── schemas/
│   ├── classifier_config.py # Schemas Pydantic
│   └── predict.py
├── data/
│   ├── train/               # Dataset de treinamento
│   │   ├── good/           # 458 imagens
│   │   └── bad/            # 458 imagens
│   └── predict/            # Imagens de teste
├── model/                   # Modelos treinados
│   └── svm_v1/
│       ├── model.pkl
│       └── classes.json
└── runs/                    # Resultados de predições
```

---

## 10. Características Técnicas Destacadas

### 10.1 Robustez

- **Normalização de contraste**: CLAHE adapta-se a diferentes condições de iluminação
- **Binarização adaptativa**: Otsu ajusta-se automaticamente ao histograma
- **Validação cruzada**: Garante generalização do modelo
- **Tratamento de erros**: Fallbacks para segmentação quando métodos principais falham

### 10.2 Performance

- **Processamento em lote**: Worker processa múltiplas imagens sequencialmente
- **Modelo em memória**: Evita recarregamento a cada predição
- **Features eficientes**: 12 dimensões otimizadas para classificação rápida

### 10.3 Escalabilidade

- **Arquitetura assíncrona**: MQTT permite múltiplos workers
- **Containerização**: Fácil deploy e escalonamento horizontal
- **Modularidade**: Componentes desacoplados e reutilizáveis

---

## 11. Limitações e Melhorias Futuras

### 11.1 Limitações Atuais

1. **Segmentação**: Pode ter dificuldades com grãos muito sobrepostos
2. **Features**: Limitado a características geométricas e de cor básicas
3. **Modelo**: SVM linear pode não capturar padrões complexos
4. **Dataset**: Relativamente pequeno (916 amostras)

### 11.2 Possíveis Melhorias

1. **Deep Learning**: Substituir SVM por CNN para melhor precisão
2. **Data Augmentation**: Aumentar dataset com transformações
3. **Features avançadas**: Textura, histogramas de gradientes orientados
4. **Segmentação**: U-Net ou Mask R-CNN para segmentação mais robusta
5. **Calibração**: Melhorar estimativas de confiança
6. **Interface Web**: Dashboard para visualização de resultados
7. **Métricas**: Implementar recall, precision, F1-score detalhados

---

## 12. Conclusão

O projeto demonstra uma implementação completa e bem estruturada de um sistema de classificação de grãos de cacau utilizando técnicas modernas de visão computacional. O módulo `classifier` é particularmente robusto, utilizando algoritmos clássicos mas eficazes como Watershed para segmentação e SVM para classificação.

A arquitetura modular facilita manutenção e extensão, enquanto a integração MQTT permite uso em ambientes de produção. O código está bem organizado, com separação clara de responsabilidades e uso adequado de bibliotecas especializadas.

O sistema está pronto para uso prático e pode servir como base para futuras melhorias, especialmente com a incorporação de técnicas de deep learning para aumentar ainda mais a precisão e robustez.

---

**Data do Relatório**: Novembro 2024  
**Versão do Projeto**: 0.1.0  
**Python**: 3.12+  
**Principais Dependências**: OpenCV 4.11, scikit-learn 1.7
