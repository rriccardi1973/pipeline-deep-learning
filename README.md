# Pipeline Base de Entrenamiento y Validación en PyTorch

## Descripción del proyecto

Este proyecto corresponde al primer checkpoint del Proyecto Final de Data Science III.

El objetivo es desarrollar un pipeline base de entrenamiento y validación para un modelo de clasificación utilizando PyTorch. El foco principal de esta entrega está puesto en la construcción de una infraestructura de entrenamiento reproducible y organizada, incorporando las etapas fundamentales del ciclo de aprendizaje de una red neuronal.

Para la prueba técnica se utiliza el dataset Iris, un conjunto de datos clásico de clasificación multiclase compuesto por cuatro variables de entrada y tres clases posibles.

## Estructura del repositorio

El proyecto se encuentra organizado de la siguiente manera:

```text
pipeline-deep-learning/
├── data/
│   └── README.md
├── outputs/
│   └── .gitkeep
├── src/
│   └── train.py
├── .gitignore
├── README.md
└── requirements.txt
```

La carpeta `data` está destinada al almacenamiento de datasets.

La carpeta `src` contiene la lógica principal del modelo y el pipeline de entrenamiento.

La carpeta `outputs` se utiliza para almacenar los resultados generados durante el entrenamiento, como el modelo entrenado y las curvas de pérdida y accuracy.

## Configuración del entorno

El pipeline detecta automáticamente el dispositivo disponible para ejecutar el modelo.

El orden de selección es:

1. GPU mediante CUDA.
2. GPU de Apple mediante MPS.
3. CPU.

También se fijan semillas de aleatoriedad en Python, NumPy y PyTorch para mejorar la reproducibilidad del experimento.

## Dataset

Para este experimento se utiliza el dataset Iris disponible en la librería scikit-learn.

El dataset contiene 150 observaciones correspondientes a tres especies de flores Iris.

Cada observación posee cuatro variables:

* longitud del sépalo;
* ancho del sépalo;
* longitud del pétalo;
* ancho del pétalo.

La variable objetivo corresponde a la especie de la flor.

Los datos se dividen en un conjunto de entrenamiento del 80 % y un conjunto de validación del 20 %.

Además, las variables de entrada son estandarizadas utilizando `StandardScaler`.

El escalado se ajusta únicamente utilizando los datos de entrenamiento para evitar fuga de información hacia el conjunto de validación.

## Arquitectura del modelo

Se implementó una red neuronal multicapa simple mediante `nn.Module` y `nn.Sequential`.

La arquitectura utilizada es:

```text
Entrada: 4 variables
        ↓
Capa lineal: 4 → 16 neuronas
        ↓
Función de activación ReLU
        ↓
Capa lineal: 16 → 3 clases
        ↓
Salida del modelo
```

No se utiliza una función Softmax explícita en la capa de salida debido a que `CrossEntropyLoss` trabaja directamente con los logits generados por el modelo.

## Hiperparámetros

Los principales hiperparámetros utilizados son:

* Épocas: 100
* Batch size: 16
* Learning rate: 0.01
* Optimizador: Adam
* Función de pérdida: CrossEntropyLoss
* Neuronas en la capa oculta: 16
* Semilla de aleatoriedad: 42

Se seleccionó un learning rate de 0.01 debido a que permite una convergencia rápida para un dataset pequeño y de baja complejidad como Iris.

## Ciclo de entrenamiento

Durante cada época se ejecutan las siguientes etapas:

1. Los datos son trasladados al dispositivo seleccionado mediante `.to(device)`.
2. Se eliminan los gradientes acumulados utilizando `optimizer.zero_grad()`.
3. Se realiza el forward pass del modelo.
4. Se calcula la función de pérdida.
5. Se ejecuta `loss.backward()` para calcular los gradientes mediante autograd.
6. El optimizador actualiza los pesos mediante `optimizer.step()`.

El uso de `optimizer.zero_grad()` es necesario porque PyTorch acumula los gradientes por defecto.

## Ciclo de validación

Al finalizar cada época se evalúa el modelo utilizando el conjunto de validación.

Durante esta etapa se utilizan:

```python
model.eval()
```

y:

```python
@torch.no_grad()
```

Esto permite evaluar el modelo sin actualizar sus parámetros y evita el cálculo innecesario de gradientes.

## Métricas utilizadas

Durante el entrenamiento se registran las siguientes métricas para cada época:

* pérdida de entrenamiento;
* pérdida de validación;
* accuracy de entrenamiento;
* accuracy de validación.

El modelo con menor pérdida de validación se guarda automáticamente en:

```text
outputs/best_model.pt
```

También se generan las siguientes curvas:

```text
outputs/loss_curve.png
outputs/accuracy_curve.png
```

## Interpretación de la curva de pérdida

Durante las primeras épocas se espera observar una disminución de la pérdida tanto en el conjunto de entrenamiento como en el conjunto de validación.

La reducción progresiva de la pérdida indica que el modelo está ajustando sus parámetros y mejorando su capacidad de clasificación.

Si la pérdida de entrenamiento continúa disminuyendo mientras la pérdida de validación comienza a aumentar, podría existir un problema de overfitting.

En este experimento se espera que ambas curvas disminuyan y posteriormente se estabilicen, debido a la baja complejidad del dataset Iris y al tamaño reducido de la arquitectura utilizada.

## Versión de PyTorch

La versión instalada de PyTorch puede verificarse ejecutando:

```python
import torch

print(torch.__version__)
```

Las dependencias necesarias para ejecutar el proyecto se encuentran definidas en el archivo `requirements.txt`.

## Instalación

Para instalar las dependencias del proyecto ejecutar:

```bash
pip install -r requirements.txt
```

## Ejecución

Desde la carpeta principal del repositorio ejecutar:

```bash
python src/train.py
```

Durante el entrenamiento se mostrarán en consola los valores de pérdida y accuracy correspondientes al conjunto de entrenamiento y validación.

## Resultados esperados

El modelo debería mostrar una reducción progresiva de la función de pérdida durante las épocas de entrenamiento.

Al mismo tiempo, el accuracy debería aumentar y estabilizarse en valores elevados.

El seguimiento conjunto de las métricas de entrenamiento y validación permite identificar posibles señales de overfitting y evaluar la capacidad de generalización del modelo.

## Posibles mejoras futuras

Como continuación del proyecto se podrían incorporar:

* Early Stopping.
* F1-Score.
* Matriz de confusión.
* Búsqueda de hiperparámetros.
* TensorBoard o MLflow para tracking de experimentos.
* Un conjunto independiente de test.
* Modelos de mayor complejidad.
