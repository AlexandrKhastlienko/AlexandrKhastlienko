<p align="center">

  <img src="./EA92039F-DE25-484B-82CE-BE6C8B5D18B6.png" alt="Александр Хаслиенко" width="100%">

</p>

# Александр Хаслиенко

### ML-engineer · System Design · MLOps · Computer Vision · Edge AI   

Разрабатываю ML-системы от подготовки данных и инференса до интеграции с оборудованием и управления артефактами.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

![Computer Vision](https://img.shields.io/badge/Computer_Vision-2563EB?style=flat-square)
![Edge AI](https://img.shields.io/badge/Edge_AI-0F766E?style=flat-square)
![Anomaly Detection](https://img.shields.io/badge/Anomaly_Detection-B45309?style=flat-square)
![MLOps](https://img.shields.io/badge/MLOps-7C3AED?style=flat-square)

</div>

---

## Обо мне

Я ML-инженер с практическим опытом в компьютерном зрении, обнаружении аномалий и построении инфраструктуры для командной ML-разработки. Работаю не только с моделями, но и с окружением, в котором они используются: Linux, Docker, GPU-инференс, промышленные камеры, встраиваемые устройства и системы хранения данных.

Для меня важны воспроизводимость экспериментов, понятная логика принятия решений и возможность проверить результат на реальных данных, а не только в ноутбуке.

## Проекты

### 01 · AI Stamping — визуальный контроль качества деталей на конвейере

![Computer Vision](https://img.shields.io/badge/Computer_Vision-2563EB?style=for-the-badge)
![YOLO](https://img.shields.io/badge/YOLO--Seg-00A86B?style=for-the-badge)
![ONNX](https://img.shields.io/badge/ONNX-005CED?style=for-the-badge&logo=onnx&logoColor=white)
![NVIDIA Jetson](https://img.shields.io/badge/NVIDIA_Jetson-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![Arduino](https://img.shields.io/badge/Arduino-00878F?style=for-the-badge&logo=arduino&logoColor=white)

**Задача.** Автоматизировать контроль металлических деталей: получить изображения с двух ракурсов, обнаружить дефекты поверхности и выдать одно решение по каждой детали для дальнейшей сортировки.

**Как устроена система.** Две промышленные камеры **Hikrobot** последовательно снимают одну деталь. Сигналы датчиков и управление движением конвейера обрабатывает **Arduino Mega 2560**. Снимки связываются с идентификатором детали (`part_id`), чтобы результаты разных камер не смешивались. Вычислительная часть ориентирована на **NVIDIA Jetson Orin**.

На стороне ML используется **YOLO-сегментация** с тремя типами дефектов: `chip` (скол), `unpolished` (неполированная область) и `termo` (термический дефект). Для инференса модель экспортируется в **ONNX**; обработка изображений выполняется в Python. Результаты двух кадров объединяются в одно решение: дефект на любом снимке приводит к статусу `DEFECT`, отсутствие необходимого кадра или ошибка обработки — к `ERROR`, отсутствие выявленных дефектов при корректной паре — к `NORMAL` с учётом правил контроля сторон детали.

**Схема обработки:**

```mermaid
flowchart LR
    A["Датчики и Arduino"] --> B["Две камеры Hikrobot"]
    B --> C["Два снимка / part_id"]
    C --> D["YOLO-seg / ONNX"]
    D --> E{"Контроль пары"}
    E --> F["NORMAL"]
    E --> G["DEFECT"]
    E --> H["ERROR"]
```

**Что сделано в программном пилоте:** загрузка конфигурации, ONNX-инференс пары изображений, оркестрация обработки, сохранение исходных и размеченных кадров, формирование `result.json`, интерфейс оператора и отправка управляющих команд по Serial. Отдельно прорабатываются синхронизация съёмки, управление подачей следующей детали и передача решения сортировщику.

**Статус:** программный пилот реализован; интеграция с реальным конвейером и оборудованием — в разработке.

**Стек:** Python · Ultralytics YOLO · OpenCV · ONNX Runtime · Docker · NVIDIA Jetson · Hikrobot · Arduino / Serial

### 02 · Anti-Fraud — ранжирование подозрительных событий без учителя

![Anomaly Detection](https://img.shields.io/badge/Anomaly_Detection-7C3AED?style=for-the-badge)
![Unsupervised ML](https://img.shields.io/badge/Unsupervised_ML-DB2777?style=for-the-badge)
![Mahalanobis](https://img.shields.io/badge/Robust_Mahalanobis-DC2626?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

**Задача.** Построить систему, которая выделяет нетипичные наблюдения из потока данных и **ранжирует их по степени аномальности**. Основная сложность — сильный **дисбаланс классов**: нормальных наблюдений существенно больше, чем аномальных. При таких условиях обучение обычного бинарного классификатора может быть ограничено количеством и репрезентативностью примеров редких событий.

**Подход.** Вместо обучения на двух классах используется **one-class / unsupervised anomaly detection**: модель изучает структуру нормального поведения, а затем оценивает, насколько новое наблюдение от неё отклоняется. В основе скоринга — **робастное расстояние Махаланобиса**, которое учитывает не только величину отклонений отдельных признаков, но и взаимосвязи между ними. Для повышения устойчивости применяются предварительная обработка признаков и регуляризация оценки ковариации.

**Как работает пайплайн.** Данные проходят подготовку и преобразование признаков → вычисляется числовой **anomaly score** → наблюдения сортируются по убыванию подозрительности. Отдельный этап превращает скоринг в сигналы: калибровка порогов, агрегация результатов во временных окнах и постобработка помогают уменьшить количество кратковременных ложных срабатываний. Таким образом, ранжирование и окончательное решение о сигнале разделены — порог можно настраивать под требования к ложным тревогам.

**Схема скоринга:**

```mermaid
flowchart LR
    A["Нормальные наблюдения"] --> B["Оценка центра и ковариации"]
    B --> C["Робастная модель нормы"]
    D["Новые наблюдения"] --> E["Расстояние Махаланобиса"]
    C --> E
    E --> F["Anomaly score"]
    F --> G["Ранжирование"]
    G --> H["Порог и сигналы"]
```

**Проверка качества.** Для обучения используются нормальные наблюдения; размеченные аномальные случаи нужны для независимой проверки и подбора порогов, а не для обучения двухклассовой модели. В проекте предусмотрена валидация с разделением по сессиям (**LOSO**), чтобы снизить риск утечки между обучением и проверкой. Рассматриваются **FPR, полнота обнаружения, PR-AUC и задержка обнаружения** — при редких событиях одной accuracy недостаточно.

**Стек и методы:** Python · NumPy · Pandas · scikit-learn · Robust Mahalanobis · One-Class Learning · обработка временных рядов · сессионная валидация

> Описание ограничено общими принципами алгоритма: прикладная область, исходные данные и детали внедрения не публикуются.

### 03 · MLOps Infrastructure — инфраструктура разработки и хранения ML-артефактов

![Gitea](https://img.shields.io/badge/Gitea-609926?style=for-the-badge&logo=gitea&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-13ADC7?style=for-the-badge&logo=dvc&logoColor=white)
![MinIO](https://img.shields.io/badge/MinIO-C72E49?style=for-the-badge&logo=minio&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)

**Задача.** Организовать совместную работу над ML-проектами так, чтобы код, датасеты, веса моделей и эксперименты не смешивались в одном репозитории, а нужное состояние проекта можно было восстановить.

**Организация хранения.** **Git / Gitea** используются для исходного кода, конфигураций, истории изменений и работы с ветками. Крупные датасеты и веса моделей версионируются через **DVC**; сами файлы размещаются в S3-совместимом **MinIO**, а Git хранит ссылки на их версии. Это позволяет не загружать тяжёлые бинарные файлы непосредственно в историю Git и связывать изменения кода с соответствующими данными.

**Разделение ответственности сервисов:**

```mermaid
flowchart LR
    A["Git / Gitea"] --> B["Код, конфигурации, DVC-метаданные"]
    C["DVC"] --> D[("MinIO / S3")]
    E["MLflow"] --> F["Параметры, метрики, запуски"]
    G["Docker Compose"] --> H["Окружение сервисов"]
```

**Эксперименты и окружение.** Для фиксации запусков, параметров и метрик используется **MLflow**. Сервисы разворачиваются через **Docker Compose**; в инфраструктуре применяются **PostgreSQL**, Linux и локальная сеть, в том числе для рабочих мест без прямого доступа к интернету. Такая архитектура разделяет задачи Git, объектного хранилища и трекинга экспериментов, вместо попытки решить их одним инструментом.

**Практический результат:** организована серверная инфраструктура для командной ML-разработки и хранения версионируемых артефактов. Развитие автоматизированных пайплайнов и единых командных правил — отдельная текущая задача; полноценную CI/CD-платформу здесь не заявляю.

**Стек:** Git · Gitea · DVC · MinIO (S3) · MLflow · Docker Compose · PostgreSQL · Ubuntu

## Технологии

<div align="center">

<img src="https://skillicons.dev/icons?i=python,pytorch,opencv,sklearn,docker,git,github,postgres,linux,bash,vscode,arduino&perline=12&theme=dark" alt="Иконки основных технологий: Python, PyTorch, OpenCV, scikit-learn, Docker, Git, GitHub, PostgreSQL, Linux, Bash, VS Code, Arduino" />

</div>


| Направление | Инструменты |
| :--- | :--- |
| **Язык и анализ данных** | Python, NumPy, Pandas, Matplotlib, Jupyter |
| **ML / CV** | PyTorch, scikit-learn, Ultralytics YOLO, OpenCV |
| **Инференс** | ONNX Runtime, NVIDIA CUDA, NVIDIA Jetson |
| **MLOps** | Git, GitHub, Gitea, DVC, MLflow, MinIO (S3) |
| **Инфраструктура** | Docker, Docker Compose, Linux / Ubuntu, WSL2, PostgreSQL |
| **Локальные AI-инструменты** | Ollama, Open WebUI, MCP |
| **Разработка и интеграция** | VS Code, промышленные камеры Hikrobot, Arduino, Serial |

---

<div align="center">

*От данных и математической модели — к воспроизводимому инференсу и реальной системе.*

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0F172A,50:2563EB,100:7C3AED&height=90&section=footer" alt="" width="100%" />

</div>
