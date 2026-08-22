# Video Dataset Factory - простое объяснение проекта для интервью

Этот документ нужен, чтобы быстро и уверенно объяснить проект ML-инженеру или рекрутеру. Он написан простым языком, но с правильными техническими деталями.

## 1. Идея в одном предложении

**Video Dataset Factory** - это end-to-end pipeline, который превращает сырые видео в очищенный, описанный, проверенный и training-ready датасет для generative video / diffusion fine-tuning экспериментов.

Если совсем просто:

> Мы берем много видео, проверяем какие из них хорошие, описываем их через VLM, убираем мусор и дубликаты, собираем manifest и готовим данные для обучения/fine-tuning модели.

## 2. Простая аналогия

Представь фабрику.

На вход приходит коробка с разными видео:

- часть видео хорошая;
- часть видео темная;
- часть слишком маленькая;
- часть с watermark;
- часть почти без движения;
- часть дублируется;
- часть плохо описана или вообще без описания.

Фабрика делает:

1. Открывает каждое видео.
2. Проверяет технические параметры.
3. Достает несколько кадров.
4. Смотрит, есть ли движение.
5. Проверяет качество картинки.
6. Ищет текст и watermark.
7. Спрашивает VLM: `Что происходит на видео?`
8. Ищет near-duplicates.
9. Решает: оставить клип или отклонить.
10. Записывает результат в `manifest.jsonl`.
11. Делает отчет.
12. Готовит данные для LoRA/diffusion fine-tuning.

То есть это не просто upload video app. Это pipeline подготовки данных для video generation research.

## 3. Что делает pipeline

Общий путь данных:

```text
raw videos
  -> video metadata probing
  -> frame/keyframe sampling
  -> quality checks
  -> OCR / watermark filtering
  -> motion scoring
  -> LAION aesthetic scoring
  -> VLM dense captioning
  -> perceptual hashing
  -> deduplication
  -> manifest.jsonl
  -> reports / Streamlit dashboard
  -> Diffusers LoRA dataset export
  -> GPU training benchmarks
```

Главная мысль:

> Хорошая generative video модель требует хороших видео и хороших captions. Если обучаться на мусоре, модель тоже будет генерировать мусор.

## 4. Почему проект релевантен generative video

Для text-to-video модели важно, чтобы caption точно соответствовал видео.

Плохой пример:

```text
caption: a man standing outside
video: basketball player dribbling indoors
```

Такой пример портит обучение.

Хороший пример:

```text
A shirtless male basketball player dribbles the ball quickly on an indoor court. The handheld camera follows the action closely. Other players are visible in the background. The lighting is warm indoor gym lighting.
```

Такой caption полезен, потому что он описывает:

- subject;
- action;
- scene;
- camera motion;
- lighting;
- background;
- movement.

## 5. Что такое `manifest.jsonl`

`manifest.jsonl` - это главный результат pipeline.

JSONL означает JSON Lines: каждая строка - отдельная запись про один клип.

Примерно так:

```json
{
  "source_path": "data/clips/example.mp4",
  "duration_sec": 4.2,
  "fps": 24.0,
  "width": 512,
  "height": 512,
  "caption": "A person runs across a field...",
  "motion_score": 1.8,
  "aesthetic_score": 5.7,
  "ocr_text_area_ratio": 0.01,
  "keep": true,
  "reject_reasons": []
}
```

Зачем JSONL:

- удобно читать построчно;
- удобно использовать в training scripts;
- удобно обрабатывать большими batch-ами;
- удобно загружать в pandas/Ray/Spark-like pipelines;
- одна плохая строка не ломает весь датасет.

## 6. VLM dense captioning

VLM - это Vision-Language Model: модель, которая видит изображение и отвечает текстом.

В проекте VLM получает sampled keyframes из видео и генерирует dense caption.

Важно: captions не fake. Pipeline реально отправляет изображения в vision model и просит описать, что на них происходит.

Почему мы не отправляем все видео целиком:

- VLM providers часто работают с изображениями;
- полный video understanding дороже;
- у API есть лимиты по числу изображений;
- keyframe captioning - практичный компромисс для data curation.

## 7. Motion scoring

Motion score отвечает на вопрос:

> В клипе реально есть движение или это почти статичная картинка?

Для video generation это важно. Если датасет состоит из статичных клипов, video model плохо учит temporal dynamics.

Примеры:

- low motion: человек сидит, камера не двигается;
- moderate motion: человек идет;
- high motion: спорт, танец, быстрый camera movement.

В проекте motion используется как quality signal и как часть metadata/reporting.

## 8. OCR / watermark filtering

Многие видео из интернета содержат:

- subtitles;
- TikTok / YouTube overlays;
- usernames;
- logos;
- embedded text;
- watermark.

Для generative models это плохо: модель может научиться генерировать такие же артефакты.

В проекте есть настоящие OCR providers:

- EasyOCR;
- Tesseract;
- proxy mode для быстрых smoke tests.

Правильная формулировка на интервью:

> I implemented modular OCR/text filtering: a lightweight proxy for fast runs and real EasyOCR/Tesseract providers for actual text/watermark detection.

## 9. LAION-Aesthetics scoring

В проекте теперь есть настоящий LAION-style aesthetic scoring.

Схема такая:

```text
video frame
  -> open_clip image encoder
  -> normalized image embedding
  -> official LAION aesthetic linear head
  -> aesthetic score
```

То есть это не просто heuristic.

В текущей GitHub версии используется:

- `open_clip` image embeddings;
- official LAION `sa_0_4_vit_b_32_linear.pth` linear head;
- median score по нескольким sampled frames.

Важно говорить честно:

> LAION-Aesthetics is an image-level preference predictor. It is useful as a filtering signal, but the threshold should be calibrated on manually reviewed clips.

То есть score помогает, но не заменяет человеческую проверку и другие метрики.

## 10. pHash deduplication

Обычный file hash находит только полностью одинаковые файлы.

pHash - perceptual hash. Он помогает находить визуально похожие клипы, даже если файл перекодирован или немного изменен.

Зачем:

- убрать near-duplicates;
- повысить diversity;
- снизить overfitting;
- не считать повторяющиеся клипы за новые данные.

## 11. Streamlit dashboard

Streamlit UI нужен, чтобы проект можно было показать без терминала.

Он позволяет:

- загрузить короткие видео;
- запустить processing;
- увидеть accepted/rejected;
- посмотреть captions;
- посмотреть quality/motion/aesthetic scores;
- увидеть reject reasons;
- скачать manifest/report.

Для больших batch-ов лучше использовать CLI, а в Streamlit загружать уже готовый manifest.

## 12. CLI

CLI команда проекта - `vdf`.

Примеры:

```bash
vdf process-folder data/clips --output outputs/manifest.jsonl
vdf dedupe-manifest outputs/manifest.jsonl --output outputs/manifest_deduped.jsonl
vdf summarize-manifest outputs/manifest_deduped.jsonl --output outputs/dataset_summary.md
```

Зачем CLI:

- воспроизводимые эксперименты;
- удобно запускать на Kaggle/server;
- можно автоматизировать;
- проект выглядит как engineering tool, а не как один notebook.

## 13. Ray parallel processing

Видео можно обрабатывать независимо.

Это значит, что pipeline хорошо параллелится:

```text
video_1 -> worker_1
video_2 -> worker_2
video_3 -> worker_3
```

Ray adapter показывает, что pipeline можно масштабировать на несколько workers.

Честная формулировка:

> I added Ray-ready parallel processing for scaling preprocessing. I do not claim production multi-node cluster management, but the architecture is designed for parallel execution.

## 14. Diffusers LoRA export

Pipeline не заканчивается отчетом. Он может подготовить данные для diffusion fine-tuning.

Схема:

```text
manifest.jsonl
  -> sampled images
  -> metadata.jsonl
  -> Diffusers image-caption dataset
  -> Stable Diffusion LoRA fine-tuning
```

Почему LoRA:

- full diffusion training слишком дорогой для pet project;
- LoRA realistic на Kaggle GPU;
- показывает fine-tuning workflow;
- связывает data pipeline с generative AI training.

Правильная формулировка:

> I used LoRA as a compute-realistic fine-tuning target. The goal was not to train a foundation model from scratch, but to demonstrate the path from curated video data to diffusion model adaptation.

## 15. Kaggle GPU experiments

На Kaggle проверялась GPU часть:

- CUDA availability;
- PyTorch GPU execution;
- mixed precision;
- memory/throughput reporting;
- Accelerate launch;
- DeepSpeed-style config;
- Diffusers LoRA workflow.

Что это доказывает:

- ты умеешь запускать ML experiments на GPU;
- понимаешь memory/throughput;
- видел реальные ошибки окружения;
- умеешь работать с Accelerate/DeepSpeed tooling.

Что это не доказывает:

- production GPU cluster management;
- обучение foundation model с нуля;
- custom CUDA kernels.

## 16. Основные технологии

Python:

- основной язык проекта.

Pydantic:

- typed schemas для config и manifest records.

Typer:

- CLI interface.

OpenCV:

- frame extraction, blur/brightness, image analysis.

ffmpeg:

- video decoding/normalization.

PySceneDetect:

- scene splitting.

EasyOCR/Tesseract:

- real OCR/text filtering.

open_clip + LAION linear head:

- настоящий aesthetic scoring.

VLM / Groq vision:

- dense captioning по keyframes.

pHash:

- near-duplicate detection.

Ray:

- parallel processing.

Streamlit:

- public dashboard/demo.

PyTorch:

- GPU benchmark/training code.

Accelerate:

- single/multi-GPU launch abstraction.

DeepSpeed:

- ZeRO-style distributed training config.

Diffusers:

- LoRA fine-tuning workflow.

GitHub Actions:

- CI/tests/lint discipline.

## 17. Какие навыки проект закрывает

Video data engineering:

- обработка сырых видео;
- metadata probing;
- frame sampling;
- scene splitting;
- manifest generation.

Computer vision:

- blur/brightness/contrast/colorfulness;
- motion scoring;
- OCR;
- perceptual hashing;
- aesthetic scoring.

Multimodal AI:

- VLM dense captions.

Generative AI:

- Diffusers LoRA dataset export;
- fine-tuning workflow.

ML Research Engineering:

- configs;
- experiment logs;
- failed experiments;
- reports;
- metrics;
- reproducibility.

GPU / distributed basics:

- Kaggle T4 experiments;
- CUDA;
- mixed precision;
- Accelerate;
- DeepSpeed-style config;
- Ray preprocessing.

Software engineering:

- Python package;
- CLI;
- tests;
- CI;
- docs;
- dashboard.

## 18. Что не надо преувеличивать

Не говори:

```text
I trained a production video foundation model.
```

Говори:

```text
I built the dataset preparation and fine-tuning readiness infrastructure around generative video models.
```

Не говори:

```text
I managed production GPU clusters.
```

Говори:

```text
I validated GPU training workflows on Kaggle T4 GPUs and implemented Accelerate/DeepSpeed-ready benchmarks and configs.
```

Не говори:

```text
I wrote custom CUDA kernels.
```

Говори:

```text
I profiled GPU memory and throughput using PyTorch, mixed precision, Accelerate, and benchmark scripts.
```

## 19. Как объяснить проект ML инженеру

Можно сказать так:

```text
I built Video Dataset Factory, an end-to-end video dataset preparation pipeline for generative video and diffusion fine-tuning experiments.

It takes raw videos, samples keyframes, computes metadata, motion and quality scores, filters low-quality clips, detects OCR/text/watermark artifacts, removes near-duplicates with perceptual hashing, generates dense captions through a VLM, and writes a training-ready JSONL manifest.

I also added a Streamlit dashboard, Ray-ready parallel processing, real LAION-style aesthetic scoring via open_clip embeddings and the official LAION linear head, and Kaggle GPU experiments with Accelerate/DeepSpeed-style training benchmarks. Finally, the pipeline can export curated clips into a Diffusers LoRA fine-tuning dataset.

The goal was to practice the research-engineering workflow around generative video models: data curation, multimodal captioning, filtering, reproducible configs, GPU constraints, and the bridge from raw video data to fine-tuning.
```

## 20. Что сказать, если спросят про limitations

Хороший ответ:

```text
The project is intentionally scoped as a research-engineering pipeline, not a production foundation-model training system. It does not train a video diffusion model from scratch. Instead, it focuses on the realistic upstream work: data quality, captions, filtering, deduplication, reproducible manifests, reports, and fine-tuning readiness. The next step would be calibrating thresholds on manually reviewed clips and measuring downstream generation quality after LoRA fine-tuning.
```

## 21. Что бы улучшить дальше

Следующие хорошие шаги:

- откалибровать LAION threshold на вручную проверенных клипах;
- добавить CLIPScore между caption и frames;
- добавить video-level embeddings для semantic filtering;
- подключить W&B для experiment tracking;
- прогнать pipeline на более качественном high-resolution dataset;
- сравнить downstream LoRA results до/после фильтрации;
- добавить human review loop из Streamlit обратно в manifest.

## 22. Финальная честная формулировка

Самая сильная версия:

> Video Dataset Factory is a research-engineering project around generative video data. It does not claim to train a production video foundation model from scratch. Instead, it solves the realistic upstream problem: turning messy raw videos into curated, captioned, deduplicated, quality-scored, training-ready data, with reproducible configs, reports, a UI, parallel processing, real OCR/LAION quality checks, and GPU fine-tuning experiments.
