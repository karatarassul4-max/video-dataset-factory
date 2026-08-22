# Code Walkthrough: как устроен Video Dataset Factory

Этот файл объясняет проект как программисту: какие файлы за что отвечают, как они связаны, какой поток данных проходит через код, какие функции являются главными точками входа, и как читать проект сверху вниз.

Важно: буквально объяснить каждую физическую строку всего репозитория в одном документе невозможно без превращения файла в огромную книгу. Поэтому здесь используется практичный формат: **построчно по смысловым блокам**. То есть для каждого файла объясняется, зачем там imports, constants, classes, functions, helper-функции, и как эти части вызываются другими файлами.

## 1. Главная ментальная модель

Проект делает одну большую вещь:

```text
raw video files
  -> read metadata
  -> sample frames
  -> compute quality metrics
  -> compute motion metrics
  -> run OCR / watermark detection
  -> run LAION aesthetic scoring
  -> run VLM captioning
  -> detect caption artifacts
  -> compute perceptual hash
  -> create ClipRecord
  -> write JSONL manifest
  -> deduplicate manifest
  -> summarize / review / export / train
```

Главный объект, который связывает почти все файлы, - это `ClipRecord` из `schema.py`.

Главная функция обработки одного видео - `process_video()` из `pipeline.py`.

Главный CLI вход - `app` в `cli.py`, который создает команду `vdf`.

Главный web UI вход - `main()` в `dashboards/app.py`.

## 2. Главный call graph

Когда ты запускаешь CLI:

```bash
vdf process-folder data/clips --output outputs/manifest.jsonl
```

происходит примерно так:

```text
cli.py
  -> load_config() из config.py
  -> process_folder_command()
  -> для каждого video path вызывает process_video() из pipeline.py
      -> probe_video() из video_io.py
      -> sample_frames() из video_io.py
      -> aggregate_quality() из quality.py
          -> blur_score()
          -> brightness_score()
          -> contrast_score()
          -> colorfulness_score()
          -> TextDetector / EasyOCR / Tesseract
          -> AestheticScorer / LAION / CLIP / heuristic
      -> motion_metrics() из motion.py
      -> quality_reject_reasons() из quality.py
      -> motion_reject_reasons() из motion.py
      -> captioner.caption() из caption.py
      -> caption_reject_reasons() из caption.py
      -> clip_perceptual_hash() из duplicates.py
      -> ClipRecord из schema.py
  -> write_jsonl() из manifest.py
```

Когда ты запускаешь Streamlit:

```bash
streamlit run dashboards/app.py
```

поток похожий, но вход другой:

```text
dashboards/app.py
  -> render_upload_mode()
  -> process_uploaded_files()
  -> process_video() из pipeline.py
  -> mark_duplicates() из duplicates.py
  -> render_summary() / render_score_distributions() / render_gallery()
```

Когда готовится diffusion LoRA dataset:

```text
manifest.jsonl
  -> diffusion_finetune.py
      -> read_jsonl()
      -> sample frames from video
      -> write images/
      -> write metadata.jsonl
  -> Diffusers training script
```

## 3. Root-файлы проекта

### `pyproject.toml`

Это packaging-файл проекта.

Что в нем важно:

- `[build-system]` говорит pip/setuptools, как собирать package.
- `[project]` задает имя `video-dataset-factory`, версию, описание, Python version и базовые зависимости.
- `[project.optional-dependencies]` делит тяжелые зависимости на extras: `dev`, `dashboard`, `ocr`, `aesthetic`, `training`, `inference`, `scene` и т.д.
- `[project.scripts]` создает CLI-команду `vdf = video_dataset_factory.cli:app`.
- `[tool.pytest.ini_options]` говорит pytest искать тесты в `tests` и добавляет `src` в pythonpath.

Почему это важно:

> Без `pyproject.toml` проект был бы набором скриптов. С ним это нормальный Python package, который можно установить через `pip install -e .` и запускать как CLI.

### `requirements.txt`

Обычно содержит зависимость для Streamlit deployment. В проекте основная зависимость может быть `-e .[dashboard]`, чтобы Streamlit Cloud установил сам package и dashboard extra.

### `packages.txt`

Системные зависимости для Streamlit Cloud, например `ffmpeg`. Это не Python package, а Linux package. Видеообработка часто требует системные binaries.

### `README.md`

Главная витрина проекта:

- зачем проект нужен;
- какие features есть;
- как запустить CLI;
- как запустить Streamlit;
- как включить Groq/VLM;
- как включить OCR/LAION;
- как запускать benchmarks;
- какие результаты уже есть.

README не является runtime-кодом, но для portfolio проекта он критичен.

## 4. Config files

### `configs/default.yaml`

Главный runtime config.

Секции:

- `pipeline`: сколько кадров сэмплировать и куда писать manifest.
- `quality`: пороги reject logic: duration, resolution, brightness, blur, motion, aesthetic.
- `scene_split`: параметры PySceneDetect/ffmpeg normalization.
- `captioning`: provider captions. Для CLI smoke-run может быть `heuristic`; для настоящего VLM можно ставить `groq` или `transformers`.
- `ocr`: provider OCR. Сейчас поддерживаются proxy/none, EasyOCR, Tesseract.
- `aesthetic`: provider aesthetic scoring. Сейчас есть heuristic, clip proxy, LAION через `open_clip` + official linear head.
- `ray`: включение/настройки Ray.

Связь с кодом:

```text
config.py -> AppConfig из schema.py -> pipeline.py / cli.py / dashboard
```

### `configs/accelerate_kaggle.yaml`

Конфиг Hugging Face Accelerate для Kaggle multi-GPU запуска.

Смысл:

- сказать Accelerate, сколько процессов/GPU использовать;
- какой distributed type;
- mixed precision;
- machine rank / num machines.

### `configs/accelerate_deepspeed_zero2.yaml`

Конфиг Accelerate + DeepSpeed ZeRO-2.

Смысл:

- показать DeepSpeed-style distributed training setup;
- включить ZeRO optimization;
- использовать это с `training_entrypoint.py`.

## 5. `schema.py`: контракты данных

Этот файл лучше читать одним из первых.

Он отвечает на вопрос:

> Какие типы данных вообще существуют в проекте?

Главные imports:

- `Path` нужен для путей в configs.
- `Any` нужен для гибких provider configs вроде `captioning`, `ocr`, `aesthetic`.
- `BaseModel`, `Field` из Pydantic нужны для typed models и default factories.

Главные классы:

### `VideoMetadata`

Описывает технические параметры видео:

- `source_path`;
- `duration_sec`;
- `fps`;
- `width`;
- `height`;
- `frame_count`.

Кто создает:

```text
video_io.probe_video()
```

Кто использует:

```text
pipeline.process_video()
quality.quality_reject_reasons()
```

### `QualityConfig`

Пороги, по которым clip accepted/rejected:

- duration min/max;
- resolution min;
- blur min;
- brightness min/max;
- OCR text area max;
- motion min/max;
- aesthetic min.

Кто использует:

```text
quality_reject_reasons()
motion_reject_reasons()
pipeline.process_video()
```

### `PipelineConfig`

Общие параметры pipeline:

- `sample_frames`: сколько кадров достать из видео;
- `output_manifest`: куда писать manifest.

### `SceneSplitConfig`

Параметры scene splitting и ffmpeg export:

- detector;
- threshold;
- min scene length;
- output dir;
- fps/width/height;
- CRF/preset.

### `AppConfig`

Главный config проекта.

Он собирает:

```text
pipeline + quality + scene_split + captioning + ocr + aesthetic + ray
```

Это один объект, который передается в `process_video()`.

### `ClipRecord`

Самая важная модель manifest.

Это запись про один обработанный клип:

- metadata;
- quality scores;
- motion scores;
- OCR score;
- aesthetic score;
- pHash;
- caption;
- keep/reject decision;
- reject reasons.

Кто создает:

```text
pipeline.process_video()
```

Кто читает:

```text
manifest.py
reporting.py
duplicates.py
dashboards/app.py
diffusion_finetune.py
training_benchmark.py
```

### `SceneSegment`

Описывает один найденный scene segment после scene splitting:

- source video;
- clip path;
- scene index;
- start/end seconds;
- duration.

## 6. `config.py`: загрузка YAML в Pydantic config

Файл маленький, но важный.

Типичный поток:

```text
Path to YAML
  -> yaml.safe_load()
  -> dict
  -> AppConfig.model_validate(dict)
```

Почему так:

- YAML удобен человеку;
- Pydantic удобен коду;
- config errors ловятся раньше.

Кто вызывает:

```text
cli.py
```

Streamlit чаще создает `AppConfig()` прямо в коде, потому что UI controls динамически меняют config.

## 7. `video_io.py`: низкоуровневая работа с видео

Этот файл отвечает за boundary между реальными video files и Python-кодом.

Главные задачи:

### `is_video_file()`

Проверяет расширение файла.

Нужно для `process-folder`, чтобы не пытаться обработать `.txt`, `.json`, `.jpg` как видео.

### `iter_video_files()`

Ищет видео в папке.

Кто использует:

```text
cli.py
benchmark_pipeline.py
```

### `probe_video()`

Открывает видео через OpenCV и достает metadata:

- frame count;
- fps;
- width;
- height;
- duration.

Поток:

```text
cv2.VideoCapture(path)
  -> CAP_PROP_FRAME_COUNT
  -> CAP_PROP_FPS
  -> CAP_PROP_FRAME_WIDTH
  -> CAP_PROP_FRAME_HEIGHT
  -> VideoMetadata
```

### `sample_frames()`

Достает N кадров из видео равномерно по длине.

Зачем:

- quality scoring;
- motion scoring;
- VLM captioning;
- OCR;
- LAION aesthetic scoring;
- pHash.

Важно:

> Мы не обрабатываем каждый кадр видео. Мы берем representative frames, потому что это дешевле и достаточно для dataset filtering.

## 8. `quality.py`: качество, OCR, LAION

Это один из самых насыщенных файлов.

Его можно читать сверху вниз.

### Imports и constants

- `urllib.request` нужен для скачивания LAION linear head weights.
- `Path` нужен для cache path.
- `Protocol` нужен для интерфейсов scorer/detector.
- `cv2`, `numpy` нужны для image metrics.
- `QualityConfig`, `VideoMetadata` нужны для reject logic.

Constants:

- default LAION model/head settings;
- URL official LAION checkpoint;
- dimensions для LAION heads.

### Protocols: `AestheticScorer`, `TextDetector`

Это интерфейсы.

`AestheticScorer` должен иметь:

```python
score(frames) -> float | None
```

`TextDetector` должен иметь:

```python
text_area_ratio(frames) -> float | None
```

Зачем Protocol:

> Pipeline не должен знать, какой конкретно scorer используется. Ему важно только, что объект умеет считать score.

### Basic image metrics

Функции:

- `blur_score()`;
- `brightness_score()`;
- `contrast_score()`;
- `colorfulness_score()`;
- `estimate_text_area_ratio()`.

Смысл:

- blur через Laplacian variance;
- brightness через mean grayscale;
- contrast через standard deviation grayscale;
- colorfulness через разницу RGB channels;
- text proxy через Canny edges area.

### `aggregate_quality()`

Главная aggregator-функция quality модуля.

Вход:

```text
frames + optional aesthetic_scorer + optional text_detector
```

Выход:

```text
dict with blur, brightness, contrast, colorfulness, ocr_text_area_ratio, aesthetic_score
```

Логика:

1. Если кадров нет, вернуть все `None`.
2. Если есть real `text_detector`, использовать его.
3. Иначе использовать cheap proxy `estimate_text_area_ratio()`.
4. Посчитать median по кадрам для каждого score.
5. Если есть aesthetic scorer, посчитать aesthetic score.

Почему median:

> Median устойчивее к одному странному кадру, чем mean.

### `quality_reject_reasons()`

Решает, какие reject reasons добавить на основе metadata + quality scores.

Проверки:

- слишком короткое;
- слишком длинное;
- слишком маленькое разрешение;
- слишком blurry;
- слишком темное/яркое;
- слишком много текста/watermark;
- aesthetic ниже порога.

Эта функция не создает `ClipRecord`. Она только возвращает список причин.

### `HeuristicAestheticScorer`

Легкий CPU fallback.

Считает score из:

- brightness component;
- blur component;
- color component.

Это не LAION. Это быстрый smoke-test режим.

### `CLIPAestheticScorer`

CLIP prompt-comparison proxy.

Идея:

- сравнить кадр с prompt “high quality cinematic image”;
- сравнить с prompt “low quality blurry image...”;
- softmax probability превратить в score 0-10.

Это тоже не настоящий LAION, а proxy.

### `LAIONAestheticScorer`

Настоящий LAION-style scorer.

Поток:

```text
frames
  -> select evenly spaced frames
  -> PIL images
  -> open_clip preprocess
  -> open_clip image encoder
  -> normalize embeddings
  -> LAION linear head
  -> median score
```

Что важно:

- используется `open_clip`;
- checkpoint скачивается из official LAION aesthetic predictor repo;
- score image-level, поэтому threshold нужно калибровать.

### OCR classes

`EasyOCRTextDetector`:

- вызывает EasyOCR;
- получает boxes;
- считает площадь уверенно найденного текста.

`TesseractTextDetector`:

- вызывает pytesseract;
- берет word boxes;
- считает text area ratio.

### Factory functions

`build_text_detector(settings)`:

- читает provider из config;
- возвращает EasyOCR/Tesseract/None.

`build_aesthetic_scorer(settings)`:

- читает provider;
- возвращает Heuristic/CLIP/LAION/None.

Почему factory полезны:

> `pipeline.py` не должен знать детали создания LAION, EasyOCR или CLIP. Он просто просит factory построить нужный объект по config.

## 9. `motion.py`: движение во времени

Этот файл отвечает за temporal signal.

Основные идеи:

- видео - это не просто картинки;
- для generative video важно движение;
- motion score помогает отличать static clips от динамичных.

Главные функции:

### `motion_metrics(frames)`

Сравнивает соседние кадры.

Обычно используется optical flow / frame difference logic.

Возвращает:

- `motion_score`;
- `motion_p95_score`;
- `motion_stability_score`.

### `motion_caption(score)`

Переводит numeric score в текст:

- mostly static;
- moderate motion;
- fast/unstable motion.

Этот текст передается в `CaptionContext`, чтобы VLM prompt знал motion hint.

### `motion_reject_reasons(score, config)`

Добавляет reject reasons, если motion слишком низкий или слишком высокий.

## 10. `caption.py`: VLM captioning и caption quality control

Этот файл отвечает за текстовое описание видео.

### Constants и regex

- `GROQ_DEFAULT_VISION_MODEL` - default Groq vision model.
- `GROQ_MAX_KEYFRAMES` - лимит keyframes для Groq.
- regex для удаления `<think>`, `Final caption:`, analysis artifacts.
- regex для detection text overlay/watermark mentions.
- regex для incomplete caption endings.

### `CaptionContext`

Передает в captioner дополнительный контекст:

- clip id;
- source path;
- motion caption.

### Protocols: `Captioner`, `BatchCaptioner`

Интерфейс:

```python
caption(frames, context) -> str
```

Batch вариант:

```python
batch_caption(clips, contexts) -> list[str]
```

### `build_dense_caption_prompt()`

Создает prompt для VLM.

Сейчас он просит:

- один чистый paragraph;
- 35-90 words;
- без reasoning;
- без frame-by-frame analysis;
- без Markdown;
- включить subject/action/camera/scene/lighting/style/dynamics;
- если есть subtitles/logo/watermark/text overlay, mention briefly.

Почему mention overlay, а не скрывать:

> Если VLM видит overlay, нам важно узнать это и отклонить clip через `caption_reject_reasons()`.

### `sanitize_caption()`

Чистит ответ модели:

1. Удаляет `<think>...</think>`.
2. Удаляет одиночные `<think>` tags.
3. Если есть `Final caption:`, берет текст после него.
4. Убирает prefix `Caption:`.
5. Нормализует whitespace.

### `caption_reject_reasons()`

Новая важная функция.

Она добавляет reject reasons, если caption:

- пустой;
- содержит reasoning/Frame Analysis/The user wants;
- mention text overlay/logo/watermark/subtitles;
- обрывается на незаконченном слове вроде `with`.

Связь:

```text
pipeline.process_video()
  -> captioner.caption()
  -> caption_reject_reasons(caption)
  -> reject_reasons
```

### `select_keyframes()`

Выбирает равномерные кадры для VLM.

### `encode_frame_as_data_url()`

Кодирует frame в JPEG base64 data URL для OpenAI-compatible vision APIs.

### `build_openai_vision_messages()`

Собирает payload content:

```text
text prompt + image_url blocks
```

Используется Groq/OpenAI-compatible captioners.

### `build_vlm_messages()`

Собирает сообщения для local Transformers VLM:

- Qwen-style content blocks;
- LLaVA-style `<image>` tokens;
- fallback text prompt.

### `HeuristicCaptioner`

Fallback captioner.

Он не смотрит семантику, только brightness + motion hint. Используется для smoke tests, а не для serious VLM run.

### `CachedCaptioner`

Оборачивает captioner и кэширует captions.

Ключ:

- clip id, если есть context;
- иначе frame signature.

Зачем:

> VLM API стоит денег/времени. Если clip уже captioned, не надо повторять запрос.

### `OpenAICompatibleVisionCaptioner`

Базовый класс для hosted vision APIs.

Он:

1. Выбирает keyframes.
2. Собирает JSON payload.
3. Делает HTTP POST.
4. Парсит response.
5. Санитайзит caption.
6. Возвращает чистый текст.

### `OpenAIVisionCaptioner`

Наследник для OpenAI endpoint.

### `GroqVisionCaptioner`

Наследник для Groq endpoint.

Особенности:

- использует Groq base URL;
- token parameter называется `max_completion_tokens`;
- max keyframes ограничен `GROQ_MAX_KEYFRAMES`.

### `TransformersVLMCaptioner`

Локальная Hugging Face VLM версия.

Использует:

- `AutoProcessor`;
- `AutoModelForVision2Seq`;
- `generate()`;
- `batch_decode()`.

### `build_captioner(settings)`

Factory:

- `heuristic` -> `HeuristicCaptioner`;
- `openai` -> `OpenAIVisionCaptioner`;
- `groq` -> `GroqVisionCaptioner`;
- `transformers` -> `TransformersVLMCaptioner`;
- optional cache wrapper.

## 11. `duplicates.py`: perceptual hashing и dedupe

Этот файл отвечает за near-duplicates.

Типичная логика:

- взять frames;
- построить perceptual hash;
- сравнить hashes через Hamming distance;
- если distance ниже threshold, считать duplicate.

Главные функции:

### `clip_perceptual_hash(frames)`

Создает hash для клипа. Обычно берет representative frame(s).

### `hamming_distance(hash_a, hash_b)`

Считает отличие между двумя hashes.

### `mark_duplicates(records, threshold)`

Проходит по `ClipRecord` list и ставит:

```text
duplicate_of = existing_clip_id
reject_reasons += ['near_duplicate']
keep = false
```

Это вызывается:

- CLI command `dedupe-manifest`;
- Streamlit после upload processing.

## 12. `pipeline.py`: сердце обработки одного видео

Это главный файл для понимания runtime.

Разбор по блокам:

### Imports

`hashlib`, `Path`:

- нужны для stable clip id и путей.

`caption` imports:

- `CaptionContext` для передачи motion hint в VLM;
- `Captioner` для type hint;
- `build_captioner()` для создания captioner из config;
- `caption_reject_reasons()` для caption-based rejection.

`duplicates`:

- `clip_perceptual_hash()`.

`motion`:

- numeric metrics;
- motion text;
- motion reject logic.

`quality`:

- scorer/detector protocols;
- aggregate quality;
- factory functions;
- quality reject logic.

`schema`:

- `AppConfig`, `ClipRecord`.

`video_io`:

- `probe_video()`, `sample_frames()`.

### `stable_clip_id(path)`

Создает deterministic id:

```text
absolute path -> sha1 -> first 16 chars
```

Зачем:

- stable ID удобен для cache;
- manifest rows имеют короткий identifier.

### `process_video(...)`

Самая важная функция.

Пошагово:

1. Если captioner не передали, создать через `build_captioner(config.captioning)`.
2. Если aesthetic scorer не передали, создать через `build_aesthetic_scorer(config.aesthetic)`.
3. Если text detector не передали, создать через `build_text_detector(config.ocr)`.
4. `probe_video(path)` получает metadata.
5. `sample_frames(path, config.pipeline.sample_frames)` достает кадры.
6. `aggregate_quality()` считает quality metrics.
7. `motion_metrics()` считает motion metrics.
8. `motion_caption()` делает human-readable motion text.
9. `quality_reject_reasons()` добавляет quality reasons.
10. `motion_reject_reasons()` добавляет motion reasons.
11. `stable_clip_id()` создает clip id.
12. `CaptionContext` передает clip id/source/motion в captioner.
13. `captioner.caption()` получает VLM/heuristic caption.
14. `caption_reject_reasons()` добавляет caption reasons.
15. `clip_perceptual_hash()` считает pHash.
16. Создается `ClipRecord`.
17. `keep=not reasons`: если есть хотя бы одна причина reject, clip rejected.

Это главный invariant:

```text
reject_reasons пустой -> keep true
reject_reasons не пустой -> keep false
```

## 13. `manifest.py`: JSONL IO

Этот файл маленький, но центральный для обмена данными.

Главные функции:

### `write_jsonl(records, path)`

Пишет каждый `ClipRecord` как отдельную JSON строку.

### `read_jsonl(path)`

Читает JSONL и валидирует строки обратно в `ClipRecord`.

Почему Pydantic validation важен:

> Если manifest сломан или schema изменилась, ошибка появится сразу, а не во время training.

## 14. `reporting.py`: summaries и markdown reports

Этот файл превращает manifest в понятный отчет.

Обычно считает:

- total clips;
- accepted/rejected;
- acceptance rate;
- duplicates;
- average scores;
- reject reason counts.

Главные сущности:

- summary dataclass/model;
- `summarize_manifest(records)`;
- `render_markdown_summary(summary)`.

Кто использует:

- CLI `summarize-manifest`;
- Streamlit downloads;
- examples reports.

## 15. `scene_split.py`: scene detection и ffmpeg normalization

Этот файл работает до основного pipeline, если у тебя длинные raw videos.

Поток:

```text
raw long video
  -> PySceneDetect finds scene boundaries
  -> ffmpeg exports normalized clips
  -> SceneSegment records
```

Зачем normalization:

- одинаковый FPS;
- одинаковый размер;
- cleaner clips;
- меньше codec surprise.

Кто вызывает:

- CLI command `split-scenes`.

## 16. `ray_pipeline.py`: параллельная обработка

Этот файл нужен для scaling.

Идея:

```text
process_video(video_1)
process_video(video_2)
process_video(video_3)
```

можно запускать параллельно, потому что клипы независимы.

Ray wrapper делает remote tasks и собирает результаты.

Важно:

> Ray полезен на больших batch-ах. На маленьких папках overhead может быть больше пользы.

## 17. `benchmark_pipeline.py`: скорость preprocessing

Этот файл измеряет throughput pipeline.

Типичные метрики:

- total files;
- total seconds;
- clips per second;
- single-process vs Ray.

Зачем:

> Не просто сказать “pipeline scalable”, а измерить, где он быстрее/медленнее.

## 18. `benchmark_inference.py`: inference trade-offs

Этот файл про speed/memory benchmarking для diffusion inference.

Обычно поддерживает:

- dry run без тяжелой модели;
- real run через Diffusers;
- dtype/memory settings;
- latency;
- peak VRAM.

Это показывает понимание inference optimization, но не является core data pipeline.

## 19. `diffusion_finetune.py`: manifest -> Diffusers LoRA dataset

Этот файл связывает data pipeline с training.

Поток:

```text
manifest.jsonl
  -> read records
  -> choose clips
  -> sample frame(s)
  -> write images/
  -> write metadata.jsonl
```

Diffusers image-caption dataset ожидает примерно:

```json
{"file_name": "images/example_000.jpg", "text": "caption here"}
```

Зачем:

> Pipeline должен не только оценивать данные, но и превращать их в training-ready format.

Главные идеи:

- accepted records по умолчанию лучше;
- relaxed export можно использовать для smoke tests;
- captions идут в `metadata.jsonl`;
- images извлекаются из video frames.

## 20. `training_benchmark.py`: PyTorch GPU benchmark

Этот файл нужен, чтобы показать GPU/training mechanics.

Он не обучает production model. Он benchmark-ит training-like loop.

Обычно внутри:

- dataset from manifest or synthetic fallback;
- simple model/loss;
- mixed precision;
- throughput measurement;
- peak VRAM measurement;
- markdown/json report.

Зачем:

> Это доказывает, что ты умеешь запускать GPU experiments и измерять memory/throughput, а не только писать preprocessing.

## 21. `training_entrypoint.py`: запуск benchmark через Accelerate/DeepSpeed

Этот файл нужен как module entrypoint:

```bash
accelerate launch -m video_dataset_factory.training_entrypoint ...
```

Он парсит args, вызывает training benchmark и печатает/сохраняет результаты.

Связь:

```text
configs/accelerate_kaggle.yaml
configs/accelerate_deepspeed_zero2.yaml
  -> accelerate launch
  -> training_entrypoint.py
  -> training_benchmark.py
```

## 22. `cli.py`: все команды `vdf`

Это самый большой entrypoint-файл.

Он использует Typer.

Смысл Typer:

- Python functions превращаются в CLI commands;
- type hints становятся CLI validation;
- help генерируется автоматически.

Типичные команды:

### `split-scenes`

Вызывает `scene_split.py`.

### `process-video`

Вызывает `process_video()` для одного файла и пишет manifest.

### `process-folder`

Ищет видео в папке и обрабатывает каждое.

### `dedupe-manifest`

Читает manifest, вызывает `mark_duplicates()`, пишет новый manifest.

### `summarize-manifest`

Читает manifest, вызывает reporting, пишет markdown summary.

### `benchmark-folder`

Вызывает preprocessing benchmark.

### `benchmark-inference`

Вызывает inference benchmark.

### `benchmark-training`

Вызывает training benchmark.

### `prepare-diffusion-lora-data`

Вызывает diffusion export.

Главная роль `cli.py`:

> Собрать все модули проекта в один usable command-line tool.

## 23. `dashboards/app.py`: Streamlit UI

Этот файл делает web interface.

Основные части:

### Constants

- score columns;
- help text;
- video extensions;
- upload limits;
- default VLM model;
- LAION head URL.

### `normalize_records()`

Гарантирует, что dataframe имеет нужные columns.

### `get_secret_or_env()`

Берет secrets из environment или Streamlit secrets.

Используется для:

- `GROQ_API_KEY`;
- optional `GROQ_MODEL`.

### `apply_filters()`

Sidebar filters:

- all/accepted/rejected;
- reject reasons;
- caption/path search.

### `render_upload_mode()`

UI для загрузки видео:

- file uploader;
- duration slider;
- min aesthetic slider;
- pHash threshold;
- sample frames;
- VLM keyframes;
- Groq model input.

При нажатии кнопки вызывает `process_uploaded_files()`.

### `process_uploaded_files()`

Это web equivalent CLI processing.

Он:

1. Создает temp upload dir.
2. Создает `AppConfig()`.
3. Настраивает quality/captioning/ocr/aesthetic.
4. Создает `GroqVisionCaptioner`.
5. Создает OCR detector.
6. Создает LAION scorer.
7. Для каждого uploaded file вызывает `process_video()`.
8. После обработки вызывает `mark_duplicates()`.

### Render functions

- `render_summary()`;
- `render_reject_reasons()`;
- `render_score_distributions()`;
- `render_gallery()`;
- `render_clip_card()`;
- `render_downloads()`;
- `render_manifest_table()`.

### `main()`

Главная Streamlit entrypoint функция.

Поток:

```text
set page config
choose mode
load/process records
convert to dataframe
apply filters
render summaries/charts/gallery/table/downloads
```

## 24. Tests: что именно они защищают

### `tests/test_caption.py`

Проверяет:

- prompt содержит нужные instructions;
- `<think>` удаляется;
- `Final caption:` extraction работает;
- caption artifact rejection ловит reasoning/overlay/incomplete captions;
- keyframe selection;
- OpenAI/Groq payload shape;
- cache behavior.

Это защищает от твоей реальной ошибки, где manifest получил caption вида `The user wants... Frame Analysis...`.

### `tests/test_quality.py`

Проверяет:

- brightness/contrast/colorfulness;
- empty frames;
- text detector injection;
- EasyOCR bbox area ratio;
- unknown OCR provider;
- heuristic aesthetic score range;
- CLIP output compatibility helpers;
- LAION head path resolving/download behavior;
- low aesthetic rejection.

### `tests/test_motion.py`

Проверяет motion metrics и motion reject behavior.

### `tests/test_duplicates.py`

Проверяет pHash/Hamming distance/near duplicate marking.

### `tests/test_manifest.py`

Проверяет JSONL read/write roundtrip.

### `tests/test_reporting.py`

Проверяет summary counts и markdown rendering.

### `tests/test_scene_split.py`

Проверяет scene split helpers без тяжелого real video run.

### `tests/test_dashboard_smoke.py`

Проверяет, что Streamlit module можно импортировать/основные функции не ломаются.

### `tests/test_diffusion_finetune.py`

Проверяет manifest -> Diffusers dataset export.

### `tests/test_training_benchmark.py`

Проверяет training benchmark dry/synthetic path.

### `tests/test_benchmark_pipeline.py`

Проверяет preprocessing benchmark summary.

### `tests/test_benchmark_inference.py`

Проверяет inference benchmark dry-run logic.

## 25. Как файлы зависят друг от друга

Короткая карта:

```text
schema.py
  <- почти все файлы используют модели отсюда

config.py
  -> schema.AppConfig

video_io.py
  -> schema.VideoMetadata

quality.py
  -> schema.QualityConfig, schema.VideoMetadata

motion.py
  -> schema.QualityConfig

caption.py
  -> independent, uses frames + context

pipeline.py
  -> schema + video_io + quality + motion + caption + duplicates

manifest.py
  -> schema.ClipRecord

duplicates.py
  -> schema.ClipRecord

reporting.py
  -> schema.ClipRecord

cli.py
  -> config + pipeline + manifest + duplicates + reporting + benchmarks + diffusion export

dashboards/app.py
  -> schema + pipeline + caption + quality + duplicates + reporting

ray_pipeline.py
  -> pipeline

benchmark_pipeline.py
  -> pipeline/video_io/ray_pipeline

benchmark_inference.py
  -> optional diffusers/torch

diffusion_finetune.py
  -> manifest + video_io + schema

training_benchmark.py
  -> manifest/schema + torch/accelerate-like mechanics

training_entrypoint.py
  -> training_benchmark
```

## 26. Как читать проект как программист

Лучший порядок чтения:

1. `schema.py` - понять data models.
2. `configs/default.yaml` - понять runtime parameters.
3. `video_io.py` - понять, как видео превращается в frames/metadata.
4. `quality.py` - понять quality/OCR/LAION.
5. `motion.py` - понять temporal metrics.
6. `caption.py` - понять VLM captions и caption rejection.
7. `pipeline.py` - увидеть, как все собирается в `ClipRecord`.
8. `manifest.py` - понять JSONL format.
9. `duplicates.py` - понять dedupe.
10. `reporting.py` - понять summaries.
11. `cli.py` - понять user-facing commands.
12. `dashboards/app.py` - понять web UI.
13. `diffusion_finetune.py` - понять training export.
14. `training_benchmark.py` + `training_entrypoint.py` - понять GPU benchmark.
15. `tests/` - понять expected behavior.

## 27. Как дебажить конкретные проблемы

### Caption выглядит как reasoning

Смотреть:

- `caption.py`: `build_dense_caption_prompt()`;
- `caption.py`: `sanitize_caption()`;
- `caption.py`: `caption_reject_reasons()`;
- `tests/test_caption.py`.

### Видео accepted, хотя есть text overlay

Смотреть:

- `quality.py`: OCR/text detector;
- `caption.py`: `TEXT_ARTIFACT_RE`;
- `pipeline.py`: добавляется ли `caption_reject_reasons()`;
- `configs/default.yaml`: `max_ocr_text_area_ratio`.

### Слишком много rejected по aesthetic

Смотреть:

- `configs/default.yaml`: `quality.min_aesthetic_score`;
- `quality.py`: `LAIONAestheticScorer`;
- `dashboards/app.py`: min aesthetic slider.

LAION threshold нужно калибровать руками.

### Слишком много rejected по duration

Смотреть:

- `configs/default.yaml`: `quality.max_duration_sec`;
- `dashboards/app.py`: max duration slider.

### VLM API падает

Смотреть:

- `caption.py`: `GroqVisionCaptioner`;
- Streamlit secrets: `GROQ_API_KEY`;
- `GROQ_MAX_KEYFRAMES`;
- model name.

### Streamlit app падает при upload

Смотреть:

- `dashboards/app.py`: upload limits;
- secrets;
- installed dependencies;
- `packages.txt` for ffmpeg;
- OCR/LAION first-download behavior.

### Diffusers export пустой

Смотреть:

- manifest has `keep: true` records;
- `diffusion_finetune.py` filtering logic;
- relaxed/smoke settings;
- source video paths still exist.

## 28. Что говорить на интервью про архитектуру

Короткая инженерная версия:

> The core architecture is centered around a typed `ClipRecord` schema. `pipeline.process_video()` is the main orchestration function: it probes metadata, samples frames, computes quality/OCR/LAION and motion metrics, generates a VLM caption, applies caption quality rejection, computes perceptual hash, and returns a validated record. CLI, Streamlit, Ray, reporting, dedupe, and training export are separate layers built around that same manifest contract.

По-русски:

> Архитектура держится вокруг typed `ClipRecord`. Главная функция `process_video()` берет видео, достает metadata и frames, считает quality/OCR/LAION/motion, получает VLM caption, проверяет caption на артефакты, считает pHash и возвращает валидированную запись. CLI, Streamlit, Ray, reports, dedupe и training export - это отдельные слои вокруг одного manifest-контракта.

## 29. Самая важная мысль

Проект написан так, чтобы каждая часть была заменяемой:

- captioner можно заменить с Groq на local Transformers;
- OCR можно заменить с EasyOCR на Tesseract;
- aesthetic scorer можно заменить с LAION на heuristic/CLIP или другую модель;
- processing можно запускать через CLI, Streamlit или Ray;
- manifest можно использовать для reports, dedupe, dashboard, LoRA export или training benchmark.

Это и есть хороший research engineering design: **модули независимы, а контракт между ними стабильный**.
