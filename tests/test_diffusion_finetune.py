from pathlib import Path

from video_dataset_factory.diffusion_finetune import (
    DiffusionLoraCommandConfig,
    PreparedDiffusionDataset,
    _as_record_dict,
    build_diffusers_lora_command,
    build_diffusers_lora_shell_command,
    write_diffusion_lora_report,
    write_prepared_dataset_json,
)


class FakeClipRecord:
    def model_dump(self, mode="json"):
        return {"clip_id": "clip-1", "keep": True, "mode": mode}


def test_diffusers_lora_command_contains_core_arguments():
    command = build_diffusers_lora_command(
        DiffusionLoraCommandConfig(
            train_data_dir=Path("outputs/data"),
            output_dir=Path("outputs/lora"),
            max_train_steps=25,
            rank=4,
        )
    )

    assert command[:2] == ["accelerate", "launch"]
    assert "--train_data_dir=outputs/data" in command
    assert "--output_dir=outputs/lora" in command
    assert "--max_train_steps=25" in command
    assert "--rank=4" in command


def test_diffusers_lora_shell_command_quotes_arguments():
    command = build_diffusers_lora_shell_command(
        DiffusionLoraCommandConfig(train_script_path=Path("scripts/train lora.py"))
    )

    assert "'scripts/train lora.py'" in command or '"scripts/train lora.py"' in command


def test_diffusion_lora_export_accepts_pydantic_like_records():
    assert _as_record_dict(FakeClipRecord()) == {"clip_id": "clip-1", "keep": True, "mode": "json"}


def test_diffusion_lora_reports_are_written(tmp_path):
    dataset = PreparedDiffusionDataset(
        output_dir="outputs/diffusion_lora_dataset",
        metadata_path="outputs/diffusion_lora_dataset/metadata.jsonl",
        image_count=12,
        source_clip_count=6,
        skipped_clip_count=0,
        resolution=512,
        notes="test dataset",
    )
    json_path = tmp_path / "dataset.json"
    md_path = tmp_path / "plan.md"

    write_prepared_dataset_json(json_path, dataset)
    write_diffusion_lora_report(md_path, dataset, "accelerate launch train.py", "sd-test")

    assert '"image_count": 12' in json_path.read_text(encoding="utf-8")
    assert "Diffusion LoRA Fine-Tuning Plan" in md_path.read_text(encoding="utf-8")
    assert "accelerate launch train.py" in md_path.read_text(encoding="utf-8")
