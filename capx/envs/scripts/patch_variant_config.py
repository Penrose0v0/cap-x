"""Write a per-variant copy of a task's env yaml for a sweep.

    python -m capx.envs.scripts.patch_variant_config SRC.yaml DST.yaml VARIANT TASK [--apis S3|<ApiClass>]

Sets `env.cfg.low_level.variation` (a "+"-joined VARIANT becomes a preset list, matching
resolve_variation_config), empties `api_servers` (sweeps share one set of vision/planning
services started separately), optionally overrides `env.cfg.apis`, and sets `env.cfg.prompt`
to the corrected text for presets with a known prompt/scene conflict (variant_prompts).
"""

from __future__ import annotations

import argparse

import yaml

from capx.envs.simulators.variant_prompts import resolve_prompt
from capx.envs.simulators.variants_common import S3_API_CLASSES, S3_API_DEFAULT


def patch(src: str, dst: str, variant: str, task: str, apis: str | None) -> None:
    variation = variant.split("+") if "+" in variant else variant
    with open(src) as f:
        data = yaml.safe_load(f)

    data["env"]["cfg"]["low_level"]["variation"] = variation
    data["api_servers"] = []
    if apis == "S3":
        data["env"]["cfg"]["apis"] = [S3_API_CLASSES.get(task, S3_API_DEFAULT)]
    elif apis:
        data["env"]["cfg"]["apis"] = [apis]

    corrected_prompt = resolve_prompt(task, variation)
    if corrected_prompt is not None:
        data["env"]["cfg"]["prompt"] = corrected_prompt

    with open(dst, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("variant")
    p.add_argument("task")
    p.add_argument("--apis", default=None, help='"S3" for the task\'s reduced API, or an explicit API class name')
    a = p.parse_args()
    patch(a.src, a.dst, a.variant, a.task, a.apis)


if __name__ == "__main__":
    main()
