"""Record a fresh 200-episode batch at the validated early-alignment standard."""

import json
from pathlib import Path

from usv_vla import batch_collect as bc
from usv_vla.collection_standard import PROFILE, apply_standard, inspect_standard
from usv_vla.scene_plan import validate_scene


def validate_episode(episode, **kwargs):
    proof = bc.validate_episode(episode, **kwargs)
    alignment = inspect_standard(episode)
    if not alignment["passed"]:
        raise ValueError("Trajectory smoothness failed: early alignment: "+"; ".join(alignment["issues"]))
    proof["navigation_standard"] = alignment
    bc._atomic_json(Path(episode)/"batch_acceptance.json", proof)
    return proof


def main():
    parser = bc.parser()
    parser.description = __doc__
    parser.set_defaults(output_root=bc.OUTPUTS/"episodes/red_round_200_20261027", stop_on_quality_failure=True)
    parser.add_argument("--source-root", type=Path, default=bc.OUTPUTS/"episodes/red_round_200_20261026")
    parser.add_argument("--planning-audit", type=Path,
                        default=bc.PROJECT_ROOT/"reports/early_alignment_planning_audit_20261027.json")
    args = parser.parse_args()
    if args.output_root.resolve() == args.source_root.resolve():
        raise ValueError("New collection output must differ from its source")
    existing = args.output_root/"manifest.json"
    if existing.exists():
        saved = bc._read_json(existing)
        for entry in saved["entries"]:
            plan = bc._read_json(args.output_root/entry["scene"])
            if plan.get("demonstration_standard", {}).get("profile") != PROFILE:
                raise ValueError("Cannot resume a batch without the current demonstration standard")
    def generator(category, index, seed):
        manifest = bc._read_json(args.source_root/"manifest.json")
        original = next(e for e in manifest["entries"] if e["category"] == category and e["category_index"] == index)
        if original["seed"] != seed:
            raise ValueError("Source scene seed mismatch")
        source = bc._read_json(args.source_root/original["scene"])
        params = {}
        if source["route"]["kind"] == "cylinder_pass":
            audit = bc._read_json(args.planning_audit)
            if audit.get("remaining_failures") or audit.get("planning_passed") != 60:
                raise ValueError("Planning audit is not complete")
            row = next(e for e in audit["entries"] if e["episode_id"] == original["episode_id"])
            if row.get("replacement_scene_json"):
                source = bc._read_json(row["replacement_scene_json"])
            params = {k: row[k] for k in ("control_forward_m", "lateral_m")}
            for key in ("maneuver_speed_mps", "start_backoff_m"):
                if key in row:
                    params[key] = row[key]
        scene = apply_standard(source, **params)
        scene["initial_visibility"] = validate_scene(scene)
        return scene
    return bc.collect(args, scene_generator=generator, validator=validate_episode)


if __name__ == "__main__":
    raise SystemExit(main())
