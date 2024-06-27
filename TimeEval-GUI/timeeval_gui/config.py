import gutenTAG
from pathlib import Path

GUTENTAG_CONFIG_SCHEMA_ANOMALY_KIND_URL: str = f"https://github.com/HPI-Information-Systems/gutentag/raw/v{gutenTAG.__version__}/generation-config-schema/anomaly-kind.guten-tag-generation-config.schema.yaml"
TIMEEVAL_FILES_PATH: Path = Path("timeeval-files")

SKIP_DOCKER_PULL: bool = True
