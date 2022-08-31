import json
import sys

sys.path.append("./")
from util.toml import parse_toml

config = parse_toml("./config/config.toml")
degration_config = config["second-order-degradation"]
print(
    json.dumps(
        degration_config,
        indent=4,
        ensure_ascii=False,
        sort_keys=False,
        separators=(",", ":"),
    )
)
