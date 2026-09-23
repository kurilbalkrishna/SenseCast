"""Write the API's OpenAPI spec to docs/02-design/openapi.json (the API contract)."""
import json
from pathlib import Path

from sensecast.api.app import create_app

out = Path(__file__).resolve().parents[1] / "docs" / "02-design" / "openapi.json"
out.write_text(json.dumps(create_app().openapi(), indent=2))
print(f"wrote {out}")
