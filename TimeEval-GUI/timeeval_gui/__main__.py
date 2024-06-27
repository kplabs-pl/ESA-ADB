import sys
from pathlib import Path

from streamlit import cli as stcli


index_path = str(Path(__file__).parent.absolute() / "🏠_Home.py")
sys.argv = ["streamlit", "run", index_path]
sys.exit(stcli.main())
