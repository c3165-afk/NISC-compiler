"""Run the suite without installing packages or changing the environment."""
import sys
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'src'))
# This workspace includes a Linux venv. Its pure-Python libraries are usable
# with the supplied Windows Python; do not execute its Linux binaries.
for site in sorted((root / '.compiler' / 'lib').glob('python*/site-packages')):
    sys.path.append(str(site))
suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_*.py')
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(not result.wasSuccessful())
