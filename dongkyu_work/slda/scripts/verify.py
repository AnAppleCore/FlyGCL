"""CPU parity and actual main.py synthetic-stream checks for SLDA."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from dongkyu_work.analytic_common.tests.verify_heads import verify_method
from dongkyu_work.analytic_common.tests.verify_integration import verify_method as verify_integration
if __name__ == '__main__':
    verify_method('slda')
    verify_integration('slda')
