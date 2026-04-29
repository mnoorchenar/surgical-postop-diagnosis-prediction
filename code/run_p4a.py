"""Run only Stage P4A — called directly so output streams live."""
import sys
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

exec(open('paper_pipeline.py').read().split("if __name__")[0])
run_p4a()
