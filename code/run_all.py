"""
run_all.py  -- runs P4A -> P4B -> P5 sequentially, streams to log.
Usage:  python -u run_all.py
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.stderr = sys.stdout

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# load all stage functions
exec(open('paper_pipeline.py').read().split("if __name__")[0], globals())

run_p4a()
run_p4b()
run_p5()
