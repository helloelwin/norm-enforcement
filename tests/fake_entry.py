"""Run the coupled runner or the coupled labeler with the fake model, for dry runs of job tooling (no GPU, no API).

    PYTHONPATH=<tree> python tests/fake_entry.py coupled <coupled.py arguments>
    PYTHONPATH=<tree> python tests/fake_entry.py label <evaluate_coupled arguments>
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_llm  # noqa: E402


def main():
    which, *argv = sys.argv[1:]
    fake_llm.install(mode="random", seed=abs(hash(tuple(argv))) % 10_000)
    if which == "coupled":
        from norm_enforcement import coupled
        asyncio.run(coupled.main_async(argv))
    else:
        import chz
        from norm_enforcement.evaluation import evaluate_coupled as ec
        asyncio.run(ec.main_async(chz.entrypoint(ec.Config, argv=argv)))


if __name__ == "__main__":
    main()
