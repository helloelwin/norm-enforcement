"""Run one environment with the fake model and label it with the release's labeler (used to compare code versions).

    PYTHONPATH=<tree to test> python tests/run_env.py <social|chatbot|fishing> <hash|random> <fake seed> <results_dir> [config args]

The norm_enforcement package comes from PYTHONPATH, so the same driver can run the release and the branch.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_llm  # noqa: E402


def main():
    env, mode, fake_seed, results_dir, *args = sys.argv[1:]
    fake_llm.install(mode=mode, seed=int(fake_seed))
    import chz
    from norm_enforcement.environments.chatbot import ChatbotEnvConfig, ChatbotEnvironment
    from norm_enforcement.environments.fishing import FishingEnvConfig, FishingEnvironment
    from norm_enforcement.environments.social import SocialEnvConfig, SocialEnvironment
    from norm_enforcement.evaluation import evaluate_violation as ev

    config_cls, env_cls = {"social": (SocialEnvConfig, SocialEnvironment), "chatbot": (ChatbotEnvConfig, ChatbotEnvironment),
                           "fishing": (FishingEnvConfig, FishingEnvironment)}[env]
    cfg = chz.entrypoint(config_cls, argv=args + [f"results_dir={results_dir}"])
    asyncio.run(env_cls(cfg).run())
    asyncio.run(ev.main_async(ev.Config(transcript_path=cfg.save_path, results_dir=results_dir)))
    print("package:", sys.modules["norm_enforcement"].__file__)


if __name__ == "__main__":
    main()
