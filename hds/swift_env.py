"""ms-swift gym plugin: `--external_plugins hds/swift_env.py --gym_env hds_companion`.

Env vars:
  HDS_ARM            reward arm A|B|C|D (default C)
  HDS_PLAYER_URLS    comma-separated simulator endpoints
  HDS_PLAYER_MODEL   served model name of the simulator
  HDS_EPISODE_LOG    optional jsonl path; one line of metrics per finished episode
  HDS_JUDGE_URLS     (arm D) comma-separated persona-judge endpoints; HDS_JUDGE_MODEL its name
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hds import character, judge, llm  # noqa: E402
from hds.episode import Episode  # noqa: E402
from hds.rewards import reward  # noqa: E402
from swift.rollout import Env, envs  # noqa: E402

ARM = os.environ.get("HDS_ARM", "C")
URLS = os.environ.get("HDS_PLAYER_URLS", "http://127.0.0.1:8101/v1").split(",")
MODEL = os.environ.get("HDS_PLAYER_MODEL", "q35")
LOG = os.environ.get("HDS_EPISODE_LOG")
JUDGE_URLS = os.environ.get("HDS_JUDGE_URLS", ",".join(URLS)).split(",")
JUDGE_MODEL = os.environ.get("HDS_JUDGE_MODEL", MODEL)
JUDGE_BLIND = os.environ.get("HDS_JUDGE_BLIND", "1" if os.environ.get("HDS_ARM") == "H" else "0") == "1"
_jrr = {"i": 0}


class CompanionEnv(Env):

    async def reset(self, config):
        spec = json.loads(config.data_dict["spec"])
        self.ep = Episode(spec, llm.player_fn(URLS, MODEL))
        obs = await self.ep.start()
        return obs, {"layout": spec["layout"]}, character.system_prompt()

    async def step(self, action):
        reply = action[-1]["content"] or ""
        obs, done = await self.ep.step(reply)
        if not done:
            return obs, 0.0, False, {}
        m = self.ep.metrics()
        persona = None
        if ARM in ("D", "E", "F", "G", "H") and not m["violation"]:  # judge only feasible episodes; violations are gated anyway
            url = JUDGE_URLS[_jrr["i"] % len(JUDGE_URLS)]
            _jrr["i"] += 1
            try:
                persona = (await judge.score(url, JUDGE_MODEL, self.ep.transcript, blind=JUDGE_BLIND))["persona"]
            except Exception:
                persona = None
        r = reward(m, ARM, persona)
        if LOG:
            with open(LOG, "a") as f:
                f.write(json.dumps({"t": time.time(), "arm": ARM, "reward": r, **{k: m[k] for k in (
                    "layout", "ptype", "solved", "quit", "turns", "violation", "leak", "invented", "wrong_fact",
                    "acts_for_player", "mean_reply_chars", "stages_solved", "stages", "tag_repeat", "copy_rate", "first_release", "max_unprompted_release")}, "persona": persona}, ensure_ascii=False) + "\n")
        return obs, r, True, {"solved": float(m["solved"]), "violation": float(m["violation"])}

    async def close(self):
        pass


envs["hds_companion"] = CompanionEnv
