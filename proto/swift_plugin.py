"""ms-swift gym plugin: companion env for the GRPO pipeline smoke test.

The policy plays the companion (Rin). Each env.step sends the companion reply to
a simulated player served by vLLM (OpenAI API), applies the player's action to
the verifiable puzzle, and returns the player's next message as observation.
Reward is accumulated per step and summed by swift's GYMScheduler.
"""
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import make_puzzle  # noqa: E402
from run_episodes import CHARACTER, PLAYER_SYS, PLAYER_TYPES, parse_json  # noqa: E402
from swift.rollout import Env, envs  # noqa: E402

PLAYER_URL = os.environ.get("PLAYER_URL", "http://127.0.0.1:8100/v1")
PLAYER_MODEL = os.environ.get("PLAYER_MODEL", "q9")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "8"))
_clients = {}


def client():
    # swift may run each rollout in a fresh event loop; httpx connections are
    # bound to the loop that opened them, so keep one client per running loop.
    loop = asyncio.get_running_loop()
    c = _clients.get(id(loop))
    if c is None or c[0] is not loop:
        c = (loop, AsyncOpenAI(base_url=PLAYER_URL, api_key="x", timeout=120))
        _clients[id(loop)] = c
    return c[1]


class CompanionEnv(Env):

    async def _player(self):
        r = await client().chat.completions.create(
            model=PLAYER_MODEL, messages=self.play_msgs, max_tokens=200, temperature=1.0,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        p = parse_json(r.choices[0].message.content or "")
        self.play_msgs.append({"role": "assistant", "content": json.dumps(p, ensure_ascii=False)})
        fb = ""
        if p["action"]:
            ok = self.pz.check(p["action"])
            if ok is None:
                fb = "（指令格式不对，没有反应。）"
            else:
                self.attempts += 1
                self.solved = bool(ok)
                fb = "（咔哒一声，门开了！）" if ok else "（没有反应，好像不对。）"
        self.last_fb = fb
        text = (p["say"] or "（玩家没说话）") + (f"\n[玩家操作：{p['action']}] {fb}" if p["action"] else "")
        return p, fb, text

    async def reset(self, config):
        d = config.data_dict
        self.pz = make_puzzle(d["kind"], int(d["seed"]))
        self.ptype = d["ptype"]
        self.turn, self.attempts, self.solved, self.leaked = 0, 0, False, False
        self.play_msgs = [
            {"role": "system", "content": PLAYER_SYS.format(ptype=PLAYER_TYPES[self.ptype], view=self.pz.player_view())},
            {"role": "user", "content": "（游戏开始。铃飘在你旁边。）"}]
        _, _, text = await self._player()
        system = CHARACTER + "\n\n当前场景（只有你知道）：" + self.pz.companion_view()
        return text, {"kind": d["kind"]}, system

    async def step(self, action):
        reply = action[-1]["content"] or ""
        self.turn += 1
        reward = 0.0
        if not self.leaked and self.pz.leaked(reply):
            self.leaked = True
            reward -= 1.0  # hint, don't solve
        reward -= 0.002 * max(0, len(reply) - 80)  # brevity budget for real-time play
        self.play_msgs.append({"role": "user", "content": f"铃：{reply}" + (f"\n{self.last_fb}" if self.last_fb else "")})
        p, fb, text = await self._player()
        done = self.solved or p["quit"] or self.turn >= MAX_TURNS
        if self.solved:
            reward += 1.0
        elif p["quit"]:
            reward -= 0.5
        info = {"solved": float(self.solved), "leaked": float(self.leaked), "quit": float(p["quit"])}
        return text, reward, done, info

    async def close(self):
        pass


envs["companion_env"] = CompanionEnv
