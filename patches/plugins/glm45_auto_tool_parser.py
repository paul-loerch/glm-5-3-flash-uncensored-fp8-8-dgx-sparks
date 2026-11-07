# SPDX-License-Identifier: Apache-2.0
# glm45_auto_tool_parser.py — vLLM tool-parser plugin for the GLM-5.3-Flash TP=8 recipe (vLLM v0.32.0).
#
# Load with:  --tool-parser-plugin /opt/recipe-patches/plugins/glm45_auto_tool_parser.py --tool-call-parser glm45_auto
#
# Why: vLLM v0.32.0 (#56403) raised the GLM-4.7/GLM-5.x tool parser's default strict level to FUNCTION, so every
# request that carries `tools` with tool_choice "auto" is decoded under an xgrammar structural tag. On this recipe
# (MTP with 2 draft tokens at batch size 1 and 1 at 2-8, TP=8 over RoCE) that costs most of the decode speed on agent
# turns. Measured 2026-10-23 on the same 16 recorded agent turns: no grammar 79.9 tok/s median (AL 2.69); with the
# default grammar 9.2 tok/s median (AL 1.54; slow steps even where drafts are accepted).
#
# What: the same parser class with the pre-v0.32 default restored. A grammar is attached only when the client asks
# for one: tool_choice "required" or a named function, any tool with "strict": true, or parallel_tool_calls false.
# Output parsing is unchanged. From the vLLM release that contains #61328 on, the same behaviour is available without
# a plugin: pass --tool-strict-level auto explicitly.
from vllm.logger import init_logger
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.glm47_moe_tool_parser import Glm47MoeModelToolParser

logger = init_logger("vllm.recipe.glm45_auto_tool_parser")


class Glm45AutoToolParser(Glm47MoeModelToolParser):
    # None = no parser-specific floor: the server level (--tool-strict-level, default auto) applies as given.
    default_tool_strict_level = None


ToolParserManager.register_module(name="glm45_auto", module=Glm45AutoToolParser)
logger.info("[recipe-plugin] tool parser glm45_auto registered (GLM-4.7 parser, default strict level: none)")
