"""saywhat: voice input for Hermes that asks instead of guessing.

A local faster-whisper STT provider measures per-word confidence. When a dictated
message has words it is unsure of, the agent offers sentence-level then word-level
choices (via ``clarify``) before acting, and every answer teaches a personal
dictionary. A practice mode scores how clearly the user is understood.
"""

from . import hooks, schemas
from .provider import Recognizer, SaywhatTranscriptionProvider, transcribe_utterance
from .runtime import Runtime
from .tools import Tools

TOOLSET = "saywhat"


class _LazyState:
    """Opens ``ctx.state`` on first use, so registering touches no files."""

    def __init__(self, ctx):
        self._ctx = ctx

    def get(self, key, default=None):
        return self._ctx.state.get(key, default)

    def set(self, key, value):
        self._ctx.state.set(key, value)


def register(ctx, **kwargs):
    rt = Runtime(_LazyState(ctx), ctx.get_config)
    recognizer = Recognizer()
    ctx.register_transcription_provider(SaywhatTranscriptionProvider(rt, recognizer))

    ctx.register_hook("pre_transcription", hooks.make_pre_transcription(rt))
    ctx.register_hook("pre_llm_call", hooks.make_pre_llm_call(rt))
    ctx.register_hook("post_tool_call", hooks.make_post_tool_call(rt))

    tools = Tools(rt, lambda path: transcribe_utterance(rt, recognizer, path), ctx.dispatch_tool)
    ctx.register_tool(name="saywhat_practice", toolset=TOOLSET, schema=schemas.PRACTICE,
                      handler=tools.practice, emoji="🎙️")
    ctx.register_tool(name="saywhat_progress", toolset=TOOLSET, schema=schemas.PROGRESS,
                      handler=tools.progress, emoji="📈")
    ctx.register_tool(name="saywhat_dictionary", toolset=TOOLSET, schema=schemas.DICTIONARY,
                      handler=tools.dictionary, emoji="📖")
    ctx.register_command("saywhat", tools.command, description="saywhat status, practice and dictionary",
                         args_hint="[practice <sentence> | practice stop | teach <heard> = <term> | forget <term>]")
