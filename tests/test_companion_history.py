from langchain_core.messages import AIMessage, HumanMessage

from src.agent.companion import _format_history
from src.server.graph import _messages_to_chat_history


def test_format_history_marks_previous_persona():
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "我在", "speaker": "小艾"},
    ]
    text = _format_history(history)
    assert "Alleys（小艾）" in text


def test_messages_to_chat_history_preserves_persona():
    messages = [
        HumanMessage(content="你觉得刘星怎么样"),
        AIMessage(
            content="机灵鬼一个",
            additional_kwargs={"persona_id": "du", "persona_name": "阿毒"},
        ),
    ]
    history = _messages_to_chat_history(messages)
    assert history[1]["speaker"] == "阿毒"
