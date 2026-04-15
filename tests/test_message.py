from uuid import UUID

from wanxiang.core.message import Message, MessageStatus


def test_message_defaults_and_auto_fields() -> None:
    msg = Message(
        intent="Write a short draft about AI-native workflows.",
        content="Draft v1",
        sender="writer",
    )

    # UUID should be generated and valid.
    assert UUID(msg.id)
    assert msg.parent_id is None
    assert msg.turn == 1
    assert msg.status == MessageStatus.SUCCESS
    assert msg.context == []
    assert msg.metadata == {}
    assert msg.timestamp is not None


def test_create_reply_links_parent_and_accumulates_context() -> None:
    writer_msg = Message(
        intent="Produce first draft.",
        content="Draft v1",
        sender="writer",
    )

    review_msg = writer_msg.create_reply(
        intent="Review logic, depth, and readability.",
        content="Needs revision: add concrete examples.",
        sender="reviewer",
        status=MessageStatus.NEEDS_REVISION,
        metadata={"score": 0.62},
    )

    assert review_msg.parent_id == writer_msg.id
    assert review_msg.turn == 2
    assert review_msg.context == ["Draft v1"]
    assert review_msg.metadata == {"score": 0.62}


def test_three_round_reply_chain_tracks_context_and_turn() -> None:
    m1 = Message(
        intent="Write a technical article draft.",
        content="Draft v1",
        sender="writer",
    )
    m2 = m1.create_reply(
        intent="Review and provide feedback.",
        content="Needs revision: expand section 2 and fix transitions.",
        sender="reviewer",
        status=MessageStatus.NEEDS_REVISION,
    )
    m3 = m2.create_reply(
        intent="Rewrite according to reviewer feedback.",
        content="Draft v2 with expanded section 2.",
        sender="writer",
        status=MessageStatus.SUCCESS,
    )
    m4 = m3.create_reply(
        intent="Final review for publication readiness.",
        content="Approved.",
        sender="reviewer",
        status=MessageStatus.SUCCESS,
    )

    assert m2.turn == 2
    assert m3.turn == 3
    assert m4.turn == 4

    assert m2.parent_id == m1.id
    assert m3.parent_id == m2.id
    assert m4.parent_id == m3.id

    assert m2.context == ["Draft v1"]
    assert m3.context == ["Draft v1", "Needs revision: expand section 2 and fix transitions."]
    assert m4.context == [
        "Draft v1",
        "Needs revision: expand section 2 and fix transitions.",
        "Draft v2 with expanded section 2.",
    ]


def test_to_dict_contains_all_fields() -> None:
    msg = Message(
        intent="Review the draft.",
        content="Draft content",
        sender="reviewer",
        status=MessageStatus.ERROR,
        metadata={"reason": "timeout"},
        parent_id="parent-123",
        context=["Draft v1"],
        turn=3,
    )
    data = msg.to_dict()

    assert data["id"] == msg.id
    assert data["intent"] == "Review the draft."
    assert data["content"] == "Draft content"
    assert data["sender"] == "reviewer"
    assert data["status"] == "error"
    assert data["timestamp"] == msg.timestamp.isoformat()
    assert data["parent_id"] == "parent-123"
    assert data["context"] == ["Draft v1"]
    assert data["turn"] == 3
    assert data["metadata"] == {"reason": "timeout"}


def test_to_prompt_includes_intent_content_and_context() -> None:
    msg = Message(
        intent="Evaluate draft quality and return actionable fixes.",
        content="Draft v2",
        sender="reviewer",
        status=MessageStatus.NEEDS_REVISION,
        context=["Draft v1", "Needs better examples."],
        turn=2,
    )

    prompt = msg.to_prompt()

    assert "Intent:" in prompt
    assert "Evaluate draft quality and return actionable fixes." in prompt
    assert "Content:" in prompt
    assert "Draft v2" in prompt
    assert "Context:" in prompt
    assert "1. Draft v1" in prompt
    assert "2. Needs better examples." in prompt

