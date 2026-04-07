from memory_system.action_ontology import create_action_draft, classify_action_prompt, summarize_action_ontology


def test_action_ontology_classifies_browser_scout_as_implemented():
    match = classify_action_prompt("find the top 10 github repos for local llms and bring back the best one")

    assert match.action_id == "browser_scout"
    assert match.status == "implemented"
    assert match.confirmation_required is False
    assert match.safe_next_step == "execute"


def test_action_ontology_gates_messages_rides_and_food_orders():
    message = classify_action_prompt("imessage my sister and ask what she wants")
    ride = classify_action_prompt("get me an uber in 10 minutes to the airport")
    food = classify_action_prompt("order pizza on doordash")

    assert message.action_id in {"ask_contact", "draft_message"}
    assert message.confirmation_required is True
    assert message.safe_next_step == "draft_then_confirm"

    assert ride.action_id == "book_ride_quote"
    assert ride.confirmation_required is True
    assert "action_status:planned" in ride.residual_constraints

    assert food.action_id == "food_order_quote"
    assert food.confirmation_required is True
    assert "action_status:planned" in food.residual_constraints


def test_action_ontology_summary_exposes_safe_capabilities():
    summary = summarize_action_ontology()

    assert summary["action_count"] >= 6
    assert "messaging" in summary["domains"]
    assert summary["confirmation_required_count"] >= 4
    assert any(item["action_id"] == "browser_scout" for item in summary["capabilities"])


def test_action_ontology_v2_drafts_contact_message_without_sending():
    draft = create_action_draft("ask my sister what she wants from DoorDash")

    assert draft.ok is True
    assert draft.action_id == "ask_contact"
    assert draft.confirmation_required is True
    assert draft.recipients == ["Sister"]
    assert draft.body == "What do you want from DoorDash?"
    assert draft.draft_text == "To Sister: What do you want from DoorDash?"
    assert "confirmation_required" in draft.residual_constraints


def test_action_ontology_v2_keeps_service_actions_bridge_gated():
    draft = create_action_draft("get me an uber in 10 minutes to the airport")

    assert draft.ok is False
    assert draft.action_id == "book_ride_quote"
    assert "service_bridge_required" in draft.residual_constraints
    assert draft.safe_next_step == "design_or_bridge_required"
