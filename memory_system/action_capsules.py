from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re
from typing import Any
from urllib.parse import quote

from .action_ontology import ActionDraftPayload, classify_action_prompt, create_action_draft


@dataclass(frozen=True)
class ActionBridgeOption:
    option_id: str
    label: str
    kind: str
    url: str
    instructions: str = ""


@dataclass(frozen=True)
class ActionCapsule:
    prompt: str
    capsule_id: str
    action_id: str
    title: str
    domain: str
    risk_level: str
    authorization_level: str
    confirmation_required: bool
    auto_submit_allowed: bool
    status: str
    summary: str
    slots: dict[str, str] = field(default_factory=dict)
    draft_text: str = ""
    bridge_kind: str = ""
    bridge_url: str = ""
    bridge_label: str = ""
    bridge_options: list[ActionBridgeOption] = field(default_factory=list)
    bridge_instructions: str = ""
    verifier_requirements: list[str] = field(default_factory=list)
    auto_submit_blockers: list[str] = field(default_factory=list)
    residual_constraints: list[str] = field(default_factory=list)


def _capsule_id(action_id: str, prompt: str) -> str:
    digest = hashlib.sha1(f"{action_id}:{prompt}".encode("utf-8")).hexdigest()[:12]
    return f"{action_id}:{digest}"


def _clean_text(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _normalize_text(value: str) -> str:
    text = _clean_text(value).lower()
    text = re.sub(r"[^a-z0-9+$.\s-]+", " ", text)
    return " ".join(text.split())


def _title_text(value: str) -> str:
    text = _clean_text(value)
    replacements = {
        "doordash": "DoorDash",
        "uber eats": "Uber Eats",
        "uber": "Uber",
        "lyft": "Lyft",
    }
    for raw, pretty in replacements.items():
        text = re.sub(rf"\b{re.escape(raw)}\b", pretty, text, flags=re.IGNORECASE)
    return text


def _extract_between(prompt: str, start_pattern: str, stop_pattern: str) -> str:
    match = re.search(start_pattern + r"\s+(.+?)(?:" + stop_pattern + r"|$)", prompt, flags=re.IGNORECASE)
    if not match:
        return ""
    return _title_text(str(match.group(1) or "").strip(" .,"))


def _extract_food_slots(prompt: str) -> dict[str, str]:
    raw = _clean_text(prompt)
    normalized = _normalize_text(raw)
    slots: dict[str, str] = {}
    if "doordash" in normalized:
        slots["service"] = "DoorDash"
    elif "uber eats" in normalized:
        slots["service"] = "Uber Eats"
    else:
        slots["service"] = "food delivery"

    restaurant = _extract_between(raw, r"\bfrom\b", r"\b(?:with|make it|have|and|give|tip|for delivery|to)\b")
    if restaurant and restaurant.lower() not in {"doordash", "uber eats"}:
        slots["restaurant"] = restaurant

    item = _extract_between(raw, r"\b(?:get|order|want|buy|grab)\b", r"\b(?:from|with|make it|and|give|tip)\b")
    if item:
        item = re.sub(r"^(?:me|a|an|some)\s+", "", item, flags=re.IGNORECASE).strip()
    if not item:
        known_items = ["pizza", "burger", "sushi", "taco", "tacos", "burrito", "wings", "salad", "food"]
        for known_item in known_items:
            if re.search(rf"\b{re.escape(known_item)}\b", normalized):
                item = known_item
                break
    if item:
        slots["item"] = _title_text(item)

    modifiers = _extract_between(raw, r"\b(?:with|make it(?: have)?|have)\b", r"\b(?:and|give|tip)\b")
    if modifiers:
        slots["modifiers"] = modifiers

    tip_match = re.search(r"(?:\$([0-9]+(?:\.[0-9]{1,2})?)\s*(?:tip)?|tip(?:\s+the\s+dasher)?\s+\$?([0-9]+(?:\.[0-9]{1,2})?))", raw, flags=re.IGNORECASE)
    if tip_match:
        amount = tip_match.group(1) or tip_match.group(2)
        slots["tip"] = f"${amount}"

    return slots


def _extract_ride_slots(prompt: str) -> dict[str, str]:
    raw = _clean_text(prompt)
    slots: dict[str, str] = {}
    if re.search(r"\buber\b", raw, flags=re.IGNORECASE):
        slots["service"] = "Uber"
    elif re.search(r"\blyft\b", raw, flags=re.IGNORECASE):
        slots["service"] = "Lyft"
    else:
        slots["service"] = "ride share"
    destination = _extract_between(raw, r"\bto\b", r"\b(?:in|at|around|from|for)\b")
    if destination:
        slots["destination"] = destination
    time_match = re.search(r"\b(?:in\s+\d+\s+minutes?|now|today|tomorrow|at\s+\d+(?::\d+)?)\b", raw, flags=re.IGNORECASE)
    if time_match:
        slots["pickup_time"] = _title_text(time_match.group(0))
    return slots


def _slot_lines(slots: dict[str, str]) -> list[str]:
    return [f"{key.replace('_', ' ').title()}: {value}" for key, value in slots.items() if value]


def _food_draft_text(slots: dict[str, str]) -> str:
    lines = ["Food order capsule"]
    lines.extend(_slot_lines(slots))
    return "\n".join(lines)


def _ride_draft_text(slots: dict[str, str]) -> str:
    lines = ["Ride quote capsule"]
    lines.extend(_slot_lines(slots))
    return "\n".join(lines)


def _sms_bridge_url(draft: ActionDraftPayload) -> str:
    body = quote(draft.body or "", safe="")
    return f"sms:?&body={body}" if body else ""


def _food_search_query(*, restaurant: str, item: str) -> str:
    clean_restaurant = _clean_text(restaurant)
    clean_item = _clean_text(item)
    if clean_restaurant and clean_item:
        restaurant_tokens = set(_normalize_text(clean_restaurant).split())
        item_tokens = set(_normalize_text(clean_item).split())
        if item_tokens and item_tokens <= restaurant_tokens:
            return clean_restaurant
        return f"{clean_restaurant} {clean_item}"
    return clean_restaurant or clean_item or "food"


def _service_search_url(service: str, query: str) -> str:
    clean_query = quote(_clean_text(query), safe="")
    if service.lower() == "doordash":
        return f"https://www.doordash.com/search/store/{clean_query}/" if clean_query else "https://www.doordash.com/"
    if service.lower() == "uber eats":
        return f"https://www.ubereats.com/search?q={clean_query}" if clean_query else "https://www.ubereats.com/"
    if service.lower() == "uber":
        return "https://www.uber.com/us/en/start-riding/"
    if clean_query:
        return f"https://www.google.com/search?q={quote(service + ' ' + _clean_text(query), safe='')}"
    return ""


def _generic_web_search_url(service: str, query: str) -> str:
    search = _clean_text(" ".join(part for part in [service, query] if part))
    return f"https://www.google.com/search?q={quote(search, safe='')}" if search else "https://www.google.com/"


def _service_bridge_options(service: str, query: str) -> list[ActionBridgeOption]:
    url = _service_search_url(service, query)
    if not url:
        return []
    if service in {"DoorDash", "Uber Eats"}:
        return [
            ActionBridgeOption(
                option_id="service_app",
                label=f"Open {service} App",
                kind="universal_link",
                url=url,
                instructions="Let iOS route this through the installed service app if available.",
            ),
            ActionBridgeOption(
                option_id="service_web",
                label=f"Open {service} Web",
                kind="in_app_web",
                url=url,
                instructions="Open the same capsule bridge inside Memla Browser to keep the web path visible.",
            ),
            ActionBridgeOption(
                option_id="generic_web_search",
                label="Search Web",
                kind="in_app_web",
                url=_generic_web_search_url(service, query),
                instructions="Start from a general web search when the service URL drops part of the capsule intent.",
            ),
        ]
    return [
        ActionBridgeOption(
            option_id="service_web",
            label=f"Open {service}",
            kind="in_app_web",
            url=url,
            instructions="Open the capsule bridge inside Memla Browser.",
        )
    ]


def _message_capsule(prompt: str, draft: ActionDraftPayload) -> ActionCapsule:
    blockers = ["ios_messages_requires_user_send"]
    if draft.recipients:
        blockers.append("recipient_alias_not_bound_yet")
    return ActionCapsule(
        prompt=prompt,
        capsule_id=_capsule_id(draft.action_id, prompt),
        action_id=draft.action_id,
        title=draft.title,
        domain=draft.domain,
        risk_level=draft.risk_level,
        authorization_level="open_confirmation_screen" if draft.ok else "draft_only",
        confirmation_required=True,
        auto_submit_allowed=False,
        status="bridge_ready" if draft.ok else "needs_slots",
        summary="Message draft is ready. Memla can open Messages, then the user chooses the recipient and presses Send." if draft.ok else "Memla needs a recipient and message body before opening a message draft.",
        slots={
            "recipients": ", ".join(draft.recipients),
            "channel": draft.channel,
            "body": draft.body,
        },
        draft_text=draft.draft_text,
        bridge_kind="ios_sms_compose_body_only" if draft.ok else "",
        bridge_url=_sms_bridge_url(draft),
        bridge_label="Open Message Draft" if draft.ok else "",
        bridge_options=[
            ActionBridgeOption(
                option_id="ios_messages_body",
                label="Open Message Draft",
                kind="ios_messages",
                url=_sms_bridge_url(draft),
                instructions="User chooses the recipient and presses Send. Memla does not auto-send.",
            )
        ] if draft.ok else [],
        bridge_instructions="User chooses the recipient and presses Send. Memla does not auto-send.",
        verifier_requirements=["user_selected_recipient", "user_pressed_send"],
        auto_submit_blockers=blockers,
        residual_constraints=draft.residual_constraints,
    )


def _food_capsule(prompt: str, draft: ActionDraftPayload) -> ActionCapsule:
    slots = _extract_food_slots(prompt)
    service = slots.get("service", "food delivery")
    item = slots.get("item", "")
    restaurant = slots.get("restaurant", "")
    search_query = _food_search_query(restaurant=restaurant, item=item)
    bridge_options = _service_bridge_options(service, search_query)
    bridge_url = bridge_options[0].url if bridge_options else _service_search_url(service, search_query)
    blockers = [
        "consumer_ordering_bridge_not_implemented",
        "price_not_verified",
        "delivery_address_not_verified",
        "checkout_schema_not_verified",
        "payment_requires_user_confirmation",
    ]
    if not item:
        blockers.insert(0, "missing_item")
    if not restaurant:
        blockers.insert(0, "missing_restaurant")
    return ActionCapsule(
        prompt=prompt,
        capsule_id=_capsule_id(draft.action_id, prompt),
        action_id=draft.action_id,
        title=draft.title,
        domain=draft.domain,
        risk_level=draft.risk_level,
        authorization_level="open_confirmation_screen",
        confirmation_required=True,
        auto_submit_allowed=False,
        status="service_bridge_required",
        summary="Food order capsule prepared. Memla can structure the cart intent, but checkout must stay user-confirmed until the service bridge and price verifiers exist.",
        slots=slots,
        draft_text=_food_draft_text(slots),
        bridge_kind="commerce_bridge_options",
        bridge_url=bridge_url,
        bridge_label=f"Open {service}",
        bridge_options=bridge_options,
        bridge_instructions="Use the structured capsule to search/build the cart. Stop at checkout for user review and purchase confirmation.",
        verifier_requirements=["restaurant_match", "item_match", "modifier_match", "tip_match", "total_price_limit", "delivery_address_match", "user_checkout_confirmation"],
        auto_submit_blockers=blockers,
        residual_constraints=draft.residual_constraints,
    )


def _ride_capsule(prompt: str, draft: ActionDraftPayload) -> ActionCapsule:
    slots = _extract_ride_slots(prompt)
    service = slots.get("service", "ride share")
    bridge_options = _service_bridge_options(service, "")
    bridge_url = bridge_options[0].url if bridge_options else _service_search_url(service, "")
    blockers = [
        "ride_booking_bridge_not_implemented",
        "pickup_location_not_verified",
        "price_not_verified",
        "driver_eta_not_verified",
        "booking_requires_user_confirmation",
    ]
    if "destination" not in slots:
        blockers.insert(0, "missing_destination")
    return ActionCapsule(
        prompt=prompt,
        capsule_id=_capsule_id(draft.action_id, prompt),
        action_id=draft.action_id,
        title=draft.title,
        domain=draft.domain,
        risk_level=draft.risk_level,
        authorization_level="open_confirmation_screen",
        confirmation_required=True,
        auto_submit_allowed=False,
        status="service_bridge_required",
        summary="Ride capsule prepared. Memla can structure the ride intent, but booking must stay user-confirmed until pickup, price, ETA, and service bridge verifiers exist.",
        slots=slots,
        draft_text=_ride_draft_text(slots),
        bridge_kind="web_or_deeplink_bridge",
        bridge_url=bridge_url,
        bridge_label=f"Open {service}",
        bridge_options=bridge_options,
        bridge_instructions="Use the structured capsule to request a quote. Stop before booking for user review and confirmation.",
        verifier_requirements=["pickup_location_match", "destination_match", "pickup_time_match", "price_limit", "eta_acceptance", "user_booking_confirmation"],
        auto_submit_blockers=blockers,
        residual_constraints=draft.residual_constraints,
    )


def create_action_capsule(prompt: str) -> ActionCapsule:
    match = classify_action_prompt(prompt)
    draft = create_action_draft(prompt)
    if match.action_id in {"ask_contact", "draft_message", "send_email"}:
        return _message_capsule(prompt, draft)
    if match.action_id == "food_order_quote":
        return _food_capsule(prompt, draft)
    if match.action_id == "book_ride_quote":
        return _ride_capsule(prompt, draft)
    if match.action_id == "browser_scout":
        return ActionCapsule(
            prompt=prompt,
            capsule_id=_capsule_id(match.action_id, prompt),
            action_id=match.action_id,
            title=match.title,
            domain=match.domain,
            risk_level=match.risk_level,
            authorization_level="auto_execute",
            confirmation_required=False,
            auto_submit_allowed=False,
            status="implemented",
            summary="Browser scout is safe to run automatically because it only searches, reads, ranks, and reports.",
            slots={"goal": _clean_text(prompt)},
            draft_text="",
            bridge_kind="memla_runtime",
            bridge_url="",
            bridge_label="Run Scout",
            bridge_instructions="Execute through Memla's bounded browser scout runtime.",
            verifier_requirements=["result_cards_present", "best_match_ranked", "report_returned"],
            auto_submit_blockers=[],
            residual_constraints=match.residual_constraints,
        )
    return ActionCapsule(
        prompt=prompt,
        capsule_id=_capsule_id(match.action_id, prompt),
        action_id=match.action_id,
        title=match.title,
        domain=match.domain,
        risk_level=match.risk_level,
        authorization_level="draft_only",
        confirmation_required=match.confirmation_required,
        auto_submit_allowed=False,
        status=match.status,
        summary="Memla recognizes this action, but it still needs a safe bridge before execution.",
        slots={},
        draft_text=draft.draft_text,
        bridge_kind="",
        bridge_url="",
        bridge_label="",
        bridge_instructions="",
        verifier_requirements=[],
        auto_submit_blockers=["safe_bridge_not_implemented"],
        residual_constraints=match.residual_constraints,
    )


def action_capsule_to_dict(capsule: ActionCapsule) -> dict[str, Any]:
    return asdict(capsule)
