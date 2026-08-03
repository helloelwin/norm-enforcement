"""Action dataclasses and parsing for multi-action agent outputs."""

import re
from dataclasses import dataclass, field
from typing import Optional


def truncate_by_words(text: str, max_words: int) -> str:
    """Truncate text to a maximum number of words.

    Args:
        text: The text to truncate.
        max_words: Maximum number of words allowed.

    Returns:
        Truncated text (preserves original spacing up to the word limit).
    """
    if not text or max_words <= 0:
        return text

    words = text.split()
    if len(words) <= max_words:
        return text

    # Walk through original text to find position after max_words words
    count = 0
    i = 0
    while count < max_words and i < len(text):
        # Skip whitespace
        while i < len(text) and text[i].isspace():
            i += 1
        # Skip word
        if i < len(text):
            while i < len(text) and not text[i].isspace():
                i += 1
            count += 1

    return text[:i]


@dataclass
class PostAction:
    """A public post action (for social media environment)."""

    content: str


@dataclass
class ResponseAction:
    """A response to a user request (for chatbot arena environment)."""

    content: str


@dataclass
class PrivateMessageAction:
    """A private message sent to another agent."""

    recipient_id: int
    content: str


@dataclass
class ReportAction:
    """A report filed against another agent."""

    target_id: int
    reason: str


@dataclass
class HarvestAction:
    """Harvest action for fishing environment."""

    amount: int


@dataclass
class PublicMessageAction:
    """A public message broadcast to all agents (fishing environment only)."""

    content: str


@dataclass
class AgentActionSet:
    """All actions an agent takes in one turn.

    For social media: post is required, others are optional.
    For chatbot arena: response is required, others are optional.
    For fishing: harvest is required, others are optional.
    """

    post: Optional[PostAction] = None
    response: Optional[ResponseAction] = None
    private_message: Optional[PrivateMessageAction] = None
    reports: list[ReportAction] = field(default_factory=list)
    harvest: Optional[HarvestAction] = None
    public_message: Optional[PublicMessageAction] = None


def parse_xml_tag_with_attr(text: str, tag: str, attr_name: str) -> tuple[Optional[str], Optional[str]]:
    """Parse XML tag content and a specific attribute value.

    Args:
        text: The text containing XML tags.
        tag: The XML tag to parse (without angle brackets).
        attr_name: The attribute name to extract.

    Returns:
        Tuple of (content, attr_value). Both are None if tag not found.
    """
    # Pattern matches <tag attr="value">content</tag> or <tag attr='value'>content</tag>
    # Accepts both bare integers ("3") and prefixed formats ("AGENT_3", "Agent 3", etc.)
    pattern = rf'<{tag}\s+{attr_name}=["\'](?i:agent[_ ]?)?(\d+)["\']>(.*?)</{tag}>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(2).strip(), match.group(1)
    return None, None


def parse_xml_tag(text: str, tag: str) -> Optional[str]:
    """Parse the content of a specific XML tag from the given text.

    Args:
        text: The text containing XML tags.
        tag: The XML tag to parse (without angle brackets).

    Returns:
        The content inside the specified XML tag, or None if not found.
    """
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def parse_post_action(text: str, max_words: int = 0) -> Optional[PostAction]:
    """Parse a post action from LLM output.

    Args:
        text: Raw LLM output text.
        max_words: Maximum words for post content (0 = no limit).
    """
    content = parse_xml_tag(text, "post")
    if content:
        if max_words > 0:
            content = truncate_by_words(content, max_words)
        return PostAction(content=content)
    return None


def parse_response_action(text: str, max_words: int = 0) -> Optional[ResponseAction]:
    """Parse a response action from LLM output.

    Args:
        text: Raw LLM output text.
        max_words: Maximum words for response content (0 = no limit).
    """
    content = parse_xml_tag(text, "response")
    if content:
        if max_words > 0:
            content = truncate_by_words(content, max_words)
        return ResponseAction(content=content)
    return None


def parse_private_message_action(text: str, max_words: int = 0) -> Optional[PrivateMessageAction]:
    """Parse a private message action from LLM output.

    Expected format: <private_message recipient="2">message content</private_message>

    Args:
        text: Raw LLM output text.
        max_words: Maximum words for message content (0 = no limit).
    """
    content, recipient_str = parse_xml_tag_with_attr(text, "private_message", "recipient")
    if content is not None and recipient_str is not None:
        try:
            recipient_id = int(recipient_str)
            if max_words > 0:
                content = truncate_by_words(content, max_words)
            return PrivateMessageAction(recipient_id=recipient_id, content=content)
        except ValueError:
            return None
    return None


def parse_report_actions(text: str) -> list[ReportAction]:
    """Parse all report actions from LLM output.

    Expected format: <report target="3">reason for report</report>
    Multiple reports are allowed, but only one per unique target (first wins).
    """
    # Accepts both bare integers ("3") and prefixed formats ("AGENT_3", "Agent 3", etc.)
    pattern = r'<report\s+target=["\'](?i:agent[_ ]?)?(\d+)["\']>(.*?)</report>'
    matches = re.findall(pattern, text, re.DOTALL)
    reports = []
    seen_targets: set[int] = set()
    for target_str, reason in matches:
        reason = reason.strip()
        if reason.lower().startswith("no report"):
            continue
        try:
            target_id = int(target_str)
        except ValueError:
            continue
        if target_id in seen_targets:
            continue
        seen_targets.add(target_id)
        reports.append(ReportAction(target_id=target_id, reason=reason))
    return reports


def parse_social_media_actions(
    text: str,
    max_post_words: int = 0,
    max_message_words: int = 0,
) -> AgentActionSet:
    """Parse all actions from social media agent output.

    Args:
        text: Raw LLM output text.
        max_post_words: Maximum words for post (0 = no limit).
        max_message_words: Maximum words for private message (0 = no limit).

    Returns:
        AgentActionSet with parsed actions.
    """
    return AgentActionSet(
        post=parse_post_action(text, max_words=max_post_words),
        response=None,
        private_message=parse_private_message_action(text, max_words=max_message_words),
        reports=parse_report_actions(text),
    )


def parse_chatbot_arena_actions(
    text: str,
    max_response_words: int = 0,
    max_message_words: int = 0,
) -> AgentActionSet:
    """Parse all actions from chatbot arena agent output.

    Args:
        text: Raw LLM output text.
        max_response_words: Maximum words for response (0 = no limit).
        max_message_words: Maximum words for private message (0 = no limit).

    Returns:
        AgentActionSet with parsed actions.
    """
    return AgentActionSet(
        post=None,
        response=parse_response_action(text, max_words=max_response_words),
        private_message=parse_private_message_action(text, max_words=max_message_words),
        reports=parse_report_actions(text),
    )


def parse_harvest_action(text: str, max_harvest: int = 0) -> Optional[HarvestAction]:
    """Parse harvest amount from <harvest>N</harvest>.

    Args:
        text: Raw LLM output text.
        max_harvest: Maximum harvest amount (0 = no limit).

    Returns:
        HarvestAction if found, None otherwise.
    """
    content = parse_xml_tag(text, "harvest")
    if content:
        try:
            amount = int(content.strip())
            if max_harvest > 0:
                amount = min(amount, max_harvest)
            amount = max(0, amount)  # Non-negative
            return HarvestAction(amount=amount)
        except ValueError:
            return None
    return None


def parse_public_message_action(text: str, max_words: int = 0) -> Optional[PublicMessageAction]:
    """Parse public message from <public_message>...</public_message>.

    Args:
        text: Raw LLM output text.
        max_words: Maximum words for message content (0 = no limit).

    Returns:
        PublicMessageAction if found, None otherwise.
    """
    content = parse_xml_tag(text, "public_message")
    if content:
        if max_words > 0:
            content = truncate_by_words(content, max_words)
        return PublicMessageAction(content=content)
    return None


def parse_fishing_actions(
    text: str,
    max_harvest: int = 0,
    max_message_words: int = 0,
) -> AgentActionSet:
    """Parse all actions from fishing environment agent output.

    Args:
        text: Raw LLM output text.
        max_harvest: Maximum harvest amount (0 = no limit).
        max_message_words: Maximum words for messages (0 = no limit).

    Returns:
        AgentActionSet with parsed actions.
    """
    return AgentActionSet(
        harvest=parse_harvest_action(text, max_harvest=max_harvest),
        public_message=parse_public_message_action(text, max_words=max_message_words),
        private_message=parse_private_message_action(text, max_words=max_message_words),
        reports=parse_report_actions(text),
    )


