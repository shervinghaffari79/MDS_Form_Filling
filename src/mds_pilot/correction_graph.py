"""Per-item correction graph: later sentences that may change an item's status are linked to the earlier mentions they follow."""
import re

from .graph import expand_with_followers

UPDATE_CUES = re.compile(r"\b(held|hold|holding|discontinu\w*|stopp\w*|resum\w*|restart\w*|ruled? out|no evidence|cancel\w*|error|incorrect\w*|"
                         r"revised|changed to|switched|confirmed|never|not (given|started|taking|continued|received|have|had)|does not|did not)\b", re.I)
FOLLOW_WINDOW = 2
INSTRUCTION = ("Read the sentences in order. A later sentence that says an earlier statement is wrong overrides it. A later sentence that only "
               "says the item stopped does not erase use earlier in this stay.")


def is_update(text):
    return bool(UPDATE_CUES.search(text))


def select_context(candidates, mention, update, sections):
    """Mask of candidates plus sentences that follow a candidate mention and look like status updates. All arrays cover one note."""
    near = expand_with_followers(candidates, mention, sections, FOLLOW_WINDOW)
    return candidates | (near & update)


def build_links(mention, update):
    """For the sentences in context, in reading order: {later position: earlier mention positions}. A link needs an update after a mention."""
    links = {}
    earlier = []
    for i in range(len(mention)):
        if update[i] and earlier:
            links[i] = list(earlier)
        if mention[i]:
            earlier.append(i)
    return links


def links_text(links):
    """links use 0-based positions; prompts number sentences from 1."""
    if not links:
        return "Linked updates: none."
    parts = [f"[{later + 1}] may update " + ", ".join(f"[{e + 1}]" for e in earlier) for later, earlier in links.items()]
    return "Linked updates: " + "; ".join(parts) + "."


def add_to_prompt(prompt, links=None):
    """Instruction alone when links is None, otherwise the instruction and the links."""
    extra = INSTRUCTION if links is None else f"{INSTRUCTION}\n{links_text(links)}"
    return f"{prompt}\n\n{extra}"
