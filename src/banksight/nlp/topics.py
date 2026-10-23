"""Find the main topic of a complaint with the topic model saved by notebook 07."""


def find_topic(masked_text, topic_bundle):
    """Return (topic number, topic name). If the text has no known words, return (-1, 'unclassified')."""
    weights = topic_bundle["nmf"].transform(topic_bundle["vectorizer"].transform([masked_text]))
    if weights.sum() == 0:
        return -1, "unclassified"
    topic_number = int(weights.argmax())
    return topic_number, str(topic_bundle["topic_labels"][topic_number])
