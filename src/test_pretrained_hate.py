from transformers import pipeline

MODEL_ID = "AuricErgeson/hate-speech-detector"

print("Loading pretrained hate-speech model...")

classifier = pipeline(
    "text-classification",
    model=MODEL_ID,
    device=-1,  # Run on CPU
)

tests = [
    "People from that ethnic group are inferior and should not have equal rights.",
    "People of that religion should be expelled from our country.",
    "People of that caste should never have equal rights.",
    "People of that race should not be allowed to live in our neighbourhood.",
    "People from this religious community do not deserve the same rights as others.",
    "Our town would be better if people from that ethnic group were not allowed to settle here.",
    "People from that religion should never be trusted with positions of authority.",
    "Members of that caste should be kept out of important professions.",
    "People of that nationality should never be granted citizenship, regardless of their circumstances.",
    "People from that community should study and work separately from everyone else.",
    "You are a complete idiot.",
    "I enjoyed watching the movie yesterday.",
]

for text in tests:
    result = classifier(text, truncation=True, max_length=128)[0]

    print(f"\nText: {text}")
    print(f"Prediction: {result['label']}")
    print(f"Score: {result['score']:.2%}")