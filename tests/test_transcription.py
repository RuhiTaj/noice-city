from voice_trainer.transcription import normalize_transcript, word_error_rate


def test_normalize_transcript() -> None:
    assert normalize_transcript("Hello,  WORLD!") == "hello world"


def test_word_error_rate() -> None:
    assert word_error_rate("one two three", "one two three") == 0.0
    assert word_error_rate("one two three", "one four three") == 1 / 3
