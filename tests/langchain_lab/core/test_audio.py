import unittest
from unittest import TestCase

from langchain_lab.core.audio import openai_speech_to_text


class TestAudio(TestCase):

    @unittest.skip("Skip test_transcriptions")
    def test_transcriptions(self):
        openai_speech_to_text("input.wav")


if __name__ == "__main__":
    unittest.main()
