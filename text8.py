
import os
import urllib.request
import zipfile
from collections import Counter


# https://mattmahoney.net/dc/textdata.html
# https://huggingface.co/datasets/afmck/text8
class Text8:
    words: list[str]
    frequencies: Counter[str]
    word_to_id: dict[str, int]
    id_to_word: dict[int, str]

    _URL: str = "https://mattmahoney.net/dc"
    _FILE: str = "text8"

    def __init__(self, max_words: int | None = None):
        path = f"{self._FILE}.zip"
        if not os.path.exists(path):
            print("Downloading text8 dataset...")
            urllib.request.urlretrieve(f"{self._URL}/{path}", filename=path)

        with zipfile.ZipFile(path, "r") as zip_file:
            with zip_file.open(self._FILE) as file:
                text = file.read().decode("utf-8")
                self.words = text.split(maxsplit=max_words)[:-1] if max_words is not None else text.split()

        self.frequencies = Counter(self.words)
        self.word_to_id = {}
        self.id_to_word = {}
        for id, word in enumerate(self.frequencies.keys()):
            self.word_to_id[word] = id
            self.id_to_word[id] = word


    def tokenize(self, words: list[str]):
        return map(self.word_to_id.get, words)


    def detokenize(self, ids: list[int]):
        return map(self.id_to_word.get, ids)
