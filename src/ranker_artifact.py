"""Read a ranker artifact without resolving Python classes."""

import pickle


class RankerArtifactUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"랭커 파일에 지원하지 않는 객체가 있습니다: {module}.{name}")
