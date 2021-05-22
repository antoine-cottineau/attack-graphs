import numpy as np
from typing import Dict, List, Tuple


class RankingMethod:
    def __init__(self, ids_exploits: List[int]):
        self.ids_exploits = ids_exploits

    def rank_exploits(self) -> List[Tuple[int, float]]:
        scores: Dict[int, float] = {}

        # Evaluate the score when removing no exploit
        print("Applying ranking method without removing any exploit")
        scores[None] = self.get_score()

        # Evaluate the scores when removing one exploit
        for id_exploit in self.ids_exploits:
            print("Applying ranking method with exploit {} removed".format(
                id_exploit))
            scores[id_exploit] = self.get_score_with_exploit_removed(
                id_exploit)

        # Sort the exploits by looking at how removing them affect the score
        ordered_exploits = []
        argsort = np.argsort(list(scores.values()))
        for i in range(len(self.ids_exploits) + 1):
            id_exploit = list(scores)[argsort[i]]
            ordered_exploits.append((id_exploit, scores[id_exploit]))

        return ordered_exploits

    def get_score(self) -> float:
        return

    def get_score_with_exploit_removed(self, id_exploit: int) -> float:
        return
